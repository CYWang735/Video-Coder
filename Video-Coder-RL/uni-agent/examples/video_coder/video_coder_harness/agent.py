"""mini-swe-agent DefaultAgent with two guards for clean trajectories.

- Repetition: consecutive near-identical commands with the same result first get a warning appended to their
  output, and the run stops (exit status RepeatedActions) if the agent keeps repeating.
- Harness boundary: commands that inspect the harness (VCSFT_* variables, the implementation of check_script /
  run_blender, the bpy installation) or write outside the task directory are not executed; the agent gets a
  BLOCKED message instead. Blocked commands are counted so such trajectories can be left out of SFT data.
"""

import difflib
import re
import shlex

from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.exceptions import LimitsExceeded

TOOLS = ("run_blender", "check_script")
# Shell-level rules: applied to the command with here-document bodies removed, so the Python written into main.py
# (comments like "frame -> /tmp", "x > 0") is not mistaken for shell redirections or commands.
_INSPECT = re.compile(rf"^\s*(?:sudo\s+)?(?:type|which|command\s+-v|cat|less|more|head|tail|sed|awk|file|readlink|"
                      rf"realpath|ls|stat|strings|od|xxd)\b[^;&|]*\b({'|'.join(TOOLS)})\b")
_ENV_CMD = re.compile(r"(?m)(?:^|[;&|(]|\$\()\s*(?:env|printenv|set|export\s+-p|declare\s+-[px])\s*(?:$|[|;&>)])")
# writes to an absolute path (redirection, tee, cp/mv/ln target, mkdir/touch), except /dev/null and /dev/std*
_ABS_WRITE = re.compile(r"(?:(?<![0-9&<])>>?|\btee(?:\s+-a)?|\b(?:mkdir|touch)(?:\s+-p)?)\s*['\"]?"
                        r"(/(?!dev/(?:null|stdout|stderr)\b)[^\s'\";|&]*)")
_ABS_TARGET = re.compile(r"\b(?:cp|mv|ln|install|rsync)\b[^\n;|&]*\s['\"]?(/(?!dev/null\b)[^\s'\";|&]+)['\"]?\s*(?:$|[;|&\n])")
# Whole-text rules (here-document bodies included, so `python3 - <<EOF ... open(...)` is covered too):
# harness variables and locations that hold credentials or lie outside the task directory.
_HARNESS_VAR = re.compile(r"\bVCSFT_")
_SENSITIVE = re.compile(r"(~/|\$\{?HOME\b|/home/|/root/|\.ssh/|\.netrc\b|\.config/|\.env\b|\bOPENAI_API_KEY\b|"
                        r"\b[A-Z0-9_]*(?:API_KEY|SECRET|TOKEN|PASSWORD)\b|\bsk-[A-Za-z0-9_-]{8,})")
_HEREDOC = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")


def strip_heredocs(command: str) -> str:
    """The shell text of a command without here-document bodies (the delimiter lines are dropped too)."""
    lines, out, i = command.split("\n"), [], 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in _HEREDOC.finditer(line):
            while i < len(lines) and lines[i].strip() != m.group(2):
                i += 1
            i += 1  # the closing delimiter
    return "\n".join(out)


# Python code (also inside here-documents) that writes to an absolute path.
_PY_ABS_WRITE = [
    re.compile(r"""\bopen\(\s*[rbuf]*(['"])(/[^'"]*)\1\s*,\s*[rbuf]*(['"])[^'"]*[wax+][^'"]*\3"""),
    re.compile(r"""\bPath\(\s*(['"])(/[^'"]*)\1\s*\)\s*\.\s*(?:write_text|write_bytes|mkdir|touch|unlink|rename|replace)\b"""),
    re.compile(r"""\bos\.(?:makedirs|mkdir|rename|replace|remove|unlink|rmdir)\(\s*(['"])(/[^'"]*)\1"""),
    re.compile(r"""\bfilepath\s*=\s*[rbuf]*(['"])(/[^'"]*)\1"""),
]
_PARENT_DIR = re.compile(r"(?<![\w.])\.\./")
_SEPARATORS = set("();<>|&")


def in_dir(path: str, directory: str) -> bool:
    """True if `path` is `directory` itself or lies below it ('/w-other' is not inside '/w')."""
    d = directory.rstrip("/")
    return bool(d) and (path == d or path.startswith(d + "/"))


def _mask_workdir(text: str, workdir: str) -> str:
    """Replace the task directory by '.' in `text`, but not a longer sibling name such as '<workdir>-other'."""
    if not workdir:
        return text
    return re.sub(re.escape(workdir.rstrip("/")) + r"(?![A-Za-z0-9_.-])", ".", text)


def _is_separator(token: str | None) -> bool:
    return bool(token) and all(c in _SEPARATORS for c in token)


def _shell_path_violation(shell: str, workdir: str) -> str:
    """Rules on shell words: `..` path components and `cd` to somewhere outside the task directory."""
    text = shell.replace("\\\n", " ").replace("\n", " ; ")
    try:
        lex = shlex.shlex(text, posix=True, punctuation_chars=True)
        lex.whitespace_split = True
        tokens = list(lex)
    except ValueError:  # unbalanced quotes etc.: the regex rules still apply
        return ""
    for i, tok in enumerate(tokens):
        if tok == ".." or tok.startswith("../") or "/../" in tok or tok.endswith("/.."):
            return "it reaches outside the task directory with '..'"
        if tok in ("cd", "pushd") and (i == 0 or _is_separator(tokens[i - 1])):
            nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
            target = "" if _is_separator(nxt) else nxt
            if not target or target.startswith(("~", "$", "-")):
                return f"`{tok}` without a path inside the task directory leaves it"
            if target.startswith("/") and not target.startswith("/dev/") and not in_dir(target, workdir):
                return f"it changes into a directory outside the task directory ({target})"
    return ""


def harness_violation(command: str, harness_markers: list[str], workdir: str = "") -> str:
    """Reason why a command crosses the harness boundary, or "" if it is fine."""
    for marker in harness_markers:
        if marker and marker in command:
            return f"it references the harness installation ({marker})"
    if _HARNESS_VAR.search(command):
        return "it reads the harness environment variables"
    m = _SENSITIVE.search(_mask_workdir(command, workdir))  # the task directory itself is fine
    if m:
        return f"it references files or variables outside the task directory ({m.group(1)})"
    if _PARENT_DIR.search(command):
        return "it reaches outside the task directory with '../'"
    for pattern in _PY_ABS_WRITE:  # whole text: covers `python3 -c` and here-documents
        for m in pattern.finditer(command):
            path = m.group(2)
            if not (path.startswith("/dev/") or in_dir(path, workdir)):
                return f"it writes outside the task directory ({path})"
    shell = strip_heredocs(command)
    for line in shell.splitlines():
        if _INSPECT.search(line):
            return f"it inspects how {', '.join(TOOLS)} are implemented"
    if _ENV_CMD.search(shell):
        return "it reads the environment variables"
    for pattern in (_ABS_WRITE, _ABS_TARGET):
        for m in pattern.finditer(shell):
            if not in_dir(m.group(1), workdir):
                return f"it writes outside the task directory ({m.group(1)})"
    return _shell_path_violation(shell, workdir)


BLOCKED = ("BLOCKED: this command was not executed because {reason}. Work only in the current directory and use "
           "only check_script and run_blender; do not inspect or modify the harness or run Blender any other way. "
           "For a quick experiment, write a small script in the current directory and run `run_blender "
           "experiment.py`; its print() output appears in the blender log.")
REPEAT_WARNING = ("\n\nWARNING: this is the {n}th consecutive near-identical command with the same result. Repeating "
                  "it will not help: read the output, then change the script or your approach. The task ends after "
                  "{stop} repetitions.")


class GuardedAgentConfig(AgentConfig):
    repeat_warn: int = 3
    """Append a warning from the n-th consecutive near-identical command with an identical result."""
    repeat_stop: int = 6
    """Stop the run (exit status RepeatedActions) at the n-th consecutive repetition (0 = never)."""
    repeat_similarity: float = 0.9
    """Commands at least this similar (difflib ratio) count as the same."""
    harness_markers: list[str] = []
    """Strings (paths) whose appearance in a command means it touches the harness; filled in by video_coder_harness.generate."""
    workdir: str = ""
    """The task directory; absolute paths inside it may be written."""


def _normalize_output(text: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "#", text or "")


class GuardedAgent(DefaultAgent):
    def __init__(self, *args, config_class: type = GuardedAgentConfig, **kwargs):
        super().__init__(*args, config_class=config_class, **kwargs)
        self.n_blocked = 0
        self.max_repeats = 0
        self._last: tuple[str, str] | None = None
        self._repeats = 0

    def _execute(self, action: dict) -> dict:
        reason = harness_violation(action.get("command", ""), self.config.harness_markers, self.config.workdir)
        if reason:
            self.n_blocked += 1
            return {"output": BLOCKED.format(reason=reason), "returncode": 1, "exception_info": "",
                    "extra": {"blocked": reason}}
        return self.env.execute(action)

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        outputs = [self._execute(a) for a in actions]  # env.execute raises Submitted on the submit command
        stop = False
        for action, output in zip(actions, outputs):
            key = (action.get("command", ""), _normalize_output(output.get("output", "")))
            same = (self._last is not None and key[1] == self._last[1]
                    and difflib.SequenceMatcher(None, key[0], self._last[0]).ratio() >= self.config.repeat_similarity)
            self._repeats = self._repeats + 1 if same else 1
            self._last = key
            self.max_repeats = max(self.max_repeats, self._repeats)
            if self._repeats >= self.config.repeat_warn:
                output["output"] = (output.get("output") or "") + REPEAT_WARNING.format(
                    n=self._repeats, stop=self.config.repeat_stop or "never")
            if self.config.repeat_stop and self._repeats >= self.config.repeat_stop:
                stop = True
        added = self.add_messages(*self.model.format_observation_messages(message, outputs, self.get_template_vars()))
        if stop:
            raise LimitsExceeded({"role": "exit", "content": "RepeatedActions",
                                  "extra": {"exit_status": "RepeatedActions", "submission": ""}})
        return added
