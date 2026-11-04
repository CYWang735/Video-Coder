"""Run the teacher model in mini-swe-agent over a task file and store one trajectory per task.

Layout of an output directory:
    manifest.json                 config/tasks hashes; a directory is bound to one config
    tasks/<task_id>/workspace/    the agent's cwd (main.py ends up here)
    tasks/<task_id>/traj.json     full mini-swe-agent trajectory (saved after every step)
    tasks/<task_id>/result.json   status, static check, call counts
    summary.json
"""

import os

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")

import argparse
import fcntl
import hashlib
import json
import logging
import re
import shutil
import socket
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template

from minisweagent import __version__ as mini_version
from minisweagent.environments import get_environment
from minisweagent.models import get_model

from video_coder_harness.agent import GuardedAgent
from video_coder_harness.blender_exec import execute
from video_coder_harness.judge import judge_video
from video_coder_harness.static_check import check_file
from video_coder_harness.tasks import Task, file_sha256, is_safe_task_id, load_tasks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = PROJECT_ROOT / "bin"
FINAL_STATUSES = {"accepted", "rejected"}
_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)
# Errors after which every further call would fail the same way.
FATAL_ERRORS = {"AuthenticationError", "NotFoundError", "PermissionDeniedError", "BudgetExhausted"}

logger = logging.getLogger("video_coder_harness")


class BudgetExhausted(RuntimeError):
    pass


class CallBudget:
    """Shared across worker threads; checked *before* each API call so no paid response is discarded."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self._lock = threading.Lock()

    def take(self) -> None:
        with self._lock:
            if self.limit and self.used >= self.limit:
                raise BudgetExhausted(f"call budget of {self.limit} exhausted")
            self.used += 1


class BudgetedModel:
    """Charges every API request against the shared budget, including mini-swe-agent's internal retries.

    LitellmModel.query retries `_query` (up to MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT times) on transient errors;
    wrapping `_query` counts each of those HTTP requests. BudgetExhausted is added to the model's abort list so the
    retry loop stops at once instead of backing off and retrying a refused request.
    """

    def __init__(self, model, budget: CallBudget):
        self._model = model
        self._budget = budget
        self.requests = 0
        inner = getattr(model, "_query", None)
        self._wraps_inner = callable(inner)
        if self._wraps_inner:
            def counted_query(*args, **kwargs):
                self._charge()
                return inner(*args, **kwargs)

            model._query = counted_query
            if isinstance(getattr(model, "abort_exceptions", None), list):
                model.abort_exceptions = [*model.abort_exceptions, BudgetExhausted]

    def _charge(self) -> None:
        self._budget.take()
        self.requests += 1

    def query(self, messages, **kwargs):
        if not self._wraps_inner:  # models without an HTTP layer (deterministic test models)
            self._charge()
        return self._model.query(messages, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


def load_config(path: str | Path, overlays: list[str | Path] = ()) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for overlay in overlays:
        cfg = deep_merge(cfg, yaml.safe_load(Path(overlay).read_text()))
    cfg["model"] = _expand_env(cfg["model"])  # e.g. api_base: ${VLLM_API_BASE}; prompts are left untouched
    if cfg.get("harness", {}).get("judge"):
        cfg["harness"]["judge"] = _expand_env(cfg["harness"]["judge"])
    return cfg


def _expand_env(value):
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, str) and "${" in value:
        expanded = os.path.expandvars(value)
        if "${" in expanded:
            sys.exit(f"unset environment variable in config value: {value}")
        return expanded
    return value


# litellm provider prefix -> environment variable that holds the API key (read by litellm itself, never by us)
API_KEY_ENV = {"openai/": "OPENAI_API_KEY", "moonshot/": "MOONSHOT_API_KEY"}


def missing_api_key(model_name: str) -> str:
    """Name of the unset API-key variable for this model, or "" (unknown providers and local servers are not checked)."""
    for prefix, var in API_KEY_ENV.items():
        if model_name.startswith(prefix) and not os.environ.get(var):
            return var
    return ""


def deep_merge(base: dict, override: dict) -> dict:
    """Nested dicts merge key by key; any other value in `override` replaces the base value."""
    out = dict(base)
    for k, v in override.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def config_hash(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def harness_markers(harness: dict) -> list[str]:
    """Paths of the harness itself; a command mentioning one is blocked (see video_coder_harness.agent)."""
    bpy_env_dir = str(Path(os.path.expandvars(harness["bpy_python"])).parent.parent)
    return [str(PROJECT_ROOT / "video_coder_harness"), str(PROJECT_ROOT / "bin"), str(PROJECT_ROOT / "configs"),
            bpy_env_dir, os.path.expandvars(harness["bpy_libs"])]


def agent_env_vars(cfg_env: dict, harness: dict) -> dict:
    """Env for agent subshells: our bin/ on PATH, and credentials blanked so they cannot leak into trajectories."""
    env = {k: "" for k in os.environ if _SECRET_ENV.search(k)}
    env |= dict(cfg_env)
    env["PATH"] = f"{BIN_DIR}:{os.environ.get('PATH', '')}"
    env["VCSFT_PYTHON"] = sys.executable
    env["VCSFT_HARNESS"] = json.dumps(harness)  # read by bin/run_blender
    return env


def template_vars(cfg: dict) -> dict:
    h = cfg["harness"]
    return {
        "blender_version": h["blender_version"] or "<unset>",
        "blender_available": h["blender_available"],
        "fps": h["fps"],
        "frame_count": h["frame_count"],
        "max_frame_count": h.get("max_frame_count", h["frame_count"]),
        "resolution": h["resolution"],
        "script_name": h["script_name"],
        "output_video": h["output_video"],
        "visual_feedback": bool(h.get("visual_feedback")),
        "preview_frames": h.get("preview_frames", 8),
        "judge_enabled": bool((h.get("judge") or {}).get("enabled")),
    }


def render_prompts(cfg: dict, task: Task) -> dict:
    tv = template_vars(cfg) | {"task": task.prompt}
    return {
        name: Template(cfg["agent"][name], undefined=StrictUndefined).render(**tv)
        for name in ("system_template", "instance_template")
    }


def is_model_turn(m: dict) -> bool:
    """Chat Completions assistant message or Responses API response object."""
    return m.get("role") == "assistant" or m.get("object") == "response"


def count_format_errors(messages: list[dict]) -> int:
    return sum(m.get("extra", {}).get("interrupt_type") == "FormatError" for m in messages)


def task_dir_for(run_dir: Path, task_id: str) -> Path:
    """<run>/tasks/<task_id>, refusing ids that would point anywhere else (e.g. `..`)."""
    tasks_root = (run_dir / "tasks").resolve()
    task_dir = run_dir / "tasks" / task_id
    if not is_safe_task_id(task_id) or task_dir.resolve().parent != tasks_root:
        raise ValueError(f"unsafe task id {task_id!r}: its directory would be outside {tasks_root}")
    return task_dir


def run_task(task: Task, cfg: dict, run_dir: Path, budget: CallBudget) -> dict:
    task_dir = task_dir_for(run_dir, task.id)
    workspace = task_dir / "workspace"
    if workspace.exists():  # a previous non-final attempt; start clean so leftovers cannot leak in
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True)
    traj_path = task_dir / "traj.json"
    prev = read_result(run_dir, task.id)
    attempt = (prev or {}).get("attempt", 0) + 1
    if traj_path.is_file():  # keep earlier attempts for auditing
        traj_path.rename(task_dir / f"traj.attempt-{attempt - 1}.json")
    (task_dir / "task.json").write_text(json.dumps({"id": task.id, "prompt": task.prompt, "meta": task.meta}, ensure_ascii=False, indent=2))

    env_cfg = dict(cfg["environment"])
    env_cfg["cwd"] = str(workspace)
    env_cfg["env"] = agent_env_vars(env_cfg.get("env", {}), cfg["harness"])
    env = get_environment(env_cfg, default_type="local")
    model = BudgetedModel(get_model(config=dict(cfg["model"])), budget)
    agent = GuardedAgent(model, env, **cfg["agent"], harness_markers=harness_markers(cfg["harness"]),
                         workdir=str(workspace.resolve()), output_path=traj_path)

    started = time.time()
    error = None
    try:
        info = agent.run(task=task.prompt, **template_vars(cfg))
        exit_status = info.get("exit_status", "")
    except Exception as e:  # agent already saved the trajectory with the traceback
        exit_status, error = type(e).__name__, str(e)

    n_calls = agent.n_calls - (exit_status == "BudgetExhausted")  # the refused call was counted but never sent
    script = workspace / cfg["harness"]["script_name"]
    n_format_errors = count_format_errors(agent.messages)
    check = {"ok": False, "errors": [], "warnings": [], "stats": {}}
    execution = judgement = None
    try:  # a harness failure here (render timeout, ffmpeg, disk) must not lose the result or its API calls
        check = check_file(script)
        if error is None and exit_status == "Submitted" and check["ok"] and cfg["harness"]["blender_available"]:
            execution = final_execution(script, task_dir, cfg["harness"])
            judge_cfg = cfg["harness"].get("judge") or {}
            if execution["ok"] and judge_cfg.get("enabled"):
                judgement = run_judge(task, task_dir, execution, judge_cfg, budget)
    except Exception as e:
        logger.exception("task %s: harness error after the agent finished", task.id)
        error, exit_status = f"{type(e).__name__}: {e}", "HarnessError"
    judge_ok = judgement is None or (judgement["verdict"] == "pass"
                                     and (judgement.get("score") or 0) >= judgement_min_score(cfg))
    if judgement is not None and judgement["verdict"] == "error":
        error, exit_status = judgement["error"], judgement.get("exit_status", "JudgeError")
    if error is not None:
        status = "budget_exhausted" if exit_status == "BudgetExhausted" else "error"
    elif exit_status == "Submitted" and check["ok"] and (execution is None or execution["ok"]) and judge_ok:
        status = "accepted"
    else:
        status = "rejected"
    judge_calls = (judgement or {}).get("n_calls", 0)
    result = {
        "task_id": task.id,
        "status": status,
        "exit_status": exit_status,
        "error": error,
        "static_check": check,
        "execution": execution,
        "judge": judgement,
        "verification": ("blender_render+judge" if judgement is not None else "blender_render")
                        if cfg["harness"]["blender_available"] else "static_only",
        "attempt": attempt,
        "n_calls": n_calls,
        "n_requests": model.requests,
        "judge_calls": judge_calls,
        "n_calls_total": (prev or {}).get("n_calls_total", 0) + model.requests + judge_calls,
        "n_steps": sum(is_model_turn(m) for m in agent.messages),
        "n_format_errors": n_format_errors,
        "n_blocked": agent.n_blocked,
        "max_repeats": agent.max_repeats,
        "cost": agent.cost,
        "model_name": cfg["model"]["model_name"],
        "seconds": round(time.time() - started, 1),
    }
    if script.is_file():
        shutil.copy(script, task_dir / "final_script.py")
    (task_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def crash_result(run_dir: Path, task: Task, exc: Exception, cfg: dict) -> dict:
    """Result for a task whose run_task raised (e.g. while preparing its workspace); retried on resume."""
    prev = read_result(run_dir, task.id) or {}
    result = {
        "task_id": task.id, "status": "error", "exit_status": "HarnessError", "error": f"{type(exc).__name__}: {exc}",
        "static_check": {"ok": False, "errors": [], "warnings": [], "stats": {}}, "execution": None, "judge": None,
        "verification": "none", "attempt": prev.get("attempt", 0) + 1, "n_calls": 0, "n_requests": 0,
        "judge_calls": 0, "n_calls_total": prev.get("n_calls_total", 0), "n_steps": 0, "n_format_errors": 0,
        "n_blocked": 0, "max_repeats": 0, "cost": 0.0, "model_name": cfg["model"]["model_name"], "seconds": 0.0,
    }
    try:
        task_dir = task_dir_for(run_dir, task.id)
    except ValueError:
        return result  # nowhere safe to write it
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "result.json").write_text(json.dumps(result, indent=2))
    return result


def lock_run_dir(run_dir: Path):
    """Exclusive lock on the output directory; a second generate process on the same directory exits.

    Returns the open lock file, which must stay referenced for the lifetime of the process.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / ".lock"
    fd = open(path, "a+")
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fd.seek(0)
        holder = fd.read().strip() or "unknown process"
        fd.close()
        sys.exit(f"{run_dir} is in use by another video_coder_harness.generate process ({holder}); wait for it to finish or "
                 "use a different --output directory")
    fd.seek(0)
    fd.truncate()
    fd.write(f"pid {os.getpid()} on {socket.gethostname()} since {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    fd.flush()
    return fd


def judgement_min_score(cfg: dict) -> int:
    return (cfg["harness"].get("judge") or {}).get("min_score", 0)


def run_judge(task: Task, task_dir: Path, execution: dict, judge_cfg: dict, budget: CallBudget) -> dict:
    """VLM check that the accepted render shows the description; saved to final_run/judge.json."""
    video = execution["video"]
    try:
        judgement = judge_video(task.prompt, task_dir / "final_run" / "output.mp4", video["n_frames"],
                                video.get("fps") or 24, judge_cfg, take_call=budget.take)
    except BudgetExhausted as e:
        judgement = {"verdict": "error", "error": str(e), "exit_status": "BudgetExhausted", "n_calls": 0}
    except Exception as e:  # API failure: the task is retried on resume, not rejected
        judgement = {"verdict": "error", "error": f"{type(e).__name__}: {e}", "exit_status": "JudgeError", "n_calls": 1}
    (task_dir / "final_run" / "judge.json").write_text(json.dumps(judgement, indent=2, ensure_ascii=False))
    return judgement


def final_execution(script: Path, task_dir: Path, harness: dict) -> dict:
    """Re-run the submitted script from scratch in a fresh directory; keep its video next to the result."""
    run_dir = task_dir / "final_run"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir()
    shutil.copy(script, run_dir / harness["script_name"])
    report = execute(run_dir / harness["script_name"], run_dir, harness)
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    return {k: report.get(k) for k in ("ok", "problems", "error", "seconds", "video")}


def blender_preflight(harness: dict) -> None:
    """Fail before any paid API call if bpy/Mesa cannot render."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "preflight.py"
        w, h = harness["resolution"]
        script.write_text(PREFLIGHT_SCRIPT.format(frames=harness["frame_count"], w=w, h=h, fps=harness["fps"],
                                                  out=harness["output_video"]))
        small = dict(harness, resolution=[w // 10, h // 10])  # the preflight renders at 10%
        r = execute(script, tmp, small, preview=False)
    if not r["ok"]:
        sys.exit(f"Blender preflight failed: {r['problems']} {r.get('error') or ''}\n{r.get('traceback') or ''}")
    logger.info("Blender preflight ok (%.1fs)", r["seconds"])


PREFLIGHT_SCRIPT = """import bpy, os
s = bpy.context.scene
s.frame_start, s.frame_end = 1, {frames}
s.render.engine = 'BLENDER_WORKBENCH'
s.render.resolution_x, s.render.resolution_y, s.render.resolution_percentage = {w}, {h}, 10
s.render.fps = {fps}
s.render.image_settings.file_format = 'FFMPEG'
s.render.ffmpeg.format, s.render.ffmpeg.codec = 'MPEG4', 'H264'
s.render.filepath = os.path.join(os.getcwd(), '{out}')
bpy.ops.render.render(animation=True)
"""


def read_result(run_dir: Path, task_id: str) -> dict | None:
    if not is_safe_task_id(task_id):
        return None
    path = run_dir / "tasks" / task_id / "result.json"
    return json.loads(path.read_text()) if path.is_file() else None


def calls_used(run_dir: Path) -> int:
    return sum(json.loads(p.read_text()).get("n_calls_total", 0) for p in (run_dir / "tasks").glob("*/result.json"))


def write_manifest(run_dir: Path, cfg: dict, tasks_path: Path) -> None:
    manifest = {
        "config_sha256": config_hash(cfg),
        "tasks_file": str(tasks_path.resolve()),
        "tasks_sha256": file_sha256(tasks_path),
        "model_name": cfg["model"]["model_name"],
        "mini_swe_agent_version": mini_version,
        "config": cfg,
    }
    path = run_dir / "manifest.json"
    if path.is_file():
        old = json.loads(path.read_text())
        if old["config_sha256"] != manifest["config_sha256"]:
            sys.exit(f"{run_dir} was generated with a different config/model; use a new --output directory")
        if old["tasks_sha256"] != manifest["tasks_sha256"]:
            logger.warning("task file changed since this directory was created; existing results are kept by task id")
        manifest["created"] = old.get("created")
    manifest.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", required=True, help=".jsonl or .json task file")
    p.add_argument("--output", required=True, help="run directory (resumable)")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "teacher.yaml"))
    p.add_argument("--overlay", action="append", default=[],
                   help="YAML merged over --config (repeatable), e.g. to swap the model but keep the prompts")
    p.add_argument("--model", help="override model.model_name, e.g. openai/<model-id>")
    p.add_argument("--prompt-field", help="task field holding the scene description (auto-detected by default)")
    p.add_argument("--id-field", default="id")
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--limit", type=int, default=0, help="number of tasks to take after --offset (0 = all)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-calls", type=int, required=False, default=0,
                   help="total API call budget for this output directory, across resumes (0 = unlimited)")
    p.add_argument("--retry-rejected", action="store_true", help="also rerun tasks that finished as rejected")
    p.add_argument("--dry-run", action="store_true", help="render prompts to plan.json; no API calls")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config, args.overlay)
    if args.model:
        cfg["model"]["model_name"] = args.model
    tasks_path = Path(args.tasks)
    tasks = load_tasks(tasks_path, args.prompt_field, args.id_field)
    tasks = tasks[args.offset:]
    if args.limit:
        tasks = tasks[: args.limit]
    run_dir = Path(args.output)

    if args.dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        plan = {"n_tasks": len(tasks), "model_name": cfg["model"]["model_name"], "first_task_id": tasks[0].id if tasks else None,
                "prompts_for_first_task": render_prompts(cfg, tasks[0]) if tasks else None}
        (run_dir / "plan.json").write_text(json.dumps(plan, indent=2, ensure_ascii=False))
        print(f"dry run: {len(tasks)} tasks; rendered prompts in {run_dir / 'plan.json'}")
        return

    if not cfg["harness"]["blender_version"]:
        sys.exit("set harness.blender_version in the config (the Blender version the scripts must target)")
    if var := missing_api_key(cfg["model"]["model_name"]):
        sys.exit(f"{var} is not set")

    run_lock = lock_run_dir(run_dir)  # noqa: F841  (held until the process exits)
    if cfg["harness"]["blender_available"]:
        blender_preflight(cfg["harness"])
    write_manifest(run_dir, cfg, tasks_path)
    used = calls_used(run_dir)
    if args.max_calls and used >= args.max_calls:
        sys.exit(f"call budget already used: {used}/{args.max_calls}")
    budget = CallBudget(args.max_calls - used if args.max_calls else 0)

    todo = []
    for t in tasks:
        prev = read_result(run_dir, t.id)
        if prev and (prev["status"] == "accepted" or (prev["status"] == "rejected" and not args.retry_rejected)):
            continue
        todo.append(t)
    logger.info("%d tasks selected, %d to run, %d API calls used so far", len(tasks), len(todo), used)

    stop = threading.Event()

    def worker(t: Task) -> dict | None:
        if stop.is_set():
            return None
        try:
            r = run_task(t, cfg, run_dir, budget)
        except Exception as e:  # keep the other tasks running; this one is retried on resume
            logger.exception("task %s crashed", t.id)
            r = crash_result(run_dir, t, e, cfg)
        if r["exit_status"] in FATAL_ERRORS:
            stop.set()
            logger.error("stopping: %s: %s", r["exit_status"], r["error"])
        return r

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, t) for t in todo]
        for fut in as_completed(futures):
            r = fut.result()
            if r is None:
                continue
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            logger.info("%s %s (%s, %d calls)", r["task_id"], r["status"], r["exit_status"], r["n_calls"])

    all_status: dict[str, int] = {}
    for t in tasks:
        r = read_result(run_dir, t.id)
        s = r["status"] if r else "pending"
        all_status[s] = all_status.get(s, 0) + 1
    summary = {"this_run": counts, "all_selected_tasks": all_status, "api_calls_used_total": calls_used(run_dir)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
