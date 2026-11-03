"""Convert accepted mini-swe-agent trajectories into SFT jsonl.

Formats:
  openai  {"messages": [...OpenAI chat with tool_calls / tool...], "tools": [...], "meta": {...}}
  swift   ms-swift agent format: {"tools": "<json str>", "messages": [system/user/assistant/tool_call/tool_response], "images": [...]}
  code    final scripts only: {"task_id", "prompt", "code", "video", ...}, plus <out>/code/<split>/<task_id>.py
          and the harness-rendered <out>/videos/<split>/<task_id>.mp4

Images inside messages (future Blender preview frames, as OpenAI image_url parts) are written to
<out>/images/ and referenced by path; in swift format they become <image> placeholders.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

# The agent's task directory appears in tool outputs (pwd, ls -la, tracebacks); it becomes this placeholder.
WORKDIR = "/work"
_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)
_KEY_PATTERN = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}")
_PATH_PATTERN = re.compile(r"/home/[A-Za-z0-9_.-]+/|/root/|/hnvme/|/scratch/|/apps/")

BASH_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Execute a bash command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string", "description": "The bash command to execute"}},
            "required": ["command"],
        },
    },
}


class ImageStore:
    def __init__(self, out_dir: Path):
        self.dir = out_dir / "images"

    def save(self, url: str) -> str:
        """Return a file path for an image_url; data URIs are written out, other URLs/paths are kept."""
        if not url.startswith("data:"):
            return url
        header, b64 = url.split(",", 1)
        ext = header.split("/")[1].split(";")[0] if "/" in header else "png"
        data = base64.b64decode(b64)
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{hashlib.sha1(data).hexdigest()}.{ext}"
        if not path.exists():
            path.write_bytes(data)
        return str(path.resolve())


def _content_parts(content, images: ImageStore) -> tuple[str, list[str], list]:
    """Return (text with <image> placeholders, image paths, openai-style content with images externalized)."""
    if content is None:
        return "", [], ""
    if isinstance(content, str):
        return content, [], content
    text, paths, parts = [], [], []
    for part in content:
        if part.get("type") == "text":
            text.append(part["text"])
            parts.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "image_url":
            path = images.save(part["image_url"]["url"])
            text.append("<image>")
            paths.append(path)
            parts.append({"type": "image_url", "image_url": {"url": path}})
    return "".join(text), paths, parts


def _responses_content(parts: list[dict]) -> list[dict]:
    """Responses API input/output content parts -> chat content parts."""
    out = []
    for part in parts:
        if part.get("type") in ("input_text", "output_text", "text"):
            out.append({"type": "text", "text": part["text"]})
        elif part.get("type") == "input_image":
            out.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
    return out


def normalize(messages: list[dict]) -> list[dict]:
    """Map a trajectory from either model class to chat form (Responses API items become chat messages)."""
    out = []
    for m in messages:
        if m.get("object") == "response":
            text, calls, reasoning = [], [], []
            for item in m.get("output", []):
                if item.get("type") == "message":
                    text += [p["text"] for p in _responses_content(item.get("content", []))]
                elif item.get("type") == "function_call":
                    calls.append({"id": item["call_id"], "type": "function",
                                  "function": {"name": item["name"], "arguments": item["arguments"]}})
                elif item.get("type") == "reasoning":  # only summaries are readable; encrypted content is dropped
                    reasoning += [s["text"] for s in item.get("summary") or [] if s.get("text")]
            msg = {"role": "assistant", "content": "".join(text), "tool_calls": calls}
            if reasoning:
                msg["reasoning_content"] = "\n\n".join(reasoning)
            out.append(msg)
        elif m.get("type") == "function_call_output":
            output = m["output"]  # str, or [input_text, input_image, ...] when the tool attached images
            content = _responses_content(output) if isinstance(output, list) else output
            out.append({"role": "tool", "tool_call_id": m["call_id"], "content": content})
        elif m.get("type") == "message" and isinstance(m.get("content"), list):
            out.append({**m, "content": _responses_content(m["content"])})
        else:
            out.append(m)
    return out


def to_openai(messages: list[dict], images: ImageStore) -> list[dict]:
    out = []
    for m in normalize(messages):
        role = m.get("role")
        if role == "exit":
            continue
        _, _, content = _content_parts(m.get("content"), images)
        if role == "assistant":
            msg = {"role": "assistant", "content": content or ""}
            if m.get("reasoning_content"):
                msg["reasoning_content"] = m["reasoning_content"]
            if m.get("tool_calls"):
                msg["tool_calls"] = [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}}
                    for tc in m["tool_calls"]
                ]
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "tool_call_id": m["tool_call_id"], "content": content})
        else:
            out.append({"role": role, "content": content})
    return out


def to_swift(messages: list[dict], images: ImageStore) -> tuple[list[dict], list[str]]:
    out, all_images = [], []
    for m in normalize(messages):
        role = m.get("role")
        if role == "exit":
            continue
        text, paths, _ = _content_parts(m.get("content"), images)
        all_images += paths
        if role == "assistant":
            if m.get("reasoning_content"):
                text = f"<think>\n{m['reasoning_content'].strip()}\n</think>\n\n{text}"
            if text:
                out.append({"role": "assistant", "content": text})
            for tc in m.get("tool_calls") or []:
                call = {"name": tc["function"]["name"], "arguments": json.loads(tc["function"]["arguments"])}
                out.append({"role": "tool_call", "content": json.dumps(call, ensure_ascii=False)})
        elif role == "tool":
            out.append({"role": "tool_response", "content": text})
        else:
            out.append({"role": role, "content": text})
    return out, all_images


def sanitize_messages(messages: list[dict], task_dir: Path) -> list[dict]:
    """Replace the absolute task directory with WORKDIR everywhere in a trajectory's messages."""
    text = json.dumps(messages, ensure_ascii=False)
    workspace = task_dir / "workspace"
    for variant in sorted({str(workspace.resolve()), str(workspace.absolute())}, key=len, reverse=True):
        text = text.replace(variant, WORKDIR)
    return json.loads(text)


def secret_values() -> list[str]:
    """Credentials present in this process's environment (e.g. OPENAI_API_KEY) that must never be exported."""
    return [v for k, v in os.environ.items() if _SECRET_ENV.search(k) and len(v) >= 16]


def find_leak(record: dict, images_dir: Path) -> str:
    """'secret' or 'path' if the exported record still contains a credential or a cluster path, else ''."""
    data = {k: v for k, v in record.items() if k not in ("images", "video")}  # our own output file references
    text = json.dumps(data, ensure_ascii=False).replace(str(images_dir.resolve()), "<images>")
    if _KEY_PATTERN.search(text) or any(v in text for v in secret_values()):
        return "secret"
    if _PATH_PATTERN.search(text):
        return "path"
    return ""


def is_val(task_id: str, val_ratio: float) -> bool:
    return int(hashlib.sha1(task_id.encode()).hexdigest(), 16) % 10_000 < val_ratio * 10_000


def iter_accepted(run_dirs: list[Path], args, skipped: dict):
    seen = set()

    def skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for run_dir in run_dirs:
        for result_path in sorted((run_dir / "tasks").glob("*/result.json")):
            result = json.loads(result_path.read_text())
            task_dir = result_path.parent
            if args.task_id and result["task_id"] not in args.task_id:
                continue
            if result["status"] != "accepted":
                skip(f"status={result['status']}")
            elif result["task_id"] in seen:
                skip("duplicate_task_id")
            elif result["n_format_errors"] and not args.keep_format_errors:
                skip("format_errors")
            elif result.get("n_blocked") and not args.keep_blocked:
                skip("blocked_harness_commands")
            elif args.max_steps and result["n_steps"] > args.max_steps:
                skip("too_many_steps")
            elif args.no_warnings and result["static_check"]["warnings"]:
                skip("static_warnings")
            else:
                seen.add(result["task_id"])
                traj = json.loads((task_dir / "traj.json").read_text())
                task = json.loads((task_dir / "task.json").read_text())
                yield result, traj, task, task_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="one or more generate.py output directories")
    p.add_argument("--out", required=True)
    p.add_argument("--format", choices=["openai", "swift", "code"], default="swift")
    p.add_argument("--val-ratio", type=float, default=0.02, help="split by task-id hash, stable across re-exports")
    p.add_argument("--keep-format-errors", action="store_true", help="keep trajectories with tool-call format errors")
    p.add_argument("--task-id", action="append", default=[], help="export only these task ids (repeatable)")
    p.add_argument("--keep-blocked", action="store_true", help="keep trajectories with commands blocked at the harness boundary")
    p.add_argument("--keep-paths", action="store_true",
                   help="keep records that still contain absolute cluster paths after the task directory is "
                        "rewritten to /work (records containing credentials are always dropped)")
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--no-warnings", action="store_true", help="drop scripts with static-check warnings (e.g. no animation)")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    images = ImageStore(out)
    files = {"train": open(out / "train.jsonl", "w"), "val": open(out / "val.jsonl", "w")}
    meta_files = {k: open(out / f"{k}.meta.jsonl", "w") for k in files}
    counts = {"train": 0, "val": 0}
    steps, skipped = [], {}
    for result, traj, task, task_dir in iter_accepted([Path(r) for r in args.runs], args, skipped):
        meta = {"task_id": result["task_id"], "prompt": task["prompt"], "model_name": result["model_name"],
                "verification": result["verification"], "n_steps": result["n_steps"],
                "static_warnings": result["static_check"]["warnings"], "task_meta": task["meta"]}
        split = "val" if is_val(result["task_id"], args.val_ratio) else "train"
        messages = sanitize_messages(traj["messages"], task_dir)
        if args.format == "code":
            code = (task_dir / "final_script.py").read_text()
            record = {"task_id": result["task_id"], "prompt": task["prompt"], "code": code,
                      "static_warnings": meta["static_warnings"], "verification": meta["verification"]}
            code_dir = out / "code" / split
            code_dir.mkdir(parents=True, exist_ok=True)
            (code_dir / f"{result['task_id']}.py").write_text(code)
            video = task_dir / "final_run" / "output.mp4"
            if video.is_file():  # rendered by the harness when the script was accepted
                video_dir = out / "videos" / split
                video_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(video, video_dir / f"{result['task_id']}.mp4")
                record["video"] = f"videos/{split}/{result['task_id']}.mp4"
        elif args.format == "openai":
            record = {"messages": to_openai(messages, images), "tools": [BASH_TOOL], "meta": meta}
        else:
            msgs, imgs = to_swift(messages, images)
            record = {"tools": json.dumps([BASH_TOOL]), "messages": msgs}
            if imgs:
                record["images"] = imgs
        leak = find_leak(record, images.dir)
        if leak == "secret" or (leak == "path" and not args.keep_paths):
            skipped[f"{leak}_leak"] = skipped.get(f"{leak}_leak", 0) + 1
            if args.format == "code":  # undo the files written for this record above
                (out / "code" / split / f"{result['task_id']}.py").unlink(missing_ok=True)
                (out / "videos" / split / f"{result['task_id']}.mp4").unlink(missing_ok=True)
            continue
        files[split].write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_files[split].write(json.dumps(meta, ensure_ascii=False) + "\n")  # line-aligned with the data file
        counts[split] += 1
        steps.append(result["n_steps"])
    for f in [*files.values(), *meta_files.values()]:
        f.close()

    stats = {"format": args.format, "counts": counts, "skipped": skipped,
             "mean_steps": round(sum(steps) / len(steps), 2) if steps else 0, "runs": args.runs}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
