"""Load generation tasks from .jsonl / .json. Each task needs a prompt; everything else is kept as metadata."""

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

PROMPT_FIELD_CANDIDATES = ("prompt", "instruction", "caption", "text", "description")
_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass
class Task:
    id: str
    prompt: str
    meta: dict = field(default_factory=dict)


def _read_records(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
        return records
    data = json.loads(text)
    if isinstance(data, dict):  # {"tasks": [...]} or {"data": [...]}
        data = data.get("tasks") or data.get("data")
    if not isinstance(data, list):
        raise ValueError(f"{path}: expected a JSON list of task objects")
    return data


def _resolve_prompt_field(record: dict, prompt_field: str | None) -> str:
    if prompt_field:
        return prompt_field
    for name in PROMPT_FIELD_CANDIDATES:
        if isinstance(record.get(name), str):
            return name
    raise ValueError(f"no prompt field found (tried {PROMPT_FIELD_CANDIDATES}); pass --prompt-field")


def load_tasks(path: str | Path, prompt_field: str | None = None, id_field: str = "id") -> list[Task]:
    path = Path(path)
    records = _read_records(path)
    tasks, seen = [], set()
    for i, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {i} is not an object")
        name = _resolve_prompt_field(record, prompt_field)
        prompt = record.get(name)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}: record {i} has empty or non-string {name!r}")
        prompt = prompt.strip()
        raw_id = record.get(id_field)
        task_id = _SAFE_ID.sub("_", str(raw_id)) if raw_id not in (None, "") else hashlib.sha1(prompt.encode()).hexdigest()[:12]
        if not is_safe_task_id(task_id):
            raise ValueError(f"{path}: record {i} has task id {raw_id!r}, which cannot be used as a directory name")
        if task_id in seen:
            raise ValueError(f"{path}: duplicate task id {task_id!r} (record {i})")
        seen.add(task_id)
        meta = {k: v for k, v in record.items() if k not in (name, id_field)}
        tasks.append(Task(id=task_id, prompt=prompt, meta=meta))
    return tasks


def is_safe_task_id(task_id: str) -> bool:
    """A task id is used as a directory name under <run>/tasks/: it must not be empty, `.`, `..` or contain `/`."""
    return bool(task_id) and task_id.strip(".") != "" and "/" not in task_id and "\0" not in task_id


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
