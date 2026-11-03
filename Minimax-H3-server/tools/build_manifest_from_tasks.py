#!/usr/bin/env python3
"""Turn a Video-Coder task directory into a batch_generate.py data package.

Input layout (one sub-directory per task):
    <tasks_dir>/<task_id>/task.json               "prompt" is used as the scene description
    <tasks_dir>/<task_id>/final_run/output.mp4    white-model render
    <tasks_dir>/<task_id>/result.json             "status" must be "accepted"

Output (self-contained; can be copied to another machine as is):
    <out_dir>/manifest.jsonl   {"id": <task_id>, "video": "videos/<task_id>.mp4", "scene": <prompt>}
    <out_dir>/videos/          copies of the renders
    <out_dir>/excluded.jsonl   tasks left out, with the reason

Usage: python3 tools/build_manifest_from_tasks.py <tasks_dir> <out_dir> [--symlink]
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("tasks_dir", type=Path)
    p.add_argument("out_dir", type=Path)
    p.add_argument("--symlink", action="store_true",
                   help="link the renders instead of copying them (the package then only works on this machine)")
    args = p.parse_args()

    videos_dir = args.out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    kept, excluded = [], []

    for task in sorted(d for d in args.tasks_dir.iterdir() if d.is_dir()):
        video = task / "final_run" / "output.mp4"
        try:
            prompt = json.loads((task / "task.json").read_text())["prompt"].strip()
        except (OSError, ValueError, KeyError, AttributeError):
            prompt = ""
        try:
            status = json.loads((task / "result.json").read_text()).get("status")
        except (OSError, ValueError):
            status = None

        reason = None
        if not video.is_file():
            reason = "no final_run/output.mp4"
        elif status != "accepted":
            reason = f"result status {status!r}"
        elif not prompt:
            reason = "empty task.json prompt"
        if reason:
            excluded.append({"id": task.name, "reason": reason})
            continue

        dst = videos_dir / f"{task.name}.mp4"
        if not dst.exists():
            if args.symlink:
                os.symlink(video.resolve(), dst)
            else:
                shutil.copy2(video, dst)
        kept.append({"id": task.name, "video": f"videos/{dst.name}", "scene": prompt})

    with open(args.out_dir / "manifest.jsonl", "w") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(args.out_dir / "excluded.jsonl", "w") as f:
        for row in excluded:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(kept)} tasks -> {args.out_dir / 'manifest.jsonl'}; {len(excluded)} excluded -> excluded.jsonl")


if __name__ == "__main__":
    main()
