"""Turn a Video-Coder task file (.jsonl, or .json with a list; a prompt per record) into a uni-agent parquet.

    python -m uni_agent.tasks.video_coder.preprocess --tasks ~/wangcy/Video-Coder-sftdata/data/example_tasks.jsonl \
        --out ~/data/uni_agent/video_coder_example.parquet
"""

import argparse
import json
import os

import pandas as pd


def _read_records(path: str) -> list[dict]:
    """Records of a .jsonl file (one object per line) or a .json file (a list of objects)."""
    with open(path) as f:
        if path.endswith(".json"):
            data = json.load(f)
            if not isinstance(data, list):
                raise SystemExit(f"{path}: expected a JSON list of task objects")
            return data
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompt-field", default="prompt")
    parser.add_argument("--gt-field", default="gt_video",
                        help="record field with the GT video path (for the similarity reward); copied to metadata")
    parser.add_argument("--gen-prompt-field", default="gen_prompt",
                        help="record field with the Ref2VA prompt the GT was made with; copied to metadata if present")
    parser.add_argument("--require-gt", action="store_true", help="fail if a record has no existing GT video")
    parser.add_argument("--gt-request-sidecar", action="store_true",
                        help="read <gt_video without .mp4>.json written when the GT was generated and keep its Ref2VA "
                             "request (minus prompt/conditions) as metadata.gen_request, so training regenerates with "
                             "the GT's own parameters (duration, seed, ...)")
    args = parser.parse_args()
    rows = []
    tasks = os.path.expanduser(args.tasks)
    task_dir = os.path.dirname(os.path.abspath(tasks))  # relative GT paths are relative to the task file
    for rec in _read_records(tasks):
        metadata = {"id": str(rec.get("id", ""))}
        gt = rec.get(args.gt_field)
        if gt:
            metadata["gt_video"] = os.path.abspath(os.path.join(task_dir, os.path.expanduser(gt)))
        if args.require_gt and not (gt and os.path.isfile(metadata["gt_video"])):
            raise SystemExit(f"record {rec.get('id')!r}: GT video missing or not found: {gt!r}")
        if rec.get(args.gen_prompt_field):
            metadata["gen_prompt"] = rec[args.gen_prompt_field]
        if args.gt_request_sidecar and gt:
            sidecar = os.path.splitext(metadata["gt_video"])[0] + ".json"
            if not os.path.isfile(sidecar):
                raise SystemExit(f"record {rec.get('id')!r}: no GT request sidecar {sidecar}")
            request = json.load(open(sidecar)).get("request") or {}
            metadata["gen_request"] = {k: v for k, v in request.items() if k not in ("prompt", "conditions")}
        rows.append({
            "data_source": "video_coder",
            "prompt": [{"role": "user", "content": rec[args.prompt_field].strip()}],
            "extra_info": {"tools_kwargs": {"task": {"name": "video_coder", "metadata": metadata}}},
        })
    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
