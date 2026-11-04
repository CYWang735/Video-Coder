"""Draft videos -> MiniMax-H3 manifest (Minimax-H3-server/batch_generate.py): one line per prompt with a draft.

    python make_h3_manifest.py --data data.json --drafts <out>/drafts --out <out>/h3_manifest.jsonl

--data is the same JSON as for inference (a list of {"id", "prompt", ...}, or JSONL). Prompts without an accepted
draft are listed on stderr and left out.
"""

import argparse
import json
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--drafts", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    with open(args.data) as f:
        records = json.load(f) if args.data.endswith(".json") else [json.loads(x) for x in f if x.strip()]
    lines, missing = [], []
    for r in records:
        draft = os.path.abspath(os.path.join(args.drafts, str(r["id"]), "render.mp4"))
        if os.path.isfile(draft):
            lines.append({"id": str(r["id"]), "video": draft, "scene": r["prompt"].strip()})
        else:
            missing.append(str(r["id"]))
    with open(args.out, "w") as f:
        f.writelines(json.dumps(x, ensure_ascii=False) + "\n" for x in lines)
    print(f"{len(lines)} drafts -> {args.out}")
    if missing:
        print(f"{len(missing)} prompts without a draft: {', '.join(missing[:20])}", file=sys.stderr)


if __name__ == "__main__":
    main()
