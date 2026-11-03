#!/usr/bin/env python
"""Batch Ref2VA generation with MiniMax-H3 on a running SGLang server.

Each manifest line (JSONL) describes one clip:
    {"id": "scene_0001",                 # optional, defaults to the video file name
     "video": "videos/scene_0001.mp4",   # white-model (untextured) render, required; a relative
                                         # path is resolved against the manifest's directory
     "scene": "A rainy neon street ...", # scene description, required unless "raw_prompt" is given
     "duration": 5,                      # optional, seconds (4-15); defaults to the input length
     "aspect_ratio": "16:9",             # optional, default "auto" (follows the input video)
     "seed": 42,                         # optional
     "start_time": 0.0,                  # optional, offset into the input video
     "raw_prompt": "..."}                # optional, sent verbatim instead of the template

Outputs: <out>/<id>.mp4, <out>/<id>.json (request + final status) and <out>/results.jsonl
(results-shard<i>of<n>.jsonl when --num-shards > 1, so shards can share one output directory).
Clips whose .mp4 already exists are skipped, so an interrupted run can simply be restarted.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

# {scene} is the line's scene description; the white-model video is the first video condition,
# which H3 refers to as <Video 1>.
DEFAULT_TEMPLATE = """Generate a high-quality, photorealistic video based on the following scene description and white-model reference video.

Use the scene description to determine the objects, colors, materials, and visual appearance. Follow the white-model video to preserve the scene layout, object movements, camera motion, and timing.

Maintain visual realism and temporal consistency.

Scene Description:
{scene}

Reference Video:
<Video 1>"""

MIN_SEC, MAX_SEC = 4.0, 15.0
_print_lock = threading.Lock()


def log(*a):
    with _print_lock:
        print(time.strftime("%H:%M:%S"), *a, flush=True)


def probe_duration(path):
    """Length of a video in seconds, or None if it cannot be read."""
    if shutil.which("ffprobe"):
        try:
            out = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                capture_output=True, text=True, timeout=60).stdout.strip()
            return float(out)
        except (ValueError, subprocess.SubprocessError):
            pass
    try:
        import av
        with av.open(path) as c:
            if c.duration:
                return c.duration / 1_000_000
            s = c.streams.video[0]
            return float(s.duration * s.time_base)
    except Exception:
        return None


def build_request(item, args):
    video = os.path.abspath(item["video"])
    if "raw_prompt" in item:
        prompt = item["raw_prompt"]
    else:
        prompt = args.template.format(scene=item["scene"].strip())

    duration = item.get("duration")
    if duration is None:
        src = probe_duration(video)
        duration = (src - float(item.get("start_time", 0))) if src else args.default_duration
    duration = round(min(max(float(duration), MIN_SEC), MAX_SEC), 2)

    cond = {"type": "video", "uri": "file://" + video, "role": "reference"}
    if item.get("start_time"):
        cond["start_time_seconds"] = float(item["start_time"])

    return {
        "model": "MiniMaxAI/MiniMax-H3",
        "task": "ref2va",
        "prompt": prompt,
        # The OpenAI-style "seconds" field must be an integer (7.5 is rejected with HTTP 400); H3 ignores it and
        # takes the length from target.duration_seconds, which keeps the exact value.
        "seconds": max(1, int(round(duration))),
        "conditions": [cond],
        "target": {
            "short_edge": args.short_edge,
            "aspect_ratio": item.get("aspect_ratio", args.aspect_ratio),
            "duration_seconds": duration,
        },
        "num_outputs_per_prompt": 1,
        "num_inference_steps": args.steps,
        "flow_shift": 12.0,
        "audio_flow_shift": 3.0,
        "seed": int(item.get("seed", args.seed)),
    }


def run_one(item, args):
    base = args.server.rstrip("/")
    out_mp4 = args.out / f"{item['id']}.mp4"
    req = build_request(item, args)
    record = {"id": item["id"], "video": item["video"], "request": req}

    for attempt in range(1, args.retries + 2):
        t0 = time.time()
        try:
            r = requests.post(f"{base}/v1/videos", json=req, timeout=120)
            r.raise_for_status()
            vid = r.json()["id"]
            log(f"[{item['id']}] submitted as {vid} ({req['seconds']}s, attempt {attempt})")
            while True:
                time.sleep(args.poll)
                st = requests.get(f"{base}/v1/videos/{vid}", timeout=60).json()
                status = st.get("status")
                if status == "completed":
                    break
                if status == "failed":
                    raise RuntimeError(f"server reported failure: {st.get('error') or st}")
                if time.time() - t0 > args.job_timeout:
                    raise TimeoutError(f"no result after {args.job_timeout}s (last status {status})")
            tmp = out_mp4.with_suffix(".mp4.part")
            with requests.get(f"{base}/v1/videos/{vid}/content", stream=True, timeout=600) as d:
                d.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in d.iter_content(1 << 20):
                        f.write(chunk)
            tmp.rename(out_mp4)
            record.update(status="completed", server_id=vid, seconds_taken=round(time.time() - t0, 1),
                          output=str(out_mp4))
            log(f"[{item['id']}] done in {record['seconds_taken']}s -> {out_mp4}")
            break
        except Exception as e:  # noqa: BLE001 - every failure is retried, then recorded
            record.update(status="failed", error=f"{type(e).__name__}: {e}")
            log(f"[{item['id']}] attempt {attempt} failed: {record['error']}")

    (args.out / f"{item['id']}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2))
    with _print_lock, open(args.results, "a") as f:
        f.write(json.dumps({k: v for k, v in record.items() if k != "request"}, ensure_ascii=False) + "\n")
    return record


def load_manifest(path):
    items = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            it = json.loads(line)
            if "video" not in it or ("scene" not in it and "raw_prompt" not in it):
                sys.exit(f"{path}:{n}: needs 'video' and 'scene' (or 'raw_prompt')")
            it["video"] = str((Path(path).resolve().parent / it["video"]).resolve())
            it.setdefault("id", Path(it["video"]).stem)
            items.append(it)
    ids = [it["id"] for it in items]
    dup = {i for i in ids if ids.count(i) > 1}
    if dup:
        sys.exit(f"duplicate ids in manifest: {sorted(dup)[:10]}")
    return items


def wait_for_server(base, timeout):
    t0 = time.time()
    while time.time() - t0 < timeout:
        for ep in ("/liveness", "/health", "/v1/models"):
            try:
                if requests.get(base + ep, timeout=5).status_code == 200:
                    return True
            except requests.RequestException:
                pass
        time.sleep(10)
    return False


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", required=True, help="JSONL file, one clip per line")
    p.add_argument("--out", required=True, type=Path, help="output directory")
    p.add_argument("--server", default="http://127.0.0.1:30011")
    p.add_argument("--max-inflight", type=int, default=2,
                   help="jobs submitted at once; the server runs them one by one, 2 keeps it busy")
    p.add_argument("--steps", type=int, default=50, help="denoising steps (official recipe: 50)")
    p.add_argument("--short-edge", type=int, default=480,
                   help="output short edge in px; 480 gives 864x480 for 16:9 input. 768 is the only size "
                        "MiniMax/SGLang have verified (16:9 -> 1344x768); other values log a warning")
    p.add_argument("--aspect-ratio", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default-duration", type=float, default=5.0,
                   help="used when a line has no 'duration' and the input length cannot be read")
    p.add_argument("--prompt-template", type=Path,
                   help="text file with a {scene} placeholder; replaces the built-in white-model template")
    p.add_argument("--shard-id", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--retries", type=int, default=1)
    p.add_argument("--poll", type=float, default=5.0)
    p.add_argument("--job-timeout", type=float, default=3600)
    p.add_argument("--wait-server", type=float, default=0, help="seconds to wait for the server to come up")
    p.add_argument("--skip-existing-in", type=Path, action="append", default=[], metavar="DIR",
                   help="also skip clips whose <id>.mp4 already exists in DIR (repeatable); "
                        "for continuing a run into a new output directory")
    p.add_argument("--max-consecutive-failures", type=int, default=6,
                   help="stop the run after this many clips fail in a row (e.g. disk full / quota), "
                        "so a batch job does not keep the GPUs busy for nothing; 0 disables")
    p.add_argument("--dry-run", action="store_true", help="print the first request and exit")
    args = p.parse_args()

    args.template = args.prompt_template.read_text() if args.prompt_template else DEFAULT_TEMPLATE
    args.out.mkdir(parents=True, exist_ok=True)
    args.results = args.out / ("results.jsonl" if args.num_shards == 1
                               else f"results-shard{args.shard_id}of{args.num_shards}.jsonl")

    items = load_manifest(args.manifest)[args.shard_id::args.num_shards]
    for it in items:
        if not os.path.isfile(it["video"]):
            sys.exit(f"[{it['id']}] video not found: {it['video']}")
    done_dirs = [args.out, *args.skip_existing_in]
    todo = [it for it in items if not any((d / f"{it['id']}.mp4").exists() for d in done_dirs)]
    log(f"{len(items)} clips in shard {args.shard_id}/{args.num_shards}, "
        f"{len(items) - len(todo)} already done, {len(todo)} to generate")

    if args.dry_run:
        if todo:
            print(json.dumps(build_request(todo[0], args), ensure_ascii=False, indent=2))
        return
    if not todo:
        return
    if args.wait_server and not wait_for_server(args.server.rstrip("/"), args.wait_server):
        sys.exit(f"server {args.server} not ready after {args.wait_server}s")

    ok = failed = streak = 0
    with ThreadPoolExecutor(args.max_inflight) as pool:
        for fut in as_completed([pool.submit(run_one, it, args) for it in todo]):
            try:
                good = fut.result()["status"] == "completed"
            except Exception as e:  # noqa: BLE001 - e.g. the .json/results write itself failed
                log(f"clip bookkeeping failed: {type(e).__name__}: {e}")
                good = False
            ok += good
            failed += not good
            streak = 0 if good else streak + 1
            if args.max_consecutive_failures and streak >= args.max_consecutive_failures:
                log(f"ABORT: {streak} clips failed in a row; stopping instead of burning GPU time "
                    f"(see {args.results} / the log above for the error)")
                pool.shutdown(wait=False, cancel_futures=True)
                sys.exit(2)
    log(f"finished: {ok} ok, {failed} failed (see {args.results})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
