"""Execute a generated script with bpy (Blender 4.5, Workbench via Mesa llvmpipe on CPU) and check the video it renders.

Shared by the agent's `run_blender` tool and by the harness's final re-execution, so both judge a script the same way.
Harness settings come from the config (`harness:`) or, inside agent shells, from the VCSFT_HARNESS env var (JSON).
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from video_coder_harness.video import contact_sheet, frame_stats, probe, sample_frames

RUN_BPY = Path(__file__).resolve().parent / "run_bpy.py"
_SECRET_ENV = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", re.I)
# Blender's per-frame render progress; dropped from the log shown to the agent.
_NOISE = re.compile(r"^(Fra:|Append frame|Time: |\s*Saving:|Saved: |Read prefs|Blender quit|$)")


def bpy_env(harness: dict) -> dict:
    libs = os.path.expandvars(harness["bpy_libs"])
    env = {k: v for k, v in os.environ.items() if not _SECRET_ENV.search(k)}
    env |= {
        "LD_LIBRARY_PATH": f"{libs}/lib",
        "__EGL_VENDOR_LIBRARY_DIRS": f"{libs}/share/glvnd/egl_vendor.d",
        "EGL_PLATFORM": "surfaceless",
        "LIBGL_ALWAYS_SOFTWARE": "1",
        "GALLIUM_DRIVER": "llvmpipe",
        "PYTHONNOUSERSITE": "1",
    }
    env.pop("PYTHONPATH", None)
    return env


def execute(script: str | Path, cwd: str | Path, harness: dict, preview: bool = True) -> dict:
    """Run `script` in `cwd`; it is expected to render harness['output_video'] there."""
    cwd, script = Path(cwd).resolve(), Path(script).resolve()
    video_path = cwd / harness["output_video"]
    if video_path.exists():  # never judge a stale video from an earlier run
        video_path.unlink()
    timeout = harness["exec_timeout"]
    report: dict = {"script": script.name}
    with tempfile.TemporaryDirectory() as tmp:
        result_file = Path(tmp) / "result.json"
        cmd = [os.path.expandvars(harness["bpy_python"]), str(RUN_BPY), str(script),
               "--cwd", str(cwd), "--result", str(result_file), "--samples", str(harness.get("summary_samples", 6))]
        t0 = time.time()
        try:
            proc = subprocess.run(cmd, cwd=cwd, env=bpy_env(harness), capture_output=True, text=True,
                                  errors="replace", timeout=timeout)
            log = proc.stdout + proc.stderr
            run = json.loads(result_file.read_text()) if result_file.is_file() else {}
            if not run:
                report["error"] = crash_message(proc.returncode)
        except subprocess.TimeoutExpired as e:
            log = (e.stdout or "") if isinstance(e.stdout, str) else (e.stdout or b"").decode(errors="replace")
            run, report["error"] = {}, f"timed out after {timeout}s (render included)"
        report["seconds"] = round(time.time() - t0, 1)

    report["executed"] = bool(run.get("ok"))
    if run.get("traceback"):
        # relative paths: workspace locations must not end up in training data
        report["traceback"] = _script_frames(run["traceback"], script.name).replace(f"{script.parent}/", "")
    report["log_tail"] = [ln for ln in log.splitlines() if not _NOISE.match(ln)][-25:]
    report["files_written"] = run.get("files_written", [])
    if "scene" in run:
        report["scene"] = _compact_scene(run["scene"])

    video = probe(video_path)
    report["video"] = video
    want_w, want_h = harness["resolution"]
    problems = []
    if not report["executed"]:
        problems.append("script did not execute successfully")
    if not video.get("exists"):
        problems.append(f"{harness['output_video']} was not rendered")
    else:
        if video.get("decode_error"):
            problems.append(f"video cannot be decoded: {video['decode_error']}")
        lo = harness["frame_count"]
        hi = harness.get("max_frame_count", lo)
        n = video.get("n_frames")
        if not (n is not None and lo <= n <= hi):
            want = f"{lo}" if lo == hi else f"{lo}-{hi}"
            problems.append(f"video has {n} frames, expected {want}")
        frame_range = (report.get("scene") or {}).get("frame_range")
        if lo != hi and frame_range and n != frame_range[1] - frame_range[0] + 1:
            problems.append(f"video has {n} frames but the scene frame range is {frame_range}")
        if video.get("fps") and abs(video["fps"] - harness["fps"]) > 0.01:
            problems.append(f"video is {video['fps']:g} fps, expected {harness['fps']} (set scene.render.fps)")
        if (video.get("width"), video.get("height")) != (want_w, want_h):
            problems.append(f"video is {video.get('width')}x{video.get('height')}, expected {want_w}x{want_h}")
        if n and not video.get("decode_error"):
            stats = frame_stats(video_path, n, samples=harness.get("content_samples", 8))
            report["frame_stats"] = stats
            problems += content_problems(stats, run.get("scene"))
    extra = [f for f in report["files_written"] if f != harness["output_video"]]
    if extra:
        problems.append(f"script wrote files other than {harness['output_video']}: {extra}")
    report["problems"] = problems
    report["ok"] = not problems
    if preview and video.get("n_frames"):
        n = video["n_frames"]
        frames = sorted({0, n // 3, 2 * n // 3, n - 1})
        if contact_sheet(video_path, cwd / "preview.png", frames):
            report["preview"] = {"file": "preview.png", "frames": [f + 1 for f in frames]}
    return report


_SIGNALS = {-6: "SIGABRT", -11: "SIGSEGV", -9: "SIGKILL", -7: "SIGBUS", -8: "SIGFPE"}


def crash_message(returncode: int) -> str:
    if returncode < 0:
        sig = _SIGNALS.get(returncode, f"signal {-returncode}")
        return (f"Blender crashed ({sig}) before the script finished, so there is no Python traceback; the blender "
                "log tail may show the cause. Usual causes: invalid mesh data (bmesh/mesh elements used after "
                "they were removed, or a stale bmesh index table: call ensure_lookup_table() after adding "
                "elements), or using an object or datablock after deleting it. Bisect by commenting out parts of "
                "the script.")
    return f"Blender process exited with code {returncode} without a result"


# A Workbench frame showing only background has a pixel std of ~0.6 (64x36 gray); scenes with geometry are > 3.
BLANK_STD = 2.0
STATIC_DIFF = 0.5


def content_problems(stats: dict, scene: dict | None) -> list[str]:
    """Hard checks on what the camera actually sees: mostly empty frames, no visible motion, nothing in view."""
    problems = []
    std, diff, frames = stats["std"], stats["diff"], stats["frames"]
    blank = [f for f, s in zip(frames, std) if s < BLANK_STD]
    if std and len(blank) > len(std) // 2:
        problems.append(f"the rendered frames are empty (only background) at sampled frames {blank}: "
                        "the camera does not see the scene")
    if diff and max(diff) < STATIC_DIFF:
        problems.append("the video is static: nothing visibly moves between the sampled frames")
    if scene:
        n_mesh = (scene.get("objects_by_type") or {}).get("MESH", 0)
        outside = scene.get("mesh_origins_out_of_camera_view") or {}
        n_sampled = len(scene.get("sampled_frames") or [])
        if n_mesh and len(outside) >= n_mesh and all(len(v) == n_sampled for v in outside.values()):
            problems.append("no mesh is inside the camera view in any sampled frame: aim the camera at the scene")
    return problems


HARNESS_DIR = str(Path(__file__).resolve().parent)


def _script_frames(tb: str, script_name: str) -> str:
    """Keep the traceback from the first frame inside the generated script; drop runner/runpy frames and frames
    of the harness's own guards (video_coder_harness/), whose paths must not leak into trajectories."""
    lines = tb.splitlines()
    start = next((i for i, ln in enumerate(lines) if script_name in ln and ln.lstrip().startswith("File")), None)
    if start is None:
        return tb
    kept, skipping = [], False
    for ln in lines[start:]:
        if ln.lstrip().startswith("File "):
            skipping = HARNESS_DIR in ln
        elif not ln.startswith(" "):  # the final "ExcType: message" line
            skipping = False
        if not skipping:
            kept.append(ln)
    return "Traceback (most recent call last):\n" + "\n".join(kept)


def _compact_scene(scene: dict) -> dict:
    moving = scene.get("moving_mesh_positions", {})
    return {
        "frame_range": scene.get("frame_range"),
        "fps": scene.get("fps"),
        "camera": scene.get("camera"),
        "objects_by_type": scene.get("objects_by_type"),
        "n_faces_evaluated": scene.get("n_faces_evaluated"),
        "animated_objects": scene.get("animated_objects"),
        "sampled_frames": scene.get("sampled_frames"),
        "moving_mesh_world_positions (frame, x, y, z)": {k: moving[k] for k in list(moving)[:8]},
        "n_moving_meshes": len(moving),
        "mesh_origins_outside_camera_view": {k: f"{len(v)} of {len(scene.get('sampled_frames', []))} sampled frames"
                                             for k, v in scene.get("mesh_origins_out_of_camera_view", {}).items()},
    }


def format_report(r: dict) -> str:
    head = "RUN OK: script executed and rendered a valid video" if r["ok"] else "RUN FAILED"
    out = [head, f"seconds: {r['seconds']}"]
    if r.get("error"):
        out.append(f"error: {r['error']}")
    for p in r["problems"]:
        out.append(f"problem: {p}")
    if r.get("traceback"):
        out += ["", r["traceback"].rstrip()]
    if r.get("log_tail"):
        out += ["", "blender log (tail):", *r["log_tail"]]
    v = r["video"]
    if v.get("exists"):
        out += ["", "video: " + json.dumps({k: v[k] for k in ("n_frames", "width", "height", "fps", "codec", "bytes") if k in v})]
    if r.get("scene"):
        out += ["", "scene summary:"]
        for key, value in r["scene"].items():
            if isinstance(value, dict) and value and key != "objects_by_type":
                out.append(f"  {key}:")
                out += [f"    {name}: {json.dumps(v)}" for name, v in value.items()]
            else:
                out.append(f"  {key}: {json.dumps(value)}")
    return "\n".join(out)


def harness_from_env() -> dict:
    raw = os.environ.get("VCSFT_HARNESS")
    if not raw:
        sys.exit("VCSFT_HARNESS is not set (run_blender is meant to be called inside a generation workspace)")
    return json.loads(raw)


def main() -> None:
    p = argparse.ArgumentParser(description="Execute a bpy script with Blender 4.5 and check the rendered video.")
    p.add_argument("script", nargs="?", default="main.py")
    p.add_argument("--config", help="read harness settings from this config instead of $VCSFT_HARNESS")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    if args.config:
        import yaml

        harness = yaml.safe_load(Path(args.config).read_text())["harness"]
    else:
        harness = harness_from_env()
    if not Path(args.script).is_file():
        sys.exit(f"{args.script} not found")
    r = execute(args.script, Path.cwd(), harness)
    text = json.dumps(r, indent=2) if args.json else format_report(r)
    if harness.get("visual_feedback") and r["video"].get("n_frames") and not args.json:
        text += "\n\n" + frames_attachment(Path.cwd() / harness["output_video"], r["video"]["n_frames"],
                                            r["video"].get("fps") or harness["fps"], harness)
    print(text)
    sys.exit(0 if r["ok"] else 1)


def frames_attachment(video: Path, n_frames: int, fps: float, harness: dict) -> str:
    """Evenly spaced frames of the render as inline images (mini-swe-agent multimodal tags), one per frame.

    video_coder_harness.models turns each tag into a real image in the tool result, preceded by its caption line.
    """
    frames = sample_frames(video, n_frames, harness.get("preview_frames", 8), harness.get("preview_width", 640))
    lines = [f"rendered frames ({len(frames)} evenly spaced, in order):"]
    for frame, url in frames:
        lines.append(f"Frame {frame} (t = {(frame - 1) / fps:.2f} s):")
        lines.append(f"<MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>{url}</MSWEA_MULTIMODAL_CONTENT>")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
