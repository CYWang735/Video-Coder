"""Inspect a rendered video with the ffmpeg binary bundled by imageio-ffmpeg (no system ffmpeg on the cluster)."""

import base64
import re
import subprocess
from pathlib import Path

import imageio_ffmpeg

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def probe(path: str | Path) -> dict:
    """Decode the whole video stream: returns frame count, size and fps, or an error."""
    path = Path(path)
    if not path.is_file():
        return {"exists": False}
    info: dict = {"exists": True, "bytes": path.stat().st_size}
    proc = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"],
                          capture_output=True, text=True, timeout=300)
    err = proc.stderr
    if m := re.search(r"Video: (\w+).*?, (\d{2,5})x(\d{2,5})", err):
        info["codec"], info["width"], info["height"] = m.group(1), int(m.group(2)), int(m.group(3))
    if m := re.search(r"([\d.]+) fps", err):
        info["fps"] = float(m.group(1))
    frames = re.findall(r"frame=\s*(\d+)", err)
    info["n_frames"] = int(frames[-1]) if frames else 0
    if proc.returncode != 0 or not frames:
        info["decode_error"] = err.strip().splitlines()[-1] if err.strip() else f"ffmpeg exit {proc.returncode}"
    return info


def frame_stats(path: str | Path, n_frames: int, samples: int = 6, w: int = 64, h: int = 36) -> dict:
    """Grayscale statistics of `samples` evenly spaced frames, downscaled to w x h.

    std: per-frame pixel standard deviation (≈0 when the camera sees only background).
    diff: mean absolute pixel difference between consecutive sampled frames (≈0 when nothing moves).
    """
    idx = sorted({round(i * (n_frames - 1) / (samples - 1)) for i in range(samples)})
    select = "+".join(f"eq(n\\,{f})" for f in idx)
    proc = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
                           "-vf", f"select='{select}',scale={w}:{h},format=gray", "-vsync", "0",
                           "-f", "rawvideo", "-"], capture_output=True, timeout=300)
    size = w * h
    frames = [proc.stdout[i:i + size] for i in range(0, len(proc.stdout) - size + 1, size)]
    std = []
    for f in frames:
        mean = sum(f) / size
        std.append(round((sum((p - mean) ** 2 for p in f) / size) ** 0.5, 2))
    diff = [round(sum(abs(a - b) for a, b in zip(f0, f1)) / size, 2) for f0, f1 in zip(frames, frames[1:])]
    return {"frames": [i + 1 for i in idx][: len(frames)], "std": std, "diff": diff}


def contact_sheet(path: str | Path, out: str | Path, frames: list[int], width: int = 480) -> bool:
    """Tile the given 0-based frame indices into a 2-column grid image."""
    select = "+".join(f"eq(n\\,{f})" for f in frames)
    rows = (len(frames) + 1) // 2
    proc = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-i", str(path),
                           "-vf", f"select='{select}',scale={width}:-2,tile=2x{rows}", "-frames:v", "1", str(out)],
                          capture_output=True, text=True, timeout=300)
    return proc.returncode == 0 and Path(out).is_file()


def sample_frames(video: str | Path, n_frames: int, k: int, width: int) -> list[tuple[int, str]]:
    """(1-based frame number, JPEG data URL) for k evenly spaced frames."""
    idx = sorted({round(i * (n_frames - 1) / (k - 1)) for i in range(k)})
    out = []
    for i in idx:
        proc = subprocess.run([FFMPEG, "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(video),
                               "-vf", f"select='eq(n\\,{i})',scale={width}:-2", "-frames:v", "1",
                               "-q:v", "4", "-f", "image2", "-c:v", "mjpeg", "-"], capture_output=True, timeout=120)
        if proc.returncode == 0 and proc.stdout:
            out.append((i + 1, "data:image/jpeg;base64," + base64.b64encode(proc.stdout).decode()))
    return out
