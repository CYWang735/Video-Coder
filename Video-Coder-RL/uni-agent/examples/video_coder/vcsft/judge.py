"""VLM judge: does the rendered white-model video show what the scene description asks for?

Used by the harness as the final acceptance gate (the agent never sees the judge). Frames are sampled evenly
from the video and sent with the description to a vision model through the Responses API.
"""

import json
import re
from pathlib import Path

import litellm

from vcsft.video import sample_frames

JUDGE_INSTRUCTIONS = """You are a strict reviewer of white-model (untextured clay) Blender renders.
You get the scene description a video was supposed to depict, then frames sampled evenly from that video, in order.

White-model renders use one uniform gray clay material and fixed studio lighting. IGNORE requirements about colors, \
materials, textures, clothing color, lighting mood, facial expressions, realism or photographic style: those cannot \
be shown. Judge only what geometry, layout, motion and camera can show.

1. List the essential visual requirements of the description that a white-model render can show: the objects and \
their recognizable shapes, counts, spatial arrangement, the motion or events and their order, and the camera behavior.
2. For each requirement decide from the frames whether it is satisfied, with brief evidence.
3. List serious defects: empty or mostly empty frames, key objects outside the frame or cut off, key objects too small \
to recognize, objects passing through each other, broken or degenerate geometry, motion that contradicts the description.

verdict = "pass" only if every essential requirement is satisfied and there is no serious defect, otherwise "fail".
Answer with one JSON object only, no other text:
{"requirements": [{"requirement": str, "satisfied": bool, "evidence": str}], "defects": [str], "score": int (0-10), \
"verdict": "pass" | "fail", "summary": str}"""


def build_input(description: str, frames: list[tuple[int, str]], fps: float, n_frames: int) -> list[dict]:
    content = [{"type": "input_text", "text": f"{JUDGE_INSTRUCTIONS}\n\nScene description:\n<scene_description>\n"
                                              f"{description}\n</scene_description>\n\n"
                                              f"The video has {n_frames} frames at {fps:g} fps "
                                              f"({n_frames / fps:.1f} s). Sampled frames:"}]
    for frame, url in frames:
        content.append({"type": "input_text", "text": f"Frame {frame} (t = {(frame - 1) / fps:.2f} s):"})
        content.append({"type": "input_image", "image_url": url})
    return [{"role": "user", "content": content}]


def parse_verdict(text: str) -> dict:
    """Extract the JSON object from the model's answer; raise ValueError if there is none or it is malformed."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        raise ValueError("no JSON object in judge answer")
    data = json.loads(match.group(0))
    if data.get("verdict") not in ("pass", "fail"):
        raise ValueError(f"invalid verdict: {data.get('verdict')!r}")
    return data


def _call(model: str, input_items: list[dict], api: str = "responses", **kwargs) -> str:
    if api == "chat":  # e.g. a local vLLM server: same content as Chat Completions parts
        content = [{"type": "text", "text": p["text"]} if p["type"] == "input_text"
                   else {"type": "image_url", "image_url": {"url": p["image_url"]}} for p in input_items[0]["content"]]
        r = litellm.completion(model=model, messages=[{"role": "user", "content": content}], **kwargs)
        return r.choices[0].message.content or ""
    r = litellm.responses(model=model, input=input_items, **kwargs)
    return r.output_text


def judge_video(description: str, video: str | Path, n_frames: int, fps: float, cfg: dict, take_call=None) -> dict:
    """Return {"verdict", "score", "requirements", "defects", "summary", "n_calls"} or {"verdict": "error", ...}.

    take_call: called before every API request (budget accounting); may raise to stop.
    """
    frames = sample_frames(video, n_frames, cfg.get("frames", 8), cfg.get("width", 640))
    if not frames:
        return {"verdict": "error", "error": "could not extract frames", "n_calls": 0}
    items = build_input(description, frames, fps, n_frames)
    n_calls, last_error = 0, ""
    for _ in range(cfg.get("attempts", 2)):
        if take_call:
            take_call()
        n_calls += 1
        try:
            text = _call(cfg["model"], items, cfg.get("api", "responses"), **cfg.get("model_kwargs", {}))
            data = parse_verdict(text)
            return {**data, "frames": [f for f, _ in frames], "n_calls": n_calls}
        except (ValueError, json.JSONDecodeError) as e:
            last_error = f"unparseable judge answer: {e}"
    return {"verdict": "error", "error": last_error, "n_calls": n_calls}
