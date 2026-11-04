"""mini-swe-agent model classes that pass images printed by tools (run_blender's rendered frames) to the model.

A tool prints an image as mini-swe-agent's multimodal tag, usually after a caption line:
    Frame 18 (t = 0.71 s):
    <MSWEA_MULTIMODAL_CONTENT><CONTENT_TYPE>image_url</CONTENT_TYPE>data:image/jpeg;base64,...</MSWEA_MULTIMODAL_CONTENT>
The tags are cut out of the text *before* the observation template runs (the template truncates long outputs),
and the images are attached to the tool result as real image parts, each preceded by its caption:
    Responses API: function_call_output.output = [input_text, input_text(caption), input_image, ...]
    Chat Completions: tool message content = [text, text(caption), image_url, ...]
Select with `model_class: video_coder_harness.models.VisualResponseModel` (teacher) or `video_coder_harness.models.VisualChatModel` (vLLM students).
`video_coder_harness.models.VisualChatUserModel` is the fallback for Chat APIs that reject images inside `tool` messages: the tool message keeps
the text, and one `user` message right after it carries the images.
"""

import re

from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.litellm_response_model import LitellmResponseModel
from minisweagent.models.utils.openai_multimodal import DEFAULT_MULTIMODAL_REGEX

_TAG = re.compile(DEFAULT_MULTIMODAL_REGEX)


def placeholder(k: int) -> str:
    return f"[image {k} attached below]"


def split_images(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Replace multimodal tags with numbered placeholders; return the text and [(caption, image URL)]."""
    images: list[tuple[str, str]] = []

    def repl(m: re.Match) -> str:
        if m.group(1).strip() != "image_url":
            return m.group(0)
        before = text[: m.start()].rstrip().splitlines()
        caption = before[-1].strip() if before else ""
        images.append((caption, m.group(2).strip()))
        return placeholder(len(images))

    return _TAG.sub(repl, text), images


def _strip_outputs(outputs: list[dict]) -> tuple[list[dict], list[list[tuple[str, str]]]]:
    stripped, images = [], []
    for out in outputs:
        text, imgs = split_images(out.get("output") or "")
        stripped.append({**out, "output": text})
        images.append(imgs)
    return stripped, images


def _caption(k: int, caption: str) -> str:
    return f"image {k}: {caption}" if caption else f"image {k}:"


class VisualResponseModel(LitellmResponseModel):
    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]:
        stripped, images = _strip_outputs(outputs)
        msgs = super().format_observation_messages(message, stripped, template_vars)
        for msg, imgs in zip(msgs, images):
            if not imgs:
                continue
            parts = []
            for k, (caption, url) in enumerate(imgs, 1):
                parts += [{"type": "input_text", "text": _caption(k, caption)}, {"type": "input_image", "image_url": url}]
            if msg.get("type") == "function_call_output":
                msg["output"] = [{"type": "input_text", "text": msg["output"]}, *parts]
            elif msg.get("type") == "message":
                msg["content"] = [*msg["content"], *parts]
        return msgs


class VisualChatModel(LitellmModel):
    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]:
        stripped, images = _strip_outputs(outputs)
        msgs = super().format_observation_messages(message, stripped, template_vars)
        for msg, imgs in zip(msgs, images):
            if not imgs:
                continue
            text = msg["content"] if isinstance(msg["content"], str) else ""
            content = [{"type": "text", "text": text}]
            for k, (caption, url) in enumerate(imgs, 1):
                content += [{"type": "text", "text": _caption(k, caption)}, {"type": "image_url", "image_url": {"url": url}}]
            msg["content"] = content
        return msgs


class VisualChatUserModel(LitellmModel):
    def format_observation_messages(self, message: dict, outputs: list[dict], template_vars: dict | None = None) -> list[dict]:
        stripped, images = _strip_outputs(outputs)
        msgs = super().format_observation_messages(message, stripped, template_vars)
        content = []
        for i, imgs in enumerate(images, 1):
            for k, (caption, url) in enumerate(imgs, 1):
                label = _caption(k, caption)
                content += [{"type": "text", "text": f"result {i}, {label}" if len(images) > 1 else label},
                            {"type": "image_url", "image_url": {"url": url}}]
        if content:
            msgs.append({"role": "user", "content": [{"type": "text", "text": "Images from the tool output above:"}, *content]})
        return msgs
