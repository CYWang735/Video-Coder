"""Video-Coder task: write a bpy script for a scene description; reward = vcsft acceptance (+ similarity).

The ``video_coder`` agent runs the harness and re-renders the submitted script from scratch inside the
sandbox. Without ``similarity`` the reward is the acceptance: 1.0 if accepted (submitted, static check ok,
final script renders a valid video), else 0.0. With ``similarity`` (see :mod:`.similarity`) an accepted
render is also turned into a video by MiniMax-H3 Ref2VA and compared with the sample's GT video by a VLM:
reward = exe_weight * accept_weight * accepted + similarity_weight * similarity, where exe_weight =
exp(-global_step / T) anneals the execution term over training, T being similarity.exe_anneal_steps (by default the
total training steps; the framework sets both on the task config for training rollouts, not for validation).
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

from pydantic import Field

from ..base import Task, TaskConfig, TaskResult
from ..registry import register_task
from .similarity import (
    SimilarityRewardConfig,
    build_ref2va_prompt,
    combine_reward,
    exe_weight,
    gated_out,
    score_rollout,
)

logger = logging.getLogger(__name__)


class VideoCoderTaskConfig(TaskConfig):
    name: str = "video_coder"
    similarity: SimilarityRewardConfig | None = Field(
        default=None,
        description="Optional similarity reward against metadata[gt_field] (Ref2VA + VLM judge).",
    )
    debug_random_reward: bool = Field(
        default=False,
        description="DEBUG ONLY: replace the acceptance reward by a random 0/1, to exercise non-zero gradients.",
    )


@register_task("video_coder")
class VideoCoderTask(Task):
    config_model = VideoCoderTaskConfig

    async def run(self) -> TaskResult:
        cfg: VideoCoderTaskConfig = self.config  # type: ignore[assignment]
        sim = cfg.similarity if cfg.similarity is not None and cfg.similarity.enabled else None
        gt_video = None
        if sim is not None:  # fail before the (long) rollout if the sample cannot be scored
            gt_video = cfg.metadata.get(sim.gt_field)
            if not gt_video or not Path(gt_video).is_file():
                raise FileNotFoundError(f"video_coder: GT video {sim.gt_field}={gt_video!r} not found")

        render = None
        async with self.build_sandbox() as sandbox:
            agent = self.build_agent()
            result = await agent.run(sandbox=sandbox, messages=cfg.prompt)
            out = result.output
            accepted = bool(out.get("accepted"))
            if sim is not None and accepted:  # copy the render out before the sandbox is torn down
                if not out.get("final_video"):
                    raise RuntimeError("video_coder: accepted rollout reports no final_video")
                render = await sandbox.read_file(out["final_video"])

        reward = float(accepted)
        if sim is not None and accepted:
            description = _user_text(cfg.prompt)
            prompt = cfg.metadata.get(sim.ref2va.prompt_field) or build_ref2va_prompt(
                sim.ref2va.prompt_template, description
            )
            faithfulness = None
            try:
                scored = await score_rollout(
                    sim,
                    render,
                    gt_video,
                    prompt,
                    description=description,
                    tag=str(cfg.metadata.get("id", "")),
                    request_overrides=cfg.metadata.get(sim.ref2va.request_field) or None,
                )
                similarity = scored["similarity"]
                out = {**out, "similarity": scored}
                if "scores" in scored:  # rubric structured
                    faithfulness = scored["scores"]["faithfulness"]
                    logger.info(
                        "video_coder: judge %s motion_na=%s similarity=%.3f",
                        scored["scores"],
                        scored["motion_na"],
                        similarity,
                    )
            except Exception as e:
                if sim.on_error == "raise":
                    raise
                logger.exception("video_coder: similarity scoring failed; counting it as 0")
                similarity = 0.0
                out = {**out, "similarity": {"similarity": 0.0, "error": f"{type(e).__name__}: {e}"[:500]}}
            step, total = cfg.global_steps, cfg.total_training_steps
            reward = combine_reward(
                sim, accepted, similarity, faithfulness=faithfulness, global_step=step, total_steps=total
            )
            out = {**out, "exe_weight": exe_weight(sim, step, total)}
            if gated_out(sim, faithfulness):  # accepted, but the judge found the video unrelated to the description
                logger.info("video_coder: faithfulness %s <= gate %s: reward 0", faithfulness, sim.faithfulness_gate)
                out = {**out, "similarity": {**out["similarity"], "gated": True}}

        if cfg.debug_random_reward:
            out = {**out, "true_reward": reward}
            reward = float(random.random() < 0.5)
        return TaskResult(reward=reward, accuracy=float(accepted), finished=result.finished, extra_info=out)


def _user_text(messages: list[dict]) -> str:
    return next((m["content"] for m in messages if m.get("role") == "user" and isinstance(m.get("content"), str)), "")
