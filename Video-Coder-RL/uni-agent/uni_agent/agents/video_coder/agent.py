"""video_coder: the Video-Coder harness (mini-swe-agent + its guards and tools) run inside the sandbox.

Launched like :mod:`uni_agent.agents.mini_swe_agent`. The harness lives in this repo under
``examples/video_coder`` (run_agent.py + video_coder_harness/, bin/, configs/) and is bound read-only at /opt/video-coder;
the mini-swe-agent tool dir (python + packages) is bound at /opt/mini-swe-agent. run_agent.py reads a task
config from stdin and prints the result JSON. The agent drives the gateway policy through
mini-swe-agent's tool-call loop with the ``run_blender`` / ``check_script`` tools and the Video-Coder harness's guards, and
the result carries the Video-Coder harness's acceptance verdict (``accepted``).
"""

from __future__ import annotations

import base64
import json
import logging
from typing import TYPE_CHECKING, Any

from pydantic import Field

from ..base import Agent, AgentConfig, AgentResult
from ..mini_swe_agent.agent import build_agent_command, parse_agent_result
from ..registry import register_agent

if TYPE_CHECKING:
    from uni_agent.sandbox import Sandbox

logger = logging.getLogger(__name__)


class VideoCoderConfig(AgentConfig):
    name: str = "video_coder"
    step_limit: int = Field(default=30, description="Max agent steps (teacher.yaml uses 30).")
    run_timeout: float = Field(default=3600.0, description="Wallclock cap (s) on the agent process.")
    tool_python: str = Field(description="Tool-dir python that runs the runner (e.g. /opt/mini-swe-agent/bin/python).")
    run_agent_script: str = Field(
        default="/opt/video-coder/run_agent.py", description="examples/video_coder/run_agent.py, as mounted."
    )
    harness_config: str = Field(
        default="/opt/video-coder/configs/teacher.yaml", description="Base harness config (examples/video_coder/configs)."
    )
    overlays: list[str] = Field(
        default_factory=lambda: ["/opt/video-coder/configs/uni_container.yaml"],
        description="Overlays applied on the base config, in order.",
    )
    workspace: str = Field(default="/workspace/task", description="Agent cwd inside the sandbox (must not be under /home or /root).")


@register_agent("video_coder")
class VideoCoderAgent(Agent):
    config_model = VideoCoderConfig

    async def run(
        self, *, sandbox: Sandbox, messages: list[dict[str, Any]], workdir: str | None = None
    ) -> AgentResult:
        cfg: VideoCoderConfig = self.config  # type: ignore[assignment]
        if cfg.model.base_url is None:
            raise ValueError("video_coder: config.model.base_url is not set (the gateway endpoint)")
        user = [m["content"] for m in messages if m.get("role") == "user"]
        if not user:
            raise ValueError("video_coder requires a 'user' message (the scene description)")
        request = {
            "task": user[0],
            "gateway_url": cfg.model.base_url,
            "workspace": workdir or cfg.workspace,
            "config": cfg.harness_config,
            "overlays": cfg.overlays,
            "agent": {"step_limit": cfg.step_limit},
        }
        command = build_agent_command(
            config_b64=base64.b64encode(json.dumps(request).encode()).decode(),
            tool_python=cfg.tool_python,
            run_agent_script=cfg.run_agent_script,
        )
        result = await sandbox.exec_shell(command, timeout=cfg.run_timeout)
        info = parse_agent_result(result.stdout or "")
        logger.info(
            "video_coder: exit_status=%s accepted=%s blocked=%s rc=%s",
            info.get("exit_status"), info.get("accepted"), info.get("n_blocked"), result.exit_code,
        )
        return AgentResult(
            output=info,
            transcript=list(messages),
            info={"step_limit": cfg.step_limit, "exit_status": info.get("exit_status")},
            finished=info.get("exit_status") == "Submitted",
        )
