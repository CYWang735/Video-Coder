"""Run the Video-Coder-sftdata harness (vcsft GuardedAgent + run_blender/check_script) inside the sandbox.

Input:  task config JSON from **stdin**
    - task: str          -- the scene description the agent must turn into a bpy script
    - gateway_url: str   -- OpenAI-compatible endpoint (the gateway)
    - workspace: str     -- agent working directory (created; main.py ends up here)
    - config: str        -- harness base config (default configs/teacher.yaml next to this file)
    - overlays: [str]    -- extra YAML overlays applied after the base config
    - agent: dict        -- agent overrides (e.g. step_limit)

Output: result JSON (last line of stdout): exit_status, accepted, static check, final render report, guard counters,
and final_video (path of the accepted re-render inside the sandbox, else null).
`accepted` is vcsft's acceptance: submitted + static check ok + the final script re-executed from scratch
in a fresh directory renders a valid video.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MSWEA_SILENT_STARTUP", "1")
# This directory (examples/video_coder: run_agent.py + vcsft/ bin/ configs/), bound read-only into the sandbox.
HARNESS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(HARNESS_ROOT))
TOOL_ROOT = "/opt/mini-swe-agent"  # mini-swe-agent tool dir (python + packages)
BPY_PACKAGE_DIR = "/usr/local/lib/python3.11/site-packages/bpy"  # in bpy-agent.sif
FINAL_DIR = Path("/tmp/video_coder_final")  # final re-render (final_run/output.mp4), sandbox-private


def _fail(msg: str, exit_status: str = "error") -> None:
    sys.stdout.write(json.dumps({"exit_status": exit_status, "accepted": False, "error": msg}) + "\n")
    sys.stdout.flush()


def main() -> None:
    try:
        req = json.load(sys.stdin)
        import yaml
        from minisweagent.environments import get_environment
        from minisweagent.models import get_model

        from vcsft import generate as g
        from vcsft.agent import GuardedAgent
        from vcsft.static_check import check_file

        cfg = yaml.safe_load(Path(req.get("config") or HARNESS_ROOT / "configs" / "teacher.yaml").read_text())
        for overlay in req.get("overlays", []):
            cfg = g.deep_merge(cfg, yaml.safe_load(Path(overlay).read_text()))
        cfg["agent"] = {**cfg["agent"], **req.get("agent", {})}
        harness = cfg["harness"]

        # The policy is whatever the gateway serves (Chat Completions). Keep the rest of the config's model section
        # (observation_template / format_error_template are part of what the model sees), swap only the endpoint.
        cfg["model"] = {
            **cfg["model"],
            "model_class": "vcsft.models.VisualChatModel" if harness.get("visual_feedback") else "litellm",
            "model_name": "openai/default",
            "cost_tracking": "ignore_errors",
            "model_kwargs": {**cfg["model"].get("model_kwargs", {}), "api_base": req["gateway_url"],
                             "api_key": "not-needed", "drop_params": True},
        }

        workspace = Path(req["workspace"])
        workspace.mkdir(parents=True, exist_ok=True)
        env_cfg = dict(cfg["environment"])
        env_cfg["cwd"] = str(workspace)
        env_cfg["env"] = g.agent_env_vars(env_cfg.get("env", {}), harness)
        env = get_environment(env_cfg, default_type="local")
        model = get_model(config=dict(cfg["model"]))
        # vcsft's markers are the bpy env dirs; here bpy is the image's system python (bpy_python's env dir is /usr),
        # and a "/usr" marker would block any command mentioning /usr. Mark the harness, the tool dir and bpy instead.
        markers = [str(HARNESS_ROOT), TOOL_ROOT, BPY_PACKAGE_DIR]
        agent = GuardedAgent(model, env, **cfg["agent"], harness_markers=markers, workdir=str(workspace.resolve()))

        task = req["task"]
        error = None
        try:
            info = agent.run(task=task, **g.template_vars(cfg))
            exit_status = info.get("exit_status", "")
        except Exception as e:
            exit_status, error = type(e).__name__, str(e)

        script = workspace / harness["script_name"]
        check = {"ok": False, "errors": [], "warnings": [], "stats": {}}
        execution = None
        try:
            check = check_file(script)
            if error is None and exit_status == "Submitted" and check["ok"] and harness["blender_available"]:
                # Kept (outside the agent's workspace) so the task can copy the accepted render out of
                # the sandbox, e.g. for the video reward.
                FINAL_DIR.mkdir(parents=True, exist_ok=True)
                execution = g.final_execution(script, FINAL_DIR, harness)
        except Exception as e:  # harness failure after the agent finished: not accepted, but report it
            exit_status, error = "HarnessError", f"{type(e).__name__}: {e}"

        accepted = (error is None and exit_status == "Submitted" and bool(check["ok"])
                    and (execution is None or bool(execution["ok"])))
        result = {
            "exit_status": exit_status,
            "accepted": accepted,
            "error": error,
            "static_check": {"ok": check["ok"], "errors": check["errors"], "warnings": check["warnings"]},
            "execution": execution,
            "final_video": str(FINAL_DIR / "final_run" / harness["output_video"]) if accepted and execution else None,
            "n_blocked": agent.n_blocked,
            "max_repeats": agent.max_repeats,
            "n_format_errors": g.count_format_errors(agent.messages),
            "model_stats": {"instance_cost": agent.cost, "api_calls": agent.n_calls},
        }
        sys.stdout.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")
        sys.stdout.flush()
    except Exception as e:
        _fail(f"{type(e).__name__}: {e}", exit_status=type(e).__name__)


if __name__ == "__main__":
    main()
