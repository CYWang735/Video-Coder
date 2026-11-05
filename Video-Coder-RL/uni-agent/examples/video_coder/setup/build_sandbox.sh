#!/usr/bin/env bash
# Build the Video-Coder Harness sandbox once (needs internet and Apptainer):
#   bash examples/video_coder/setup/build_sandbox.sh [DEST]          # DEST default: Video-Coder-RL/uni-agent/sandbox
#   FAKEROOT=1 bash examples/video_coder/setup/build_sandbox.sh      # if `apptainer build` needs --fakeroot here
# Produces DEST/bpy-agent.sif (Blender 4.5 / bpy) and DEST/mini-swe-agent-tool (the agent's Python, bound at
# /opt/mini-swe-agent), installs the harness' extra package into it, and fills these paths into the training and
# inference task configs where they still have their /path/to placeholders. DEST must be visible to every node.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # Video-Coder-RL/uni-agent
DEST=$(realpath -m "${1:-$ROOT/sandbox}")
mkdir -p "$DEST"
command -v apptainer >/dev/null || { echo "apptainer not found (https://apptainer.org)" >&2; exit 1; }

if [[ ! -f $DEST/bpy-agent.sif ]]; then
    echo "==> bpy-agent.sif"
    apptainer build ${FAKEROOT:+--fakeroot} "$DEST/bpy-agent.sif" "$ROOT/examples/video_coder/bpy-agent.def"
fi
if [[ ! -x $DEST/mini-swe-agent-tool/bin/python ]]; then
    echo "==> mini-swe-agent tool dir"
    bash "$ROOT/examples/mini_swe_agent/build_tool_apptainer.sh" build "$DEST/mini-swe-agent-tool.tar.gz"
    bash "$ROOT/examples/mini_swe_agent/build_tool_apptainer.sh" extract "$DEST/mini-swe-agent-tool.tar.gz" \
        "$DEST/mini-swe-agent-tool"
fi
echo "==> harness"
bash "$ROOT/examples/video_coder/install_harness.sh" "$DEST/mini-swe-agent-tool"

for cfg in examples/quickstart/training/task_config_video_coder_video_reward_apptainer.yaml \
           examples/video_coder/inference/task_config.yaml; do
    sed -i -e "s|/path/to/bpy-agent.sif|$DEST/bpy-agent.sif|" \
           -e "s|/path/to/mini-swe-agent-tool:|$DEST/mini-swe-agent-tool:|" \
           -e "s|/path/to/Video-Coder-RL/uni-agent/examples/video_coder:|$ROOT/examples/video_coder:|" "$ROOT/$cfg"
done
echo "sandbox ready in $DEST (paths filled into the task configs)"
