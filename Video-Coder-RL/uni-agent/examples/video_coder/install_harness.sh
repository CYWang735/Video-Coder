#!/usr/bin/env bash
# Prepare the mini-swe-agent tool dir for the video_coder agent.
#
#   bash install_harness.sh [TOOL_DIR]
#
# TOOL_DIR the tool dir bound at /opt/mini-swe-agent (default $HOME/wangcy/mini-swe-agent-tool-2.4.6;
#          needs mini-swe-agent >= 2.4 -- the Video-Coder harness's GuardedAgent relies on its agent API)
#
# The harness itself is this directory (run_agent.py, video_coder_harness/, bin/, configs/), bound read-only at
# /opt/video-coder by task_config_video_coder_apptainer.yaml; nothing is copied. This script only adds the
# one extra package the Video-Coder harness needs (imageio-ffmpeg) and checks the imports. Needs internet once (login node).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL="${1:-$HOME/wangcy/mini-swe-agent-tool-2.4.6}"
[[ -x "${TOOL}/bin/python" ]] || { echo "not a tool dir: ${TOOL}" >&2; exit 1; }
"${TOOL}/bin/python" -I -c "import minisweagent as m, sys; v=tuple(int(x) for x in m.__version__.split('.')[:2]); sys.exit(0 if v >= (2, 4) else 'tool dir has mini-swe-agent ' + m.__version__ + ', need >= 2.4')"
"${TOOL}/bin/python" -I -m pip install --no-cache-dir --disable-pip-version-check -q "imageio-ffmpeg==0.6.0"
"${TOOL}/bin/python" -I -m pip check
"${TOOL}/bin/python" -I -B -c "
import sys; sys.path.insert(0, '${HERE}')
import video_coder_harness.agent, video_coder_harness.models, video_coder_harness.generate, video_coder_harness.blender_exec, video_coder_harness.static_check
print('video_coder harness imports ok')"
