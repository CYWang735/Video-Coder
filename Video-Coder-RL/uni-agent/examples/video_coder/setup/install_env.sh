#!/usr/bin/env bash
# Install the Video-Coder RL / inference environment into the active Python 3.12 env (e.g. a fresh conda env):
#   conda create -n video-coder-rl python=3.12 -y && conda activate video-coder-rl
#   bash examples/video_coder/setup/install_env.sh
# Versions are the ones the paper's RL run used. Needs internet; run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"   # Video-Coder-RL/uni-agent
cd "$ROOT"
python -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else "need Python 3.12, got " + sys.version)'

# inference + training stack (vLLM pulls a matching torch)
pip install vllm==0.30.0 torch==2.13.0 transformers==5.17.0 "ray[default]==2.58.0" \
    tensordict==0.14.2 accelerate==1.15.0 peft==0.21.1 qwen-vl-utils==0.0.14
# verl's other requirements; its transformers / tensordict pins are older than Qwen3.5 needs
req=$(mktemp)
grep -vE '^(transformers|tensordict|TransferQueue)' verl/requirements.txt > "$req"
pip install -r "$req"
rm -f "$req"
# the bundled verl (Qwen3.5 fixes) and its TransferQueue data plane, without touching the versions above
pip install --no-deps -e ./verl
pip install --no-deps transferqueue==0.1.11
# Qwen3.5 GatedDeltaNet kernels: without them long rollouts are ~100x slower and run out of memory at ~100k tokens
pip install --no-deps fla-core==0.5.2 flash-linear-attention==0.5.2
# video reward / judge clients and data tools
pip install litellm httpx pandas pyarrow pyyaml jinja2 imageio-ffmpeg

python - <<'EOF'
import importlib
for m in ("torch", "vllm", "transformers", "ray", "verl", "transfer_queue", "fla", "litellm", "pandas"):
    mod = importlib.import_module(m)
    print(f"  {m:14s} {getattr(mod, '__version__', 'ok')}")
import uni_agent.tasks.video_coder.task  # noqa: F401  (the repo itself, from this directory)
print("RL environment ready")
EOF
