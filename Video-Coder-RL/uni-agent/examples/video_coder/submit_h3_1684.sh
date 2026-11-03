#!/bin/bash
# Submit a 2-node video_coder RL run (1 H3 node + 1 training node) with sbatch:
#   bash examples/video_coder/submit_h3_1684.sh
# Policy: Qwen3.5-9B Video-Coder SFT v3. Data: minimax_h3_gpt_1684 (1660 train / 24 val).
# Response cap 132k (total 140k with the 8k prompt cap). Any variable below can be overridden from the environment.
set -euo pipefail
cd "$(dirname "$0")/../.."
EXP_NAME=${EXP_NAME:-vc_h3_1684_r132k} \
MODEL_PATH=${MODEL_PATH:-/hnvme/workspace/v189bd13-VOpera/checkpoints/qwen3.5-9B-Video-Coder/v3-20261022-064456/checkpoint-456} \
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.5-9B-Video-Coder-v3} \
TRAIN_FILE=${TRAIN_FILE:-$HOME/data/uni_agent/video_coder_h3_1684_train.parquet} \
TEST_FILE=${TEST_FILE:-$HOME/data/uni_agent/video_coder_h3_1684_val.parquet} \
TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-2} N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8} \
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2} NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-0} \
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-135168} STEP_LIMIT=${STEP_LIMIT:-100} KEEP_SIM_FILES=${KEEP_SIM_FILES:-true} \
SAVE_FREQ=${SAVE_FREQ:-2} MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-2} \
PRECHECK=${PRECHECK-"pip install -q --target /tmp/\$USER/pytest_pkgs pytest && PYTHONPATH=/tmp/\$USER/pytest_pkgs:\${PYTHONPATH:-} torchrun --standalone --nproc_per_node=2 -m pytest -q -p no:cacheprovider tests/uni_agent/framework/test_fsdp2_fused_lm_head.py"} \
HOLD_ON_FAILURE=${HOLD_ON_FAILURE:-21600} \
sbatch examples/video_coder/slurm_video_coder.sbatch
