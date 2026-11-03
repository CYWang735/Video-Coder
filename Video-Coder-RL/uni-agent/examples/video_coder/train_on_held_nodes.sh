#!/bin/bash
# Start training on nodes an existing slurm_video_coder.sbatch job still holds (H3 and the Ray head are up):
#   JOBID=1108975 HEAD_NODE=h14-13 JOB_DIR=~/wangcy/uni_agent_logs/vc_sbatch1_v3_n8_s100 \
#       bash examples/video_coder/train_on_held_nodes.sh
# JOB_DIR is the log dir of that job: its job_env.sh (env, proxy, Ray address) and task_config.yaml (H3 URL) are
# reused. The new run logs to ~/wangcy/uni_agent_logs/$EXP_NAME. Same training settings as submit_h3_1684.sh.
set -euo pipefail
: "${JOBID:?set JOBID}" "${HEAD_NODE:?set HEAD_NODE (the Ray head / training node)}" "${JOB_DIR:?set JOB_DIR}"
EXP_NAME=${EXP_NAME:-vc_h3_1684_r132k}
RUN_DIR=$HOME/wangcy/uni_agent_logs/$EXP_NAME
mkdir -p "$RUN_DIR/rollouts"
srun --jobid="$JOBID" --overlap -w "$HEAD_NODE" bash -c "source $JOB_DIR/job_env.sh; \
    MODEL_PATH=${MODEL_PATH:-/hnvme/workspace/v189bd13-VOpera/checkpoints/qwen3.5-9B-Video-Coder/v3-20261022-064456/checkpoint-456} \
    CKPTS_DIR=${CKPTS_DIR:-/hnvme/workspace/v189bd13-VOpera/ckpts/$EXP_NAME} \
    TRAIN_FILE=${TRAIN_FILE:-$HOME/data/uni_agent/video_coder_h3_1684_train.parquet} \
    TEST_FILE=${TEST_FILE:-$HOME/data/uni_agent/video_coder_h3_1684_val.parquet} \
    TASK_CONFIG=${TASK_CONFIG:-$JOB_DIR/task_config.yaml} EXP_NAME=$EXP_NAME AGENT_LOG_DIR=$RUN_DIR/rollouts \
    NNODES=1 NGPUS_PER_NODE=4 GEN_TP=1 GATEWAY_COUNT=${GATEWAY_COUNT:-2} CONCURRENCY=${CONCURRENCY:-32} \
    SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-qwen3.5-9B-Video-Coder-v3} \
    TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-2} N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8} PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-2} \
    NUM_WARMUP_BATCHES=0 MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-135168} \
    SAVE_FREQ=${SAVE_FREQ:-2} MAX_CKPT_TO_KEEP=${MAX_CKPT_TO_KEEP:-2} \
    bash examples/video_coder/train_video_coder_rl.sh" 2>&1 | tee "$RUN_DIR/submit.log"
