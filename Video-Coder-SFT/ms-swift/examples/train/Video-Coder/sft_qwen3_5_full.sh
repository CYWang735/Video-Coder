#!/usr/bin/env bash
# Full-parameter SFT of local Qwen3.5 on the Video-Coder Blender agent trajectories (sft/gemini38_flash).
# Usage: bash sft_qwen3_5_full.sh [4B|9B]
#   Multi-node: run it once per node with NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT set (torchrun pass-through);
#   unset, it trains on this node alone.
#
# Data: 3644 multi-turn bash-tool trajectories from gemini-3.8-flash, with rendered frames in the tool responses;
# 14k-101k tokens each (median 43k, p90 62k), 164M tokens per epoch. Samples longer than MAX_LENGTH are dropped
# (truncation_strategy delete): 131072 keeps all of them, 98304 keeps 3642, 65536 keeps 3397.
# 3644 samples / global batch 16 = 228 optimizer steps per epoch (456 for the 2 epochs trained here).
#
# Each node has 4x H100 (96GB). The four GPUs of a node form one sequence-parallel group (each holds a quarter of every
# sequence, ~25k tokens for the longest 101k sample; ms-swift puts consecutive ranks in a group, so SP traffic stays
# on the node's NVLink), and the nodes are data-parallel replicas. gradient_accumulation_steps is derived so the
# global batch stays GLOBAL_BATCH=16 for any node count (16 on one node, 8 on two, 4 on four). ZeRO-3 shards
# weights/grads/optimizer states across the four GPUs: ~16GB/GPU for 4B, ~36GB/GPU for 9B. If 9B runs out of memory,
# rerun with DS=zero3_offload. Peak memory at these lengths has not been measured yet: run a few steps first and
# watch memory(GiB) in the log.
source /home/hpc/v189bd/v189bd13/wangcy/uni-agent/proxy.sh

SIZE=${1:-9B}
MODEL=/hnvme/workspace/v189bd13-VOpera/models/Qwen3.5-${SIZE}
OUT=/hnvme/workspace/v189bd13-VOpera/checkpoints/qwen3.5-${SIZE}-Video-Coder
DATA=/home/hpc/v189bd/v189bd13/wangcy/Video-Coder-sftdata/sft/gemini38_flash/train.json
MAX_LENGTH=${MAX_LENGTH:-131072}
DS=${DS:-zero3}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}
GPUS_PER_NODE=4
SP=4
GLOBAL_BATCH=16
DP=$(( NNODES * GPUS_PER_NODE / SP ))
GRAD_ACC=$(( GLOBAL_BATCH / DP ))
echo "node $NODE_RANK/$NNODES (master $MASTER_ADDR:$MASTER_PORT): data-parallel $DP x grad-accum $GRAD_ACC = global batch $(( DP * GRAD_ACC ))"

# Optional kernels; training also works without them, only slower / with more memory.
python -c "import fla" 2>/dev/null || echo "WARNING: flash-linear-attention not installed; Qwen3.5 linear attention uses the slow torch fallback"
if python -c "import flash_attn" 2>/dev/null; then ATTN=flash_attn; else ATTN=sdpa; echo "WARNING: flash-attn not installed; using sdpa"; fi

# Triton (used by DeepSpeed / fla) caches on NFS by default, which can hang; keep it on node-local disk.
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CELOSS_PARALLEL_SIZE=2048 \
NNODES=${NNODES} \
NODE_RANK=${NODE_RANK} \
MASTER_ADDR=${MASTER_ADDR} \
MASTER_PORT=${MASTER_PORT} \
NPROC_PER_NODE=${GPUS_PER_NODE} \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model ${MODEL} \
    --model_type qwen3_5 \
    --tuner_type full \
    --dataset ${DATA} \
    --agent_template qwen3_5 \
    --torch_dtype bfloat16 \
    --attn_impl ${ATTN} \
    --gradient_checkpointing true \
    --sequence_parallel_size ${SP} \
    --max_length ${MAX_LENGTH} \
    --truncation_strategy delete \
    --loss_scale ignore_think_prefix \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps ${GRAD_ACC} \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --deepspeed ${DS} \
    --eval_strategy no \
    --save_steps 50 \
    --save_only_model true \
    --logging_steps 1 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --output_dir ${OUT}
