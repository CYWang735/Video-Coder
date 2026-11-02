#!/usr/bin/env bash
# Full-parameter SFT of Qwen/Qwen3.8-27B on the Video-Coder Blender agent trajectories (sft/gemini38_flash).
# Usage: bash sft_qwen3_8_27b_full.sh        (normally through sft_qwen3_8_27b_full.sbatch on two 4x H100 nodes)
#   Multi-node: run once per node with NNODES / NODE_RANK / MASTER_ADDR / MASTER_PORT set (torchrun pass-through).
#
# Model: Qwen3.8-27B (Qwen3_5ForConditionalGeneration, 64 layers, hidden 5120, 24 attention / 4 KV heads, 16 / 48
# linear-attention K / V heads, so sequence_parallel_size 4 divides every head count). Template qwen3_8: like qwen3_5
# but it prepends the reasoning-effort instruction (default xhigh, same as the model's own chat_template.jinja) to
# the system message and keeps historical <think> content.
#
# Data: 3644 multi-turn bash-tool trajectories from gemini-3.8-flash (leading bold summary titles removed);
# 14k-101k tokens each (median 43k). MAX_LENGTH 131072 keeps all of them. Global batch 16 -> 228 steps per epoch.
#
# Memory: full fine-tuning keeps ~16 bytes/param of weights, grads and AdamW states (~445 GB for 27.8B params).
# DS=zero3_offload (default here) keeps the fp32 optimizer states in CPU RAM (~167 GB per node on two nodes), so each
# GPU holds only its ZeRO-3 shard of bf16 weights and grads (~14 GB on 8 GPUs) plus activations; plain zero3 would put
# ~56 GB of states on every GPU before activations, which very likely does not fit at 100k-token sequences.
# Peak memory has not been measured yet: watch memory(GiB) in the first steps.
source /home/hpc/v189bd/v189bd13/wangcy/uni-agent/proxy.sh

MODEL=/hnvme/workspace/v189bd13-VOpera/ckpts/base_models/Qwen3.8-27B
# Under ckpts/ (not checkpoints/): checkpoints/ sits on an MDT whose file-count grant is exhausted.
OUT=/hnvme/workspace/v189bd13-VOpera/ckpts/qwen3.8-27B-Video-Coder
DATA=/home/hpc/v189bd/v189bd13/wangcy/Video-Coder-sftdata/sft/gemini38_flash/train.json
MAX_LENGTH=${MAX_LENGTH:-131072}
DS=${DS:-zero3_offload}

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
    --template qwen3_8 \
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
    --save_steps 100 \
    --save_only_model true \
    --logging_steps 1 \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --output_dir ${OUT}
