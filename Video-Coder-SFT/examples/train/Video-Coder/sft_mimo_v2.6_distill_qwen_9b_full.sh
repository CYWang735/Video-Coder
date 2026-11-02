#!/usr/bin/env bash
# Full-parameter SFT of XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B on the Video-Coder Blender agent trajectories
# (sft/gemini38_flash).
# Usage: bash sft_mimo_v2.6_distill_qwen_9b_full.sh
#
# Model: an SFT of Qwen3.5-9B by Xiaomi MiMo, with the same architecture (Qwen3_5ForConditionalGeneration) and
# tokenizer. ms-swift does not list it, and cannot pick a model_type (qwen3_5 / qwen3_5_emb / ovis_ocr2 / ...) or a
# template (qwen3_5 / qwen3_8) from its config on its own, hence --model_type qwen3_5 --template qwen3_5. Its own chat_template.jinja uses a different tool-call format (single-line <tool_call>, a
# separate `tool` role, tools in their own system turn); it is trained here with the Qwen3.5 template instead, whose
# encodings are identical to Qwen3.5-9B's (input_ids, labels and pixel values checked on a gemini38_flash sample).
# Serve the result with Qwen3.5-9B's chat template to match training, e.g.
#   vllm serve <ckpt> --chat-template /hnvme/workspace/v189bd13-VOpera/models/Qwen3.5-9B/chat_template.jinja
#
# Data: 3644 multi-turn bash-tool trajectories from gemini-3.8-flash, with rendered frames in the tool responses;
# 14k-101k tokens each (median 43k, p90 62k), 164M tokens per epoch. Samples longer than MAX_LENGTH are dropped
# (truncation_strategy delete): 131072 keeps all of them, 98304 keeps 3642, 65536 keeps 3397.
# 3644 samples / global batch 16 = ~228 optimizer steps per epoch.
#
# 4x H100 (96GB): the four GPUs form one sequence-parallel group (each holds a quarter of every sequence, ~25k tokens
# for the longest 101k sample), so data parallelism is 1 and the global batch is gradient_accumulation_steps.
# ZeRO-3 shards weights/grads/optimizer states across the four GPUs, ~36GB/GPU for 9B. If it runs out of memory,
# rerun with DS=zero3_offload. Peak memory has not been measured yet: run a few steps first and watch memory(GiB).
source /home/hpc/v189bd/v189bd13/wangcy/uni-agent/proxy.sh

MODEL=/hnvme/workspace/v189bd13-VOpera/models/MiMo-V2.6-Distill-Qwen-9B
DATA=/home/hpc/v189bd/v189bd13/wangcy/Video-Coder-sftdata/sft/gemini38_flash/train.json
OUT=/hnvme/workspace/v189bd13-VOpera/checkpoints/mimo-v2.6-distill-qwen-9B-gemini38-full
MAX_LENGTH=${MAX_LENGTH:-131072}
DS=${DS:-zero3}

# Optional kernels; training also works without them, only slower / with more memory.
python -c "import fla" 2>/dev/null || echo "WARNING: flash-linear-attention not installed; Qwen3.5 linear attention uses the slow torch fallback"
if python -c "import flash_attn" 2>/dev/null; then ATTN=flash_attn; else ATTN=sdpa; echo "WARNING: flash-attn not installed; using sdpa"; fi

# Triton (used by DeepSpeed / fla) caches on NFS by default, which can hang; keep it on node-local disk.
export TRITON_CACHE_DIR=/tmp/$USER/triton_cache

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
CELOSS_PARALLEL_SIZE=2048 \
NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model ${MODEL} \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --tuner_type full \
    --dataset ${DATA} \
    --agent_template qwen3_5 \
    --torch_dtype bfloat16 \
    --attn_impl ${ATTN} \
    --gradient_checkpointing true \
    --sequence_parallel_size 4 \
    --max_length ${MAX_LENGTH} \
    --truncation_strategy delete \
    --loss_scale ignore_think_prefix \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
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
