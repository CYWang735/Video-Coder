#!/usr/bin/env bash
# LoRA SFT on a local Qwen3.5 checkpoint (adapted from the Qwen3-4B-Instruct-2507 example).
# Usage: bash sft_qwen3_5_lora.sh [4B|9B]
# Run on a compute node; datasets are fetched from ModelScope, so the proxy is needed.
source /home/hpc/v189bd/v189bd13/wangcy/uni-agent/proxy.sh

SIZE=${1:-4B}
MODEL=/hnvme/workspace/v189bd13-VOpera/models/Qwen3.5-${SIZE}
OUT=/hnvme/workspace/v189bd13-VOpera/checkpoints/qwen3.5-${SIZE}-lora

CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ${MODEL} \
    --tuner_type lora \
    --dataset 'AI-ModelScope/alpaca-gpt4-data-zh#500' \
              'AI-ModelScope/alpaca-gpt4-data-en#500' \
              'swift/self-cognition#500' \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir ${OUT} \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot
