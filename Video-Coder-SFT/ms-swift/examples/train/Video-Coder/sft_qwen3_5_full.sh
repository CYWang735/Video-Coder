MODEL=Qwen/Qwen3.5-9B
OUT=./qwen35-sft
DATA=./data/
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

# Optional kernels; training also works without them, only slower / with more memory.
python -c "import fla" 2>/dev/null || echo "WARNING: flash-linear-attention not installed; Qwen3.5 linear attention uses the slow torch fallback"
if python -c "import flash_attn" 2>/dev/null; then ATTN=flash_attn; else ATTN=sdpa; echo "WARNING: flash-attn not installed; using sdpa"; fi


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
