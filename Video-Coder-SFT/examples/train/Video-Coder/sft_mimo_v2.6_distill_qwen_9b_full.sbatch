#!/bin/bash -l
# Slurm run of sft_mimo_v2.6_distill_qwen_9b_full.sh on one node (4 H100).
# Usage: sbatch examples/train/Video-Coder/sft_mimo_v2.6_distill_qwen_9b_full.sbatch     (from the ms-swift directory)
# The MiMo training script is single-node only (fixed sequence_parallel_size 4 / grad-accum 16); do not pass -N.
# One epoch (228 steps) on one node took ~57 s/step for 9B once fla's Triton kernels were compiled (~3.6 h); on a
# fresh node the first tens of steps are slower while they compile for each new sequence length, so expect 4-4.5 h.
#
# srun starts one task on the node. It unpacks the ms-swift env onto the node's local /tmp if it is not there yet
# (unpack_ms_swift_env.sh, a few minutes), then runs sft_mimo_v2.6_distill_qwen_9b_full.sh.
# Two epochs = 456 optimizer steps; ~8 h expected on one node (one Qwen3.5-9B epoch took ~4 h).
# Log (stdout + stderr of all nodes, lines prefixed [nodeN]): /hnvme/workspace/v189bd13-VOpera/logs/<job-name>-<jobid>.txt
#SBATCH --account=v189bd
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=vc-sft-mimo
#SBATCH --output=/hnvme/workspace/v189bd13-VOpera/logs/%x-%j.txt
set -euo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export SWIFT_DIR=/home/hpc/v189bd/v189bd13/wangcy/ms-swift
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, master $MASTER_ADDR:$MASTER_PORT, model MiMo-V2.6-Distill-Qwen-9B"

# Give each task the whole node explicitly (128 cores, 4 GPUs) instead of relying on srun inheriting sbatch's
# cpus-per-task, which newer Slurm versions do not do; with 1 core the dataloader / image preprocessing would crawl.
srun --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:4 --kill-on-bad-exit=1 bash -lc '
    set -euo pipefail
    ENV=/tmp/$USER/envs/ms-swift
    [ -d "$ENV" ] || bash /home/hpc/v189bd/v189bd13/wangcy/unpack_ms_swift_env.sh "$ENV"
    module load cuda/13.0.2
    export CONDA_PREFIX=$ENV PATH=$ENV/bin:$PATH
    export NNODES=$SLURM_NNODES NODE_RANK=$SLURM_NODEID
    cd "$SWIFT_DIR"
    bash examples/train/Video-Coder/sft_mimo_v2.6_distill_qwen_9b_full.sh 2>&1 | sed -u "s/^/[node$NODE_RANK] /"
    exit ${PIPESTATUS[0]}
'
