#!/bin/bash -l
# Slurm run of sft_qwen3_5_full.sh, by default on one node (4 H100).
# Usage: sbatch examples/train/Video-Coder/sft_qwen3_5_full.sbatch [4B|9B]             (from the ms-swift directory)
#        sbatch -N 2 examples/train/Video-Coder/sft_qwen3_5_full.sbatch 9B               (other node counts: 1, 2, 4, 8, 16)
# One epoch (228 steps) on one node took ~57 s/step for 9B once fla's Triton kernels were compiled (~3.6 h); on a
# fresh node the first tens of steps are slower while they compile for each new sequence length, so expect 4-4.5 h.
#
# srun starts one task per node. Each task unpacks the ms-swift env onto its node's local /tmp if it is not there
# yet (unpack_ms_swift_env.sh, a few minutes), then runs sft_qwen3_5_full.sh with NNODES / NODE_RANK / MASTER_ADDR
# set, so torchrun on all nodes joins one job. Within a node the 4 GPUs are one sequence-parallel group; the nodes
# are data-parallel replicas, and the training script divides gradient_accumulation_steps by the node count so the
# global batch stays 16 (4 nodes: 4 x 4; same 228 optimizer steps per epoch as on one node).
# Log (stdout + stderr of all nodes, lines prefixed [nodeN]): /hnvme/workspace/v189bd13-VOpera/checkpoints/qwen3.5-9B-Video-Coder/<job-name>-<jobid>.txt
# (not logs/: that directory sits on an MDT whose file-count grant is exhausted)
#SBATCH --account=v189bd
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=vc-sft-qwen35
#SBATCH --output=/hnvme/workspace/v189bd13-VOpera/checkpoints/qwen3.5-9B-Video-Coder/%x-%j.txt
set -euo pipefail

export SIZE=${1:-9B}
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export SWIFT_DIR=/home/hpc/v189bd/v189bd13/wangcy/ms-swift
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, master $MASTER_ADDR:$MASTER_PORT, model Qwen3.5-$SIZE"

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
    bash examples/train/Video-Coder/sft_qwen3_5_full.sh "$SIZE" 2>&1 | sed -u "s/^/[node$NODE_RANK] /"
    exit ${PIPESTATUS[0]}
'
