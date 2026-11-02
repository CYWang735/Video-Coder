#!/bin/bash -l
# Two-node (2 x 4 H100) run of sft_qwen3_8_27b_full.sh.
# Usage: sbatch examples/train/Video-Coder/sft_qwen3_8_27b_full.sbatch        (from the ms-swift directory)
#
# srun starts one task per node. Each task unpacks the ms-swift env onto its node's local /tmp if needed, then runs
# the training script with NNODES / NODE_RANK / MASTER_ADDR set so torchrun on both nodes joins one 8-GPU job: within
# a node the 4 GPUs are one sequence-parallel group, the two nodes are data-parallel replicas, and grad-accum 8 keeps
# the global batch at 16 (228 steps per epoch).
#
# If training fails, the job does NOT end: it keeps the nodes and their GPUs for HOLD_ON_FAILURE seconds (default:
# until the time limit) so the failure can be debugged and the run restarted in place with
#   srun --jobid=<jobid> --overlap ... (see the log for the node list)
# Cancel with scancel once done.
# Log (stdout + stderr of all nodes, lines prefixed [nodeN]):
#   /hnvme/workspace/v189bd13-VOpera/ckpts/qwen3.8-27B-Video-Coder/<job-name>-<jobid>.txt
#SBATCH --account=v189bd
#SBATCH --partition=h100
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=1-00:00:00
#SBATCH --job-name=vc-sft-qwen38-27b
#SBATCH --output=/hnvme/workspace/v189bd13-VOpera/ckpts/qwen3.8-27B-Video-Coder/%x-%j.txt
set -uo pipefail

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export SWIFT_DIR=/home/hpc/v189bd/v189bd13/wangcy/ms-swift
HOLD_ON_FAILURE=${HOLD_ON_FAILURE:-86400}
echo "job $SLURM_JOB_ID on $SLURM_JOB_NODELIST, master $MASTER_ADDR:$MASTER_PORT, model Qwen3.8-27B"

# Give each task the whole node explicitly (128 cores, 4 GPUs) instead of relying on srun inheriting sbatch's
# cpus-per-task, which newer Slurm versions do not do.
srun --ntasks-per-node=1 --cpus-per-task=128 --gres=gpu:4 --kill-on-bad-exit=1 bash -lc '
    set -euo pipefail
    ENV=/tmp/$USER/envs/ms-swift
    [ -d "$ENV" ] || bash /home/hpc/v189bd/v189bd13/wangcy/unpack_ms_swift_env.sh "$ENV"
    module load cuda/13.0.2
    export CONDA_PREFIX=$ENV PATH=$ENV/bin:$PATH
    export NNODES=$SLURM_NNODES NODE_RANK=$SLURM_NODEID
    cd "$SWIFT_DIR"
    bash examples/train/Video-Coder/sft_qwen3_8_27b_full.sh 2>&1 | sed -u "s/^/[node$NODE_RANK] /"
    exit ${PIPESTATUS[0]}
'
rc=$?
if [ $rc -ne 0 ]; then
    echo "TRAINING FAILED (exit $rc) at $(date). Holding nodes $SLURM_JOB_NODELIST for up to ${HOLD_ON_FAILURE}s for debugging;"
    echo "  run on them with: srun --jobid=$SLURM_JOB_ID --overlap -N $SLURM_NNODES ..., cancel with: scancel $SLURM_JOB_ID"
    sleep "$HOLD_ON_FAILURE"
fi
exit $rc
