# Overrides for the NHR@FAU Helma cluster, loaded by config.sh when H3_SITE_ENV points here.
# Everything lives on node-local /tmp (wiped when the job ends): /home/hpc has little room left
# and /hnvme is over its file-count quota.

HELMA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ROOT="/tmp/$USER"

export http_proxy=http://proxy.nhr.fau.de:80 https_proxy=http://proxy.nhr.fau.de:80
export HTTP_PROXY=$http_proxy HTTPS_PROXY=$https_proxy

ENV_DIR="${ENV_DIR:-$TMP_ROOT/envs/sglang}"
CACHE_ROOT="${CACHE_ROOT:-$TMP_ROOT/cache}"
SERVER_WORKDIR="${SERVER_WORKDIR:-$TMP_ROOT/server}"

# conda comes from the python module; its package cache must not go to the full home quota.
if ! command -v conda >/dev/null; then
    type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
    module load python/3.12-conda
fi
export CONDA_PKGS_DIRS="$TMP_ROOT/conda_pkgs"
mkdir -p "$CONDA_PKGS_DIRS"

# Compute nodes have no /usr/local/cuda; use the cuda/13.0.2 module (matches torch cu130).
export CUDA_HOME="${CUDA_HOME:-/apps/SPACK/0.23.0/opt/linux-almalinux9-zen4/gcc-11.4.1/cuda-13.0.2-qm2yxli4kqp6qriu54wqrvwkkdoxdr4f}"

# SGLang and its kernel libraries also write JIT builds straight to ~/.cache, which is a symlink
# into /hnvme here; point HOME at /tmp for the server and client processes.
export HOME="$CACHE_ROOT/home"
mkdir -p "$HOME"

# The local download at SRC_MODEL has no usable processor/ (Ref2VA: dangling symlink, FL2VA: missing).
# Build an overlay on /tmp that links every component to SRC_MODEL and copies processor/ from
# model_patch/ (fetched from HF), for both partitions.
SRC_MODEL="${SRC_MODEL:-/hnvme/workspace/v189bd13-VOpera/models/MiniMax-H3}"
if [ -z "${MODEL_PATH:-}" ]; then
    MODEL_PATH="$TMP_ROOT/models/MiniMax-H3"
    mkdir -p "$MODEL_PATH"
    ln -sfn "$SRC_MODEL/model_index.json" "$MODEL_PATH/model_index.json"
    for part in Ref2VA FL2VA; do
        mkdir -p "$MODEL_PATH/$part"
        for c in model_index.json audio_vae text_encoder tokenizer transformer video_vae; do
            ln -sfn "$SRC_MODEL/$part/$c" "$MODEL_PATH/$part/$c"
        done
        rm -rf "$MODEL_PATH/$part/processor"
        cp -r "$HELMA_DIR/model_patch/$part/processor" "$MODEL_PATH/$part/processor"
    done
fi
