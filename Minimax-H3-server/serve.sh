#!/bin/bash
# Start the MiniMax-H3 Ref2VA server (SGLang Diffusion) in the foreground. Stop it with Ctrl-C.
# Loading the weights takes 3-7 minutes; the server is ready when the log prints
# "Uvicorn running on http://127.0.0.1:30011".
# Usage: bash serve.sh            (settings come from config.sh)
set -euo pipefail
source "$(dirname "$0")/config.sh"
export PATH="$ENV_DIR/bin:$PATH"

[ -d "$MODEL_PATH/Ref2VA/transformer" ] || {
    echo "model not found at $MODEL_PATH; run download_model.sh first" >&2
    exit 1
}

# SGLang JIT-compiles some CUDA kernels at startup and needs nvcc: use $CUDA_HOME if set, else the
# nvcc on PATH, else the CUDA 13 toolkit that pip installed into the env.
if [ -z "${CUDA_HOME:-}" ]; then
    if command -v nvcc >/dev/null; then
        CUDA_HOME=$(dirname "$(dirname "$(command -v nvcc)")")
    else
        CUDA_HOME="$("$ENV_DIR/bin/python" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cu13"
    fi
fi
export CUDA_HOME PATH="$CUDA_HOME/bin:$PATH"

export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/inductor"
export TVM_FFI_CACHE_DIR="$CACHE_ROOT/tvm-ffi"
export HF_HUB_OFFLINE=1   # the model is local; never try to download at startup
mkdir -p "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$SERVER_WORKDIR"

if [ "$TOPOLOGY" = auto ]; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    case "$gpu_name" in
        *H200*|*B200*|*B300*) TOPOLOGY=ulysses4 ;;
        *) TOPOLOGY=tp2u2 ;;   # 80-94 GB cards cannot keep the pipeline resident with pure Ulysses4
    esac
    echo "GPU: $gpu_name -> topology $TOPOLOGY"
fi
case "$TOPOLOGY" in
    ulysses4) PAR_ARGS=(--ulysses-degree "$NUM_GPUS") ;;
    tp2u2)    PAR_ARGS=(--tp-size 2 --ulysses-degree $((NUM_GPUS / 2))) ;;
    tp4)      PAR_ARGS=(--tp-size 4 --ulysses-degree $((NUM_GPUS / 4))) ;;
    *) echo "unknown TOPOLOGY=$TOPOLOGY" >&2; exit 1 ;;
esac

# The server writes its own copy of each video and of any uploaded input under its working directory.
cd "$SERVER_WORKDIR"
set -x
exec sglang serve \
    --model-path "$MODEL_PATH" \
    --model-variant "$MODEL_VARIANT" \
    --num-gpus "$NUM_GPUS" \
    "${PAR_ARGS[@]}" \
    --encoder-parallel auto \
    --performance-mode speed \
    --host "$HOST" \
    --port "$PORT" \
    ${EXTRA_SERVE_ARGS:-}
