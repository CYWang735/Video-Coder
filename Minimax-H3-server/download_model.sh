#!/bin/bash
# Download the MiniMax-H3 Ref2VA checkpoint (~144 GB) from Hugging Face into $MODEL_PATH.
# Interrupted downloads resume when the script is run again.
# Usage: bash download_model.sh
#   Behind a slow link to huggingface.co:  HF_ENDPOINT=https://hf-mirror.com bash download_model.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"

[ "$(basename "$MODEL_PATH")" = MiniMax-H3 ] || {
    echo "MODEL_PATH must end in /MiniMax-H3 (SGLang identifies the model by that name): $MODEL_PATH" >&2
    exit 1
}
mkdir -p "$MODEL_PATH"
"$ENV_DIR/bin/hf" download MiniMaxAI/MiniMax-H3 \
    --include "model_index.json" --include "Ref2VA/*" \
    --local-dir "$MODEL_PATH"
du -sh "$MODEL_PATH"
