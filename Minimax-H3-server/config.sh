# Settings shared by every script. Each value can also be set as an environment variable,
# which takes precedence, e.g.  NUM_GPUS=8 TOPOLOGY=ulysses4 bash run_batch.sh ...
# Sourced, not executed.

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Optional site file that sets cluster-specific overrides before the defaults below.
# Not needed on a normal GPU server.
if [ -n "${H3_SITE_ENV:-}" ]; then
    # shellcheck disable=SC1090
    source "$H3_SITE_ENV"
fi

# Python environment created by setup_env.sh (conda prefix, ~20 GB).
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/env}"

# Model directory created by download_model.sh (~144 GB for Ref2VA).
# SGLang recognises the model by the directory name, so it must end in /MiniMax-H3.
MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/models/MiniMax-H3}"

# GPUs and parallel layout for `sglang serve`.
#   auto     : ulysses4 on H200/B200-class cards, tp2u2 otherwise
#   ulysses4 : pure sequence parallel, fastest, needs ~140 GB cards
#   tp2u2    : TP2 + Ulysses2, fastest on 80-94 GB cards (H100, A100-80G)
#   tp4      : TP4, least memory per GPU, slightly slower
NUM_GPUS="${NUM_GPUS:-4}"
# Checkpoint partition: ref2va (reference-conditioned, white-model batches) or fl2va (t2va text-only + fl2va).
MODEL_VARIANT="${MODEL_VARIANT:-ref2va}"
TOPOLOGY="${TOPOLOGY:-auto}"

# Server address. The server only listens on HOST, so run the batch client on the same machine.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30011}"

# JIT kernel caches (Triton, TVM-FFI, torch inductor) and the server's working directory
# (it writes its own copy of every video and any uploaded inputs there; safe to delete).
CACHE_ROOT="${CACHE_ROOT:-$PROJECT_DIR/.cache}"
SERVER_WORKDIR="${SERVER_WORKDIR:-$PROJECT_DIR/.server}"

# SGLang Diffusion version this project was tested with.
SGLANG_VERSION="${SGLANG_VERSION:-0.5.21}"

export PROJECT_DIR ENV_DIR MODEL_PATH MODEL_VARIANT NUM_GPUS TOPOLOGY HOST PORT CACHE_ROOT SERVER_WORKDIR SGLANG_VERSION
# Packages in ~/.local would otherwise shadow (and hide missing) packages of the env.
export PYTHONNOUSERSITE=1
# The client talks to the server on 127.0.0.1; an http(s)_proxy in the environment (needed for pip on
# some clusters) would otherwise route those requests to the proxy, which answers 403.
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}" NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
