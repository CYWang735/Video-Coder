#!/bin/bash
# Create the Python environment (conda prefix at $ENV_DIR): Python 3.12, ffmpeg, SGLang Diffusion.
# Needs conda (or mamba) on PATH and internet access to conda-forge and PyPI. Takes ~10 min.
# Usage: bash setup_env.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"

CONDA=$(command -v mamba || command -v conda || true)
[ -n "$CONDA" ] || { echo "conda/mamba not found on PATH" >&2; exit 1; }

if [ ! -x "$ENV_DIR/bin/python" ]; then
    "$CONDA" create -y -p "$ENV_DIR" -c conda-forge python=3.12 pip
fi
# The H3 pipeline calls the ffmpeg/ffprobe executables; take them from conda-forge.
[ -x "$ENV_DIR/bin/ffprobe" ] || "$CONDA" install -y -p "$ENV_DIR" -c conda-forge ffmpeg

PY="$ENV_DIR/bin/python"
# pip's resolver leaves huggingface_hub out of sglang[diffusion]; transformers 5.12 needs >=1.5,<2.
"$PY" -m pip install --pre "sglang[diffusion]==$SGLANG_VERSION" "huggingface_hub>=1.5,<2"
"$PY" -m pip check

# serve.sh falls back to the CUDA 13 toolkit pip installed (nvidia/cu13) when the machine has no nvcc.
# Its layout lacks what SGLang's JIT linker expects: a lib64/ directory and an unversioned libcudart.so.
CU13="$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")/nvidia/cu13"
if [ -d "$CU13/lib" ]; then
    [ -e "$CU13/lib64" ] || ln -s lib "$CU13/lib64"
    [ -e "$CU13/lib/libcudart.so" ] || ln -s libcudart.so.13 "$CU13/lib/libcudart.so"
fi

# conda-forge's Python lacks os.memfd_create, which the reference-video decoder needs;
# a .pth file loads the ctypes replacement in every interpreter of the env.
SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cp "$PROJECT_DIR/tools/memfd_compat.py" "$SITE/memfd_compat.py"
echo "import memfd_compat" > "$SITE/memfd_compat.pth"

"$PY" - <<'EOF'
import os, shutil, sglang, torch
os.close(os.memfd_create("probe", flags=os.MFD_CLOEXEC))
assert shutil.which("ffprobe", path=os.path.dirname(os.sys.executable)), "ffprobe missing"
print(f"sglang {sglang.__version__}, torch {torch.__version__}, "
      f"CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}")
EOF
echo "environment ready: $ENV_DIR"
