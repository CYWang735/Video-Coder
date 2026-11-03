#!/bin/bash
# Unpack a tar.zst of a conda env to a node-local path and point its scripts' shebangs at the copy.
#
#   bash unpack_env.sh [ARCHIVE] [DEST] [ORIGINAL_PREFIX]
#
# Defaults: ARCHIVE=~/wangcy/uni-agent-v3-env.tar.zst, DEST=/scratch/$USER/envs/uni-agent-v3,
# ORIGINAL_PREFIX=~/.conda/envs/uni-agent. The archives build on each other, each packed with the same
# layout (`tar -C ~/.conda/envs -I zstd -cf ... uni-agent`: one top-level dir, shebangs on ORIGINAL_PREFIX):
#   uni-agent-env.tar.zst     the original ~/.conda/envs/uni-agent
#   uni-agent-v2-env.tar.zst  + TransferQueue 0.1.11 (installed --no-deps, as this verl requires) and the
#                             editable verl pointed at ~/wangcy/uni-agent-v2/verl
#   uni-agent-v3-env.tar.zst  + fla-core / flash-linear-attention 0.5.2 (--no-deps): fast kernels for Qwen3.5's
#                             GatedDeltaNet layers; without them transformers falls back to a torch reference
#                             that is ~100x slower and runs out of memory at ~100k tokens (needs FLA_TILELANG=0,
#                             set in runtime_env.yaml)
# Only DEST is written; the archives and the original env are never touched. A stamp naming the source
# archive makes it a no-op on a node that already has this copy, and an error when DEST holds a copy of
# another archive (it would run the wrong env).
#
# Python itself is relocatable (it finds its libraries from its own location), but console scripts such as
# `ray`, `vllm` and `pip` start with `#!<ORIGINAL_PREFIX>/bin/python3.12`: without the rewrite they would run
# the original env, and so would everything Ray starts.
set -euo pipefail
ARCHIVE=${1:-$HOME/wangcy/uni-agent-v3-env.tar.zst}
DEST=${2:-/scratch/$USER/envs/uni-agent-v3}
OLD=${3:-$HOME/.conda/envs/uni-agent}

# one unpack at a time per destination (e.g. two jobs on the same node); the later one finds the stamp
parent=$(dirname "$DEST")
mkdir -p "$parent"
exec 9>"$parent/.$(basename "$DEST").lock"
flock 9
[[ -f $ARCHIVE ]] || { echo "archive not found: $ARCHIVE" >&2; exit 1; }
source_id="$(realpath "$ARCHIVE") $(stat -c '%s %Y' "$ARCHIVE")"
if [[ -f $DEST/.unpacked ]]; then
    have=$(cat "$DEST/.unpacked")
    [[ $have == "$source_id" ]] || { echo "$(hostname -s): $DEST holds a copy of ${have:-an unknown archive}," \
        "not $source_id; use another DEST or remove it" >&2; exit 1; }
else
    stage="$parent/.unpack.$$"
    mkdir -p "$stage"
    t0=$(date +%s)
    tar -I zstd -xf "$ARCHIVE" -C "$stage"
    top=$(find "$stage" -mindepth 1 -maxdepth 1 -type d | head -1)   # the archive's single top-level dir
    [[ -x $top/bin/python ]] || { echo "archive has no <env>/bin/python: $ARCHIVE" >&2; exit 1; }
    rm -rf "${DEST:?}"
    mv "$top" "$DEST"
    rmdir "$stage"
    echo "$source_id" > "$DEST/.unpacked"
    echo "$(hostname -s): unpacked $ARCHIVE to $DEST in $(( $(date +%s) - t0 ))s"
fi

# regular files only (bin/ also holds directories)
with_old_shebang() { find "$DEST/bin" -maxdepth 1 -type f -exec grep -lIs "^#!$OLD/" {} + || true; }
n=$(with_old_shebang | wc -l)
with_old_shebang | xargs -r sed -i "1s|^#!$OLD/|#!$DEST/|"
left=$(with_old_shebang | wc -l)
echo "$(hostname -s): $DEST ready ($n shebangs rewritten, $left left)"
(( left == 0 ))
