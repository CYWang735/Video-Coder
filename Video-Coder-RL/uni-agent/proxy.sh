#!/usr/bin/env bash
# Usage (must be *sourced*, or the exports vanish when the script exits):
#   source proxy.sh          # proxy + NO_PROXY for this node, env only
#   restart_ray              # optional: stop Ray and start a fresh head with the env above
#
# Why NO_PROXY matters: with only http(s)_proxy set, requests to this node itself
# (the Ray dashboard that `ray job submit` talks to, gateway <-> vLLM, claude <->
# gateway) are sent to the cluster proxy, which answers 403 / "Access Denied".
# Ray workers inherit the environment of the shell that ran `ray start`, so the
# variables must be set *before* Ray starts and again in the shell that submits jobs.

if ! (return 0 2>/dev/null); then
    echo "proxy.sh must be sourced:  source ${0}" >&2
    exit 1
fi

# NHR@FAU compute nodes require this proxy for outbound HTTP/HTTPS traffic.
export http_proxy="http://proxy.nhr.fau.de:80"
export https_proxy="http://proxy.nhr.fau.de:80"
export HTTP_PROXY="$http_proxy"
export HTTPS_PROXY="$https_proxy"

# Everything on this node / cluster-internal must bypass the proxy. The node IP is
# different on every node, so it is detected here rather than hard-coded.
_node_ip="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -m1 '^10\.')"
export no_proxy="localhost,127.0.0.1,::1,${_node_ip:+${_node_ip},}$(hostname -s),$(hostname)"
export NO_PROXY="$no_proxy"
unset _node_ip

# Keep Python/Hugging Face caches on Helma's writable per-user cache path.
export PIP_CACHE_DIR="$HOME/.cache/pip"
export HF_HOME="$HOME/.cache/huggingface"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

mkdir -p "$PIP_CACHE_DIR" "$HF_DATASETS_CACHE"

# Ray/vLLM subprocesses must prefer Conda's newer libstdc++ over /lib64.
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

# Stop any running Ray (which holds a stale environment) and start a fresh single-node
# head from *this* shell. NUM_GPUS defaults to 4. Kills running Ray jobs: not run on source.
restart_ray() {
    ray stop
    ray start --head --num-gpus="${NUM_GPUS:-4}"
}

echo "NHR@FAU compute-node proxy enabled: $https_proxy"
echo "no_proxy: $no_proxy"
case ",$no_proxy," in
    *,10.*) ;;
    *) echo "WARNING: no node IP (10.x.x.x) found for $(hostname -s); is this a compute node?" >&2 ;;
esac
