#!/usr/bin/env bash
# Build the mini-swe-agent tool directory WITHOUT Docker, for apptainer sandboxes.
#
# Same content as Dockerfile.mini-swe-agent-tool (python-build-standalone + pinned
# mini-swe-agent + litellm + run_agent.py), but as a plain directory that is
# bind-mounted read-only at /opt/mini-swe-agent (task config: sandbox_kwargs.binds).
# Packed as ONE tarball so it costs a single file on quota-limited shared storage;
# each compute node unpacks it to node-local disk.
#
#   bash build_tool_apptainer.sh build   [TARBALL]        # needs internet (login node)
#   bash build_tool_apptainer.sh extract [TARBALL] [DIR]  # once per compute node
#
# Defaults: TARBALL=$HOME/wangcy/mini-swe-agent-tool.tar.gz  DIR=/tmp/$USER/mini-swe-agent-tool
set -euo pipefail

# The tool python always runs isolated (-I): no ~/.local user site, no PYTHONPATH /
# PYTHONHOME. Otherwise pip treats packages visible there as "already satisfied" and
# skips them, and the smoke test imports them from there too -- the tool dir looks fine
# on the build host but is missing those packages inside the sandbox.

PBS_RELEASE="${PBS_RELEASE:-20260602}"
PBS_PYTHON="${PBS_PYTHON:-3.12.13}"
MSA_VERSION="${MSA_VERSION:-2.4.6}"
LITELLM_VERSION="${LITELLM_VERSION:-1.103.1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mode="${1:-}"
tarball="${2:-$HOME/wangcy/mini-swe-agent-tool.tar.gz}"

case "${mode}" in
build)
    work="$(mktemp -d "${TMPDIR:-/tmp}/mini-swe-agent-build.XXXXXX")"
    trap 'rm -rf -- "${work:?}"' EXIT
    tool="${work}/tool"
    mkdir -p "${tool}" "$(dirname "${tarball}")"

    url="https://github.com/astral-sh/python-build-standalone/releases/download/${PBS_RELEASE}/cpython-${PBS_PYTHON}%2B${PBS_RELEASE}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
    echo "==> python-build-standalone ${PBS_PYTHON} (${PBS_RELEASE})"
    curl -fsSL "${url}" -o "${work}/python.tar.gz"
    tar -xzf "${work}/python.tar.gz" -C "${tool}" --strip-components=1

    echo "==> pip install mini-swe-agent==${MSA_VERSION} litellm==${LITELLM_VERSION}"
    "${tool}/bin/python" -I -m pip install --no-cache-dir --disable-pip-version-check -q \
        ${PIP_INDEX_URL:+-i "${PIP_INDEX_URL}"} \
        "mini-swe-agent==${MSA_VERSION}" "litellm==${LITELLM_VERSION}"

    cp "${SCRIPT_DIR}/run_agent.py" "${tool}/bin/run_agent.py"

    echo "==> dependency check + the imports run_agent.py performs (isolated)"
    "${tool}/bin/python" -I -m pip check
    "${tool}/bin/python" -I -c "
import sys
assert all(p.startswith('${tool}') for p in sys.path if p), sys.path
from minisweagent.config import builtin_config_dir, get_config_from_spec
get_config_from_spec(str(builtin_config_dir / 'benchmarks' / 'swebench.yaml'))
from minisweagent.environments.local import LocalEnvironment
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.agents.default import DefaultAgent
print('run_agent.py imports ok')"

    echo "==> packing ${tarball}"
    tar -C "${tool}" -czf "${tarball}.tmp" .
    mv "${tarball}.tmp" "${tarball}"
    ls -la "${tarball}"
    ;;
extract)
    dest="${3:-/tmp/${USER}/mini-swe-agent-tool}"
    [[ -f "${tarball}" ]] || { echo "tarball not found: ${tarball} (run 'build' first)" >&2; exit 1; }
    # Re-extract whenever the tarball changed (a rebuilt tool must not be shadowed by a
    # stale unpack). Do not run this while sandboxes are using ${dest}.
    stamp="$(stat -c '%s %Y' "${tarball}")"
    if [[ -f "${dest}/.tarball-stamp" && "$(cat "${dest}/.tarball-stamp")" == "${stamp}" ]]; then
        echo "up to date: ${dest}"
    else
        staging="${dest}.staging.$$"
        mkdir -p "${staging}"
        tar -xzf "${tarball}" -C "${staging}"
        echo "${stamp}" > "${staging}/.tarball-stamp"
        rm -rf -- "${dest:?}"
        mv "${staging}" "${dest}"
        echo "extracted to ${dest}"
    fi
    echo "bind it in the task config:  ${dest}:/opt/mini-swe-agent:ro"
    ;;
*)
    sed -n '2,15p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
