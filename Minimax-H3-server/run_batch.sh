#!/bin/bash
# One command for a whole batch: start the server, wait until it is ready, generate every clip
# of the manifest, then stop the server.
# Usage: bash run_batch.sh <manifest.jsonl> <output_dir> [batch_generate.py options...]
#   e.g. bash run_batch.sh data/manifest.jsonl outputs/run1
#        bash run_batch.sh data/manifest.jsonl outputs/run1 --seed 7 --steps 50
# Logs: logs/server-<time>.log (server) and logs/batch-<time>.log (progress, also printed).
set -euo pipefail
cd "$(dirname "$0")"
source ./config.sh
[ $# -ge 2 ] || { sed -n 2,7p "$0"; exit 1; }
manifest="$1"; out="$2"; shift 2

stamp=$(date +%Y%m%d-%H%M%S)${SLURM_JOB_ID:+-$SLURM_JOB_ID}   # job id keeps parallel shards apart
mkdir -p logs
server_log="logs/server-$stamp.log"
batch_log="logs/batch-$stamp.log"

# Something already answering on the port would pass the readiness check below and receive the jobs.
if "$ENV_DIR/bin/python" -c "import socket, sys; s = socket.socket(); sys.exit(s.connect_ex(('127.0.0.1', $PORT)) != 0)"; then
    echo "port $PORT is already in use (another server still running?); stop it or set PORT=..." >&2
    exit 1
fi

echo "starting server (log: $server_log)"
bash serve.sh > "$server_log" 2>&1 &
server_pid=$!
trap 'kill $server_pid 2>/dev/null; wait $server_pid 2>/dev/null || true' EXIT

# Wait up to 60 min for the weights to load; stop early if the server process dies.
"$ENV_DIR/bin/python" - "$PORT" "$server_pid" <<'EOF'
import os, sys, time, urllib.request
port, pid = sys.argv[1], int(sys.argv[2])
t0 = time.time()
while time.time() - t0 < 3600:
    try:
        os.kill(pid, 0)
    except OSError:
        sys.exit("server exited during startup; see the server log")
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5).status == 200:
            print(f"server ready after {time.time() - t0:.0f}s", flush=True)
            sys.exit(0)
    except Exception:
        pass
    time.sleep(15)
sys.exit("server not ready after 3600s")
EOF

"$ENV_DIR/bin/python" batch_generate.py --manifest "$manifest" --out "$out" \
    --server "http://127.0.0.1:$PORT" "$@" 2>&1 | tee "$batch_log"
exit "${PIPESTATUS[0]}"
