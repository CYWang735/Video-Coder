#!/usr/bin/env bash
# Video-Coder inference: draft videos for your own prompts, ready for MiniMax-H3.
#
# Run from Video-Coder-RL/uni-agent with a Ray cluster up (RL environment):
#   MODEL_PATH=/path/to/Video-Coder-9B DATA=/path/to/data.json OUT=/path/to/output \
#   bash examples/video_coder/inference/run_inference.sh
#
# DATA has the training data's format: a JSON list of {"id": ..., "prompt": ...} (gt_video is not needed).
# Output: $OUT/drafts/<id>/render.mp4 (draft video), main.py (its Blender script), result.json, and
# $OUT/h3_manifest.jsonl for Minimax-H3-server/run_batch.sh. A prompt whose draft is not accepted is retried, up to
# PASSES attempts; rerunning the command resumes (prompts with a draft are skipped).
set -euo pipefail
cd "$(dirname "$0")/../../.."

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the Video-Coder checkpoint}
DATA=$(realpath "${DATA:?set DATA to your prompt file (JSON list of id/prompt)}")
OUT=$(realpath -m "${OUT:?set OUT to an output directory}")
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-$(nvidia-smi -L | wc -l)}
CONCURRENCY=${CONCURRENCY:-$(( 32 * NNODES ))}
GATEWAY_COUNT=${GATEWAY_COUNT:-$(( 2 * NNODES ))}
PASSES=${PASSES:-3}
HERE=examples/video_coder/inference

mkdir -p "$OUT/drafts" "$OUT/logs"
# Absolute paths: the parquet is read inside the Ray job, whose working dir is an uploaded copy of this repo.
python -m uni_agent.tasks.video_coder.preprocess --tasks "$DATA" --out "$OUT/data.parquet"
sed "s|__DRAFT_DIR__|$OUT/drafts|" "$HERE/task_config.yaml" > "$HERE/.task_config.run.yaml"

for pass in $(seq 1 "$PASSES"); do
    echo "[$(date +%T)] pass $pass"
    ray job submit --runtime-env runtime_env.yaml -- python3 "$HERE/run_inference.py" \
        --data-path "$OUT/data.parquet" --model-path "$MODEL_PATH" --served-model-name video-coder \
        --task-config "$HERE/.task_config.run.yaml" \
        --temperature 1.0 --top-p 0.95 --allowed-request-sampling-param-keys temperature top_p top_k \
        --tool-parser qwen3_coder --tensor-parallel-size 1 --nnodes "$NNODES" --n-gpus-per-node "$NGPUS_PER_NODE" \
        --gpu-memory-utilization 0.85 --response-length 114688 \
        --gateway-count "$GATEWAY_COUNT" --concurrency "$CONCURRENCY" --log-dir "$OUT/logs/pass$pass" \
        > "$OUT/logs/pass$pass.log" 2>&1 || echo "pass $pass: the Ray job failed, see $OUT/logs/pass$pass.log"
    left=$(python - "$DATA" "$OUT/drafts" <<'EOF'
import json, os, sys
data, drafts = sys.argv[1:]
with open(data) as f:
    rows = json.load(f) if data.endswith(".json") else [json.loads(x) for x in f if x.strip()]
print(sum(not os.path.isfile(os.path.join(drafts, str(r["id"]), "render.mp4")) for r in rows))
EOF
)
    echo "[$(date +%T)] pass $pass done: $left prompts without a draft"
    [[ $left == 0 ]] && break
done
python "$HERE/make_h3_manifest.py" --data "$DATA" --drafts "$OUT/drafts" --out "$OUT/h3_manifest.jsonl"
