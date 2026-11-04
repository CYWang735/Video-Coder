#!/usr/bin/env bash
# RL for the video_coder (Blender) task: Video-Coder harness in bpy-agent.sif, reward = 1.0 * accepted
# + 0.5 * video reward (MiniMax-H3 Ref2VA + VLM judge against the sample's GT video).
#
# Run from the repo root on the GPU node, with a Ray head already running there:
#   cd ~/wangcy/uni-agent-v2 && source proxy.sh
#   ray status || { unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES; NUM_GPUS=2 restart_ray; }
#   MODEL_PATH=/path/to/policy bash examples/video_coder/train_video_coder_rl.sh
#
# Before the first run:
#   - env: build it as in the repository README (Training / Setup) and start Ray from it (it needs flash-linear-attention,
#     which Qwen3.5 needs for long rollouts)
#   - data: python -m uni_agent.tasks.video_coder.preprocess --tasks train.jsonl --out <train.parquet> --require-gt
#   - fill in the H3 / VLM endpoints, the VLM model name and the Ref2VA request (= the GT recipe) in
#     $TASK_CONFIG (default: task_config_video_coder_video_reward_apptainer.yaml)
#   - bash examples/video_coder/install_harness.sh  (once: imageio-ffmpeg in the tool dir)
# Set TASK_CONFIG=examples/quickstart/training/task_config_video_coder_apptainer.yaml to train on the
# acceptance reward only (no H3 / VLM needed).
set -euo pipefail

[[ -f examples/quickstart/training/train_qwen3p5_fsdp.sh ]] || { echo "run from the uni-agent repo root" >&2; exit 1; }

export MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the policy checkpoint (HF format)}
export TRAIN_FILE=${TRAIN_FILE:-$HOME/data/uni_agent/video_coder_train.parquet}
export TEST_FILE=${TEST_FILE:-$HOME/data/uni_agent/video_coder_val.parquet}
export TASK_CONFIG=${TASK_CONFIG:-examples/quickstart/training/task_config_video_coder_video_reward_apptainer.yaml}
export RUNTIME_ENV=${RUNTIME_ENV:-runtime_env.yaml}

export PROJECT_NAME=${PROJECT_NAME:-Uni-Agent-VideoCoder}
export EXP_NAME=${EXP_NAME:-video_coder_$(date +%Y%m%d%H%M)}
# Checkpoints are large (weights + optimizer state): put them on storage with room. The default is
# node-local and disappears with the Slurm job.
export CKPTS_DIR=${CKPTS_DIR:-/tmp/$USER/ckpts/$PROJECT_NAME/$EXP_NAME}
export AGENT_LOG_DIR=${AGENT_LOG_DIR:-$PWD/logs/$EXP_NAME}
export SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-$(basename "$MODEL_PATH")}

# Resources (one node). Each vLLM replica holds the full model at GEN_TP=1.
export NNODES=${NNODES:-1}
export NGPUS_PER_NODE=${NGPUS_PER_NODE:-2}
export GEN_TP=${GEN_TP:-1}
export GATEWAY_COUNT=${GATEWAY_COUNT:-2}
export CONCURRENCY=${CONCURRENCY:-32}

# Batch: GRPO needs several rollouts of the same scene with different outcomes.
export TRAIN_PROMPT_BSZ=${TRAIN_PROMPT_BSZ:-8}
export N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-4}
export PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
export NUM_WARMUP_BATCHES=${NUM_WARMUP_BATCHES:-1}
export MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-57344}
# Keep the actor's params/grads/optimizer on the CPU during the update (FSDP2 CPUOffloadPolicy): with them on
# the GPU, a ~100k-token sequence of the 9B policy runs out of memory in the backward pass.
export OFFLOAD_POLICY=${OFFLOAD_POLICY:-True}

# Algorithm (as in the earlier Qwen3.5 runs).
export ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
export LOSS_MODE=${LOSS_MODE:-gspo}
export CLIP_RATIO_LOW=${CLIP_RATIO_LOW:-4e-4}
export CLIP_RATIO_HIGH=${CLIP_RATIO_HIGH:-4e-4}
export CLIP_RATIO_C=${CLIP_RATIO_C:-10}
export LOSS_AGG_MODE=${LOSS_AGG_MODE:-token-mean}
export BYPASS_MODE=${BYPASS_MODE:-False}
export ROLLOUT_IS=${ROLLOUT_IS:-token}
export ROLLOUT_IS_THRESHOLD=${ROLLOUT_IS_THRESHOLD:-2.0}
export ROLLOUT_IS_BATCH_NORMALIZE=${ROLLOUT_IS_BATCH_NORMALIZE:-False}
export ROLLOUT_RS=${ROLLOUT_RS:-null}
# Episodes the agent did not submit (step / time / context limit, repeated format errors) keep reward 0 in their
# GRPO group, so they still lower the baseline, but their tokens are masked out of the loss.
export MASK_UNFINISHED_EPISODE=${MASK_UNFINISHED_EPISODE:-True}
export TRAJECTORY_SELECTION=${TRAJECTORY_SELECTION:-all}
# Validation runs the same (slow) H3 + VLM scoring; off by default.
export TEST_FREQ=${TEST_FREQ:--1}
export SAVE_FREQ=${SAVE_FREQ:-10}
export TRAINER_LOGGER=${TRAINER_LOGGER:-"['console','wandb']"}

# ---- preflight: fail here instead of minutes into the job ----------------------------------------
fail() { echo "preflight: $*" >&2; exit 1; }
[[ -f "$MODEL_PATH/config.json" ]] || fail "no config.json in MODEL_PATH=$MODEL_PATH"
[[ -f "$TRAIN_FILE" ]] || fail "TRAIN_FILE not found: $TRAIN_FILE"
[[ -f "$TEST_FILE" ]] || fail "TEST_FILE not found: $TEST_FILE"
[[ -f "$TASK_CONFIG" ]] || fail "TASK_CONFIG not found: $TASK_CONFIG"
ray status >/dev/null 2>&1 || fail "no Ray cluster here; start one (see the header)"
[[ -z "${ROCR_VISIBLE_DEVICES:-}" ]] || echo "preflight: warning: ROCR_VISIBLE_DEVICES is set in this shell;" \
    "if Ray was started with it, workers fail ('Please don't set ROCR_VISIBLE_DEVICES ...')" >&2

"${PYTHON:-python}" - "$TASK_CONFIG" "$TRAIN_FILE" <<'EOF' || fail "task config / data check failed (see above)"
import os, sys, urllib.request
import yaml
cfg_path, train_file = sys.argv[1:]
tasks = {t["name"]: t for t in yaml.safe_load(open(cfg_path))}
task = tasks.get("video_coder") or sys.exit(f"{cfg_path} has no video_coder task")
image = task["sandbox"].get("image")
if image and image.endswith(".sif") and not os.path.isfile(image):
    sys.exit(f"sandbox image not found: {image}")
sim = task.get("video_reward") or task.get("similarity")  # similarity: old name
if sim and sim.get("enabled", True):
    import pandas as pd
    rows = pd.read_parquet(train_file)["extra_info"]
    gt_field = sim.get("gt_field", "gt_video")
    missing = [r["tools_kwargs"]["task"]["metadata"].get("id") for r in rows
               if not os.path.isfile(r["tools_kwargs"]["task"]["metadata"].get(gt_field) or "")]
    if missing:
        sys.exit(f"{len(missing)} training rows have no GT video ({gt_field}), e.g. {missing[:3]}")
    if not os.path.isdir(sim["ref2va"]["shared_dir"]):
        os.makedirs(sim["ref2va"]["shared_dir"], exist_ok=True)
    ref2va, judge = sim["ref2va"], sim["judge"]
    h3_urls = ref2va.get("base_urls") or [ref2va["base_url"]]
    direct = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # in-cluster: never proxied
    proxy = judge.get("proxy")
    via_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({"http": proxy, "https": proxy})) if proxy else direct
    checks = [("MiniMax-H3", u.rstrip("/") + "/health", direct) for u in h3_urls]
    checks.append(("VLM judge", judge["base_url"].rstrip("/") + "/models", via_proxy))
    for name, url, opener in checks:
        try:
            opener.open(url, timeout=15)
        except urllib.error.HTTPError as e:
            # an HTTP answer means reachable (the route may not exist or need the key) -- unless it is
            # the proxy's own error page
            if "squid" in (e.headers.get("Server") or "").lower() or e.headers.get("X-Squid-Error"):
                sys.exit(f"{name}: the proxy refused {url}: HTTP {e.code} {e.headers.get('X-Squid-Error', '')}")
        except Exception as e:
            sys.exit(f"{name} not reachable at {url}: {e}")
    if judge.get("api_key_env") and not os.environ.get(judge["api_key_env"]):
        print(f"preflight: note: {judge['api_key_env']} is not set in this shell; it must be set where Ray was started")
    print(f"preflight: {len(rows)} rows with GT videos; {len(h3_urls)} H3 server(s) and the VLM endpoint reachable")
else:
    print("preflight: video reward off (acceptance reward only)")
EOF

echo "starting: model=$MODEL_PATH train=$TRAIN_FILE task_config=$TASK_CONFIG logs=$AGENT_LOG_DIR ckpts=$CKPTS_DIR"
exec bash examples/quickstart/training/train_qwen3p5_fsdp.sh
