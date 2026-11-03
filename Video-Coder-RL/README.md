# Video-Coder-RL

Agentic RL for Video-Coder. The policy is a Qwen3.5-9B model fine-tuned with Video-Coder SFT. In a sandbox it writes Blender (`bpy`) scripts, runs them, looks at preview frames, and submits a scene. It is trained with GRPO on a reward that combines execution with video similarity.

Training uses [`uni-agent/`](uni-agent), a snapshot of uni-agent (branch `feat/apptainer-video-coder`). Its [`verl/`](uni-agent/verl) directory holds verl as plain files at commit `c269afa8`, which adds a fix for Qwen3.5's fused output head. Run every command below from `uni-agent/`.

## Reward

For a rollout at training step $t$:

$$
R = \begin{cases}
0 & \text{if the scene is not accepted} \\
e^{-t/T}\,\alpha_{\text{exe}} + \alpha_{\text{sim}}\, R_{\text{video}} & \text{otherwise}
\end{cases}
$$

- **Accepted** means the agent submitted a script and it ran in Blender 4.5 without errors.
- $R_{\text{video}} \in [0, 1]$ is computed in three stages:
  1. The accepted render is used as the reference video for **MiniMax-H3** (Ref2VA).
  2. H3 generates a video with the same recipe that produced the sample's ground-truth video.
  3. A VLM judge, `qwen3.8-omni-flash` on DashScope, compares the H3 video with the ground-truth video. It scores faithfulness, layout, motion and camera, and the four sub-scores are weighted into one value.
- Defaults:
  - $\alpha_{\text{exe}} = 1.0$ (`accept_weight`)
  - $\alpha_{\text{sim}} = 0.5$ (`similarity_weight`)
  - $T$ = the total number of training steps (`exe_anneal_steps: auto`)
- Effect of the annealing: the execution term starts at 1 and decays to $e^{-1} \approx 0.37$ by the end of training, which shifts the weight toward similarity. Set `exe_anneal_steps: null` to turn the annealing off. Validation rewards are never annealed.

The reward code is in [`uni_agent/tasks/video_coder/`](uni-agent/uni_agent/tasks/video_coder): `task.py` and `similarity.py`.

## Layout

| Path (under `uni-agent/`) | What it is |
|---|---|
| `examples/video_coder/train_video_coder_rl.sh` | Launcher: checks the model, data, sandbox, H3 and judge, then starts training |
| `examples/quickstart/training/train_qwen3p5_fsdp.sh` | The verl trainer command that the launcher runs |
| `examples/quickstart/training/task_config_video_coder_similarity_apptainer.yaml` | Task config: reward, H3, judge, sandbox and agent limits |
| `examples/video_coder/{run_agent.py,vcsft/,bin/,configs/}` | Agent harness. It is bound read-only into the sandbox at `/opt/video-coder` |
| `examples/video_coder/install_harness.sh` | Adds `imageio-ffmpeg` to the mini-swe-agent tool dir and checks the harness imports |
| `uni_agent/tasks/video_coder/preprocess.py` | Converts task files (.jsonl/.json) into training parquet |
| `runtime_env.yaml` | Ray runtime env, including `FLA_TILELANG=0` and the cache paths |

## Requirements

- **GPUs:** H100 80 GB. The run in this README used 2 nodes × 4 GPUs.
- **Python env:** Python 3.12 with vLLM, the bundled verl, TransferQueue and flash-linear-attention. See [Setup](#setup).
- **Sandbox:**
  - An Apptainer image with Blender 4.5 / bpy (`bpy-agent.sif`).
  - A mini-swe-agent ≥ 2.4 tool dir, bound at `/opt/mini-swe-agent`.
- **Policy:** a Qwen3.5-9B Video-Coder SFT checkpoint in HF format. See [`../Video-Coder-SFT`](../Video-Coder-SFT).
- **MiniMax-H3 Ref2VA servers:** SGLang `/v1/videos` endpoints, deployed with [`../Minimax-H3-server`](../Minimax-H3-server) (one server per 4 GPUs). They must read `ref2va.shared_dir` at the same path as the trainer.
- **Judge key:** a DashScope API key in `DASHSCOPE_API_KEY`, exported in the shell that starts Ray. Never put it in a config file.

## Setup

1. Build the training env. The versions below are the ones we trained with.

   ```bash
   conda create -n video-coder-rl python=3.12 -y
   conda activate video-coder-rl

   # inference + training stack (vLLM pulls a matching torch)
   pip install vllm==0.30.0 torch==2.13.0 transformers==5.17.0 ray[default]==2.58.0 \
       tensordict==0.14.2 accelerate==1.15.0 peft==0.21.1 qwen-vl-utils==0.0.14
   # verl's other requirements. Skip its transformers / tensordict pins: Qwen3.5 needs the newer
   # versions above. TransferQueue is installed separately below.
   grep -vE '^(transformers|tensordict|TransferQueue)' verl/requirements.txt > /tmp/verl-req.txt
   pip install -r /tmp/verl-req.txt
   # the bundled verl (with the Qwen3.5 fused-output-head fix)
   pip install --no-deps -e ./verl
   # verl v1 trainer data plane; --no-deps keeps the versions above
   pip install --no-deps transferqueue==0.1.11
   # fast GatedDeltaNet kernels for Qwen3.5. Without them transformers falls back to a torch
   # reference that is ~100x slower and runs out of memory at ~100k tokens.
   pip install --no-deps fla-core==0.5.2 flash-linear-attention==0.5.2
   # reward / judge clients and data tools
   pip install litellm httpx pandas pyarrow pyyaml jinja2 imageio-ffmpeg
   ```

   `runtime_env.yaml` sets `FLA_TILELANG=0`. This makes flash-linear-attention use its Triton kernels, which serve any sequence length. Without it, TileLang (if installed) compiles a new kernel for every sequence length.

2. Prepare the sandbox harness (once per tool dir; needs internet once):

   ```bash
   bash examples/video_coder/install_harness.sh /path/to/mini-swe-agent-tool-dir
   ```

3. Build the training data. Each record needs a task prompt and a ground-truth video, `gt_video`. `--gt-request-sidecar` reads each GT video's H3 request from `<gt>.json`. The run below used 1,660 training scenes.

   ```bash
   python -m uni_agent.tasks.video_coder.preprocess --tasks train.jsonl \
       --out ~/data/uni_agent/video_coder_train.parquet --require-gt --gt-request-sidecar
   python -m uni_agent.tasks.video_coder.preprocess --tasks val.jsonl \
       --out ~/data/uni_agent/video_coder_val.parquet --require-gt --gt-request-sidecar
   ```

4. Edit the task config, `task_config_video_coder_similarity_apptainer.yaml`:
   - `similarity.ref2va.base_urls`: the list of H3 servers. Requests go to the least-loaded server.
   - `similarity.ref2va.shared_dir`: a directory that both the trainer and the H3 servers can see.
   - `similarity.ref2va.request`: must match the recipe that produced the GT videos.
   - `similarity.judge.proxy`: set it if compute nodes reach DashScope through an HTTP proxy.
   - `similarity.judge.extra_body: {modalities: [text]}`: text-only output from the omni judge.
   - `sandbox.image` and `sandbox.sandbox_kwargs.binds`: your paths to `bpy-agent.sif`, the tool dir and `examples/video_coder`.
   - `agent.step_limit`: the run below used `100`.

## Training

1. Start a Ray cluster over all training nodes, using the env above. Export `DASHSCOPE_API_KEY` in the same shell before starting it.
2. On the head node, from `uni-agent/`, run:

```bash
export DASHSCOPE_API_KEY=...          # where Ray was started
MODEL_PATH=/path/to/qwen3.5-9B-Video-Coder-sft \
TRAIN_FILE=~/data/uni_agent/video_coder_train.parquet \
TEST_FILE=~/data/uni_agent/video_coder_val.parquet \
CKPTS_DIR=/path/with/space/ckpts/$EXP_NAME \
NNODES=2 NGPUS_PER_NODE=4 \
TRAIN_PROMPT_BSZ=8 N_RESP_PER_PROMPT=8 PPO_MINI_BATCH_SIZE=8 NUM_WARMUP_BATCHES=0 \
TOTAL_EPOCHS=3 MAX_RESPONSE_LENGTH=114688 SAVE_FREQ=5 \
bash examples/video_coder/train_video_coder_rl.sh
```

The launcher runs a preflight check before training starts. It fails at once if any of these is wrong:
- the model has no `config.json`;
- a data file is missing;
- a training row has no GT video;
- the sandbox image is missing;
- an H3 server or the judge is unreachable.

Then it runs `train_qwen3p5_fsdp.sh`.

Each checkpoint is about 59 GB (weights plus optimizer state). Point `CKPTS_DIR` at shared storage with enough space, because the default is node-local.

To train on the execution reward only, without H3 or the judge, set `TASK_CONFIG=examples/quickstart/training/task_config_video_coder_apptainer.yaml`.

## Settings

| | Value |
|---|---|
| Algorithm | GRPO advantages, GSPO loss (clip 4e-4 / 4e-4, clip-c 10, token-mean) |
| Off-policy correction | Token-level rollout importance sampling, threshold 2.0 |
| KL / entropy | Off / 0 |
| Optimizer | AdamW, lr 1e-6 (constant), weight decay 0.1 |
| Batch | 8 prompts × 8 rollouts per step, mini-batch 8 |
| Data | 1,660 scenes → 207 steps per epoch; 3 epochs = 621 steps |
| Context | Prompt 8k + response 112k tokens; agent step limit 100 |
| Unfinished episodes | Reward 0 in the group baseline, tokens masked from the loss |
| Trainer | verl v1 `colocate_async`, FSDP2 with CPU offload of params, grads and optimizer, full fine-tuning |
| Rollout | vLLM, TP 1, visual feedback (preview frames returned as images in tool results) |
| Reward | $\alpha_{\text{exe}}=1.0$, $\alpha_{\text{sim}}=0.5$, $T$ = 621 |
| Validation | Off by default (`TEST_FREQ=-1`). It runs the same slow H3 + judge scoring |

With 4 H3 servers, one step took about 1.2–1.9 hours. Most of that time is spent waiting for H3 video generation.

## Logs

- **Agent logs:** `AGENT_LOG_DIR` (default `~/wangcy/uni_agent_logs/$EXP_NAME`). Each rollout gets its own directory, `rollouts/step_N/session-sample-*/`, which holds:
  - `task.log`, whose last `run_task done` line gives the reward;
  - `framework.log`;
  - `trajectory.json` / `.npz`.
- **Metrics:** sent to wandb, project `Uni-Agent-VideoCoder`.
- **Similarity files:** with `similarity.keep_files: true`, the H3 videos and judge outputs are kept under `shared_dir`.
