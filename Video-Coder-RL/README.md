# Video-Coder-RL

Agentic RL for the Video-Coder SFT model. The model writes Blender (`bpy`) scripts in a sandbox, runs them and submits a scene. The code is in [`uni-agent/`](uni-agent), which bundles a patched copy of verl.

## Environment

```bash
cd Video-Coder-RL/uni-agent
conda create -n video-coder-rl python=3.12 -y && conda activate video-coder-rl
pip install vllm==0.30.0 torch==2.13.0 transformers==5.17.0 ray[default]==2.58.0 \
    tensordict==0.14.2 accelerate==1.15.0 peft==0.21.1 qwen-vl-utils==0.0.14
grep -vE '^(transformers|tensordict|TransferQueue)' verl/requirements.txt > /tmp/verl-req.txt
pip install -r /tmp/verl-req.txt
pip install --no-deps -e ./verl
pip install --no-deps transferqueue==0.1.11 fla-core==0.5.2 flash-linear-attention==0.5.2
pip install litellm httpx pandas pyarrow pyyaml jinja2 imageio-ffmpeg
```

flash-linear-attention is required for long Qwen3.5 rollouts. Without it, the model falls back to a slow torch implementation that runs out of memory at about 100k tokens.

The sandbox needs two things:
- an Apptainer image with Blender 4.5 / `bpy` (`bpy-agent.sif`);
- a [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) ≥ 2.4 tool dir. Prepare it once:

```bash
cd Video-Coder-RL/uni-agent
bash examples/video_coder/install_harness.sh /path/to/mini-swe-agent-tool-dir
```

## RL Training

First, convert the RL tasks into parquet. Each record needs a task prompt and a ground-truth video (`gt_video`). The H3 request that produced each GT video is read from `<gt>.json`.

```bash
cd Video-Coder-RL/uni-agent
python -m uni_agent.tasks.video_coder.preprocess --tasks train.jsonl \
    --out data/video_coder_train.parquet --require-gt --gt-request-sidecar
python -m uni_agent.tasks.video_coder.preprocess --tasks val.jsonl \
    --out data/video_coder_val.parquet --require-gt --gt-request-sidecar
```

Next, launch the video generator for the similarity reward. Here we use [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (Ref2VA), served with SGLang. Each server uses 4 GPUs, and more servers make each step faster. We used 4 servers. See [`Minimax-H3-server`](../Minimax-H3-server) for details.

```bash
cd Minimax-H3-server
bash setup_env.sh && bash download_model.sh     # once
HOST=0.0.0.0 PORT=30011 bash serve.sh
```

The rendered video of an accepted scene is given to H3 as the reference. A VLM judge, `qwen3.8-omni-flash` on DashScope, then scores the H3 output against the GT video. The reward is

$$R = e^{-t/T} \cdot 1.0 \cdot \mathbb{1}[\text{accepted}] + 0.5 \cdot R_{\text{video}}$$

- $R = 0$ when the scene is not accepted.
- $T$ is the total number of training steps.

To use another generator or judge, revise the `similarity` section of the task config and [`uni_agent/tasks/video_coder/similarity.py`](uni-agent/uni_agent/tasks/video_coder/similarity.py).

Then edit the task config, [`examples/quickstart/training/task_config_video_coder_similarity_apptainer.yaml`](uni-agent/examples/quickstart/training/task_config_video_coder_similarity_apptainer.yaml):
- Set the launched H3 servers in `similarity.ref2va.base_urls`.
- Set `similarity.ref2va.shared_dir` to a directory that both the trainer and the H3 servers see at the same path.
- Set the sandbox `image` and `binds`.
- Enter your `DASHSCOPE_API_KEY` as an environment variable, not in the file:

```bash
export DASHSCOPE_API_KEY=...
```

Finally, start a Ray cluster over the training nodes and launch RL training from the SFT model ([`Video-Coder-SFT`](../Video-Coder-SFT)):

```bash
cd Video-Coder-RL/uni-agent
ray start --head            # on the other nodes: ray start --address=<head>:6379
MODEL_PATH=/path/to/qwen3.5-9B-Video-Coder-sft \
TRAIN_FILE=data/video_coder_train.parquet TEST_FILE=data/video_coder_val.parquet \
CKPTS_DIR=/path/to/ckpts NNODES=2 NGPUS_PER_NODE=4 \
TRAIN_PROMPT_BSZ=8 N_RESP_PER_PROMPT=8 PPO_MINI_BATCH_SIZE=8 NUM_WARMUP_BATCHES=0 \
TOTAL_EPOCHS=3 MAX_RESPONSE_LENGTH=114688 SAVE_FREQ=5 \
bash examples/video_coder/train_video_coder_rl.sh
```

We trained on 8 × 80GB GPUs (2 nodes), plus 4 GPUs per H3 server.
