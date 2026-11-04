# Video-Coder: Reinforcing Agentic Coding for Video Generation

[Overview](#-about-video-coder) · [Datasets](#-datasets) · [Results](#-performance) · [Training](#-training) · [Video Generation](#-video-generation)

## 👀 About Video-Coder

<div align="center">
  <img src="./assets/teaser.png" alt="Video-Coder overview" width="100%">
</div>

We introduce **Video-Coder**, a code-driven video generation agent that autonomously writes and executes Blender Python programs to construct dynamic 3D scenes. The rendered **Draft Video** provides explicit structure and motion signals to downstream video generators.

We develop the **Video-Coder Harness**, a unified interactive environment integrating code execution, 3D scene rendering, and execution feedback. The same harness supports trajectory synthesis, agentic reinforcement learning, and inference.

Video-Coder is trained with supervised fine-tuning followed by **harness-native agentic reinforcement learning** with dual reward feedback. An execution reward encourages executable scene programs, while a video reward evaluates the generated results. An annealing strategy gradually reduces the execution reward during training, placing greater emphasis on video quality in later stages.

## 📍 Highlights

- **Executable 3D scene programs:** explicitly represent objects, spatial layout, motion, and camera behavior.
- **Iterative refinement:** write, execute, inspect, and revise programs through the Video-Coder Harness.
- **Two-stage training:** initialize with SFT, then improve generation through agentic RL with execution and video rewards.
- **Transfer across generators:** use the trained agent with MiniMax-H3 and OmniWeaving.
- **Physics-focused evaluation:** assess generation on PhyGenBench and VBench-2.0.


## 🏆 Performance

### PhyGenBench

**Video-Coder-9B + MiniMax-H3** achieves an average score of **0.67**, compared with **0.61** for MiniMax-H3 alone. It obtains scores of **0.68 / 0.79 / 0.60 / 0.61** on Mechanics, Optics, Thermal, and Material, respectively.

<div align="center">
  <img src="./assets/table1.png" alt="PhyGenBench results" width="100%">
</div>

### VBench-2.0

**Video-Coder-9B + MiniMax-H3** achieves an average score of **78.19%**, improving over MiniMax-H3 (**54.58%**) by **23.61 percentage points**. Its Mechanics score reaches **88.67%**, and its Thermotics score reaches **70.88%**.

<div align="center">
  <img src="./assets/table2.png" alt="VBench-2.0 results" width="100%">
</div>

## 🎥 Qualitative Results

Video-Coder constructs draft videos that guide the generated videos through explicit object layout, motion, and camera control. The examples below show vehicles, human actions, and object interactions.

<div align="center">
  <img src="./assets/qualitative.png" alt="Prompts, draft videos, and generated videos" width="100%">
</div>


## 🚀 Training

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/CYWang735/Video-Coder.git
cd Video-Coder

# build SFT environment
conda create -n video-coder-sft python=3.12 -y
conda activate video-coder-sft
cd Video-Coder-SFT/ms-swift
pip install -e .
pip install flash-linear-attention      # fast Qwen3.5 linear attention (optional, much faster)
cd ../..

# build RL environment
conda create -n video-coder-rl python=3.12 -y
conda activate video-coder-rl
cd Video-Coder-RL/uni-agent
pip install vllm==0.30.0 torch==2.13.0 transformers==5.17.0 "ray[default]==2.58.0" \
    tensordict==0.14.2 accelerate==1.15.0 peft==0.21.1 qwen-vl-utils==0.0.14
grep -vE '^(transformers|tensordict|TransferQueue)' verl/requirements.txt > /tmp/verl-req.txt
pip install -r /tmp/verl-req.txt
pip install --no-deps -e ./verl
pip install --no-deps transferqueue==0.1.11 fla-core==0.5.2 flash-linear-attention==0.5.2
pip install litellm httpx pandas pyarrow pyyaml jinja2 imageio-ffmpeg
```

The RL stage also needs the Blender sandbox of the Video-Coder Harness: an Apptainer image with Blender 4.5 / `bpy` and a [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (≥ 2.4) tool directory, see the [RL guide](./Video-Coder-RL/README.md).

Download the training data [[🤗 Video-Coder-train-data]()] and unzip all files. `sft_data.json` is for SFT: put it under `Video-Coder-SFT/ms-swift/data/`. `rl_train.json` is for RL: each task has a scene description and a ground-truth video, whose path is relative to the JSON file.

### SFT Training

Our SFT simply follows [ms-swift](https://github.com/modelscope/ms-swift), using `sft_data.json` (multi-turn Video-Coder Harness trajectories).

```bash
cd Video-Coder-SFT/ms-swift
bash examples/train/Video-Coder/sft_qwen3_5_full.sh
```

The script fully fine-tunes [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) with a 128k context (sequence parallel 4) on one node with 4 × H100 GPUs. The SFT checkpoint initializes the RL stage.

### RL Training

First, convert `rl_train.json` into the training parquet.

```bash
cd Video-Coder-RL/uni-agent
python -m uni_agent.tasks.video_coder.preprocess --tasks /path/to/rl_train.json \
    --out $PWD/data/video_coder_train.parquet --require-gt --gt-request-sidecar
```

Launch the video generator for the video reward, here we use [MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3) (Ref2VA) served with SGLang. Each server uses 4 GPUs; more servers make each step faster (we used 4).

```bash
cd Minimax-H3-server
bash setup_env.sh && bash download_model.sh     # once
HOST=0.0.0.0 PORT=30011 bash serve.sh
```

The draft video of each accepted rollout is turned into a video by MiniMax-H3, and a VLM judge (`qwen3.8-omni-flash` on DashScope) scores it against the ground-truth video. You may also change to other video generators or judges by revising the `video_reward` section of the task config and [`video_reward.py`](./Video-Coder-RL/uni-agent/uni_agent/tasks/video_coder/video_reward.py).

Then, in the task config [`task_config_video_coder_video_reward_apptainer.yaml`](./Video-Coder-RL/uni-agent/examples/quickstart/training/task_config_video_coder_video_reward_apptainer.yaml), set the launched server URLs in `video_reward.ref2va.base_urls`, a `video_reward.ref2va.shared_dir` visible to the trainer and the servers at the same path, the sandbox `image` and `binds`, and `agent.step_limit: 100` (as in our run). Export your `DASHSCOPE_API_KEY` on every node before starting Ray.

Finally, start the RL training based on the SFT model.

```bash
cd Video-Coder-RL/uni-agent
ray start --head            # on the other nodes: ray start --address=<head>:6379
MODEL_PATH=/path/to/sft_checkpoint CKPTS_DIR=/path/to/ckpts \
TRAIN_FILE=$PWD/data/video_coder_train.parquet TEST_FILE=$PWD/data/video_coder_train.parquet \
NNODES=2 NGPUS_PER_NODE=4 GATEWAY_COUNT=4 CONCURRENCY=64 \
TRAIN_PROMPT_BSZ=8 N_RESP_PER_PROMPT=8 PPO_MINI_BATCH_SIZE=8 NUM_WARMUP_BATCHES=0 \
TOTAL_EPOCHS=3 MAX_RESPONSE_LENGTH=114688 SAVE_FREQ=5 \
bash examples/video_coder/train_video_coder_rl.sh
```

We trained RL on 8 × H100 GPUs (2 nodes), plus 4 GPUs per MiniMax-H3 server. More details are in the [RL guide](./Video-Coder-RL/README.md).

## 🔮 Video Generation

Follow the [MiniMax-H3 deployment and generation guide](./Minimax-H3-server/README.md) to convert draft videos into generated videos.

~~~bash
cd Minimax-H3-server
bash setup_env.sh
bash download_model.sh
bash run_batch.sh examples/demo/manifest.jsonl outputs/demo
~~~

For custom batches, provide a JSONL manifest containing the draft video path and scene description for each sample. See the generator guide for the manifest schema and multi-server configuration.

## Acknowledgements

We thank the open-source projects that support this implementation, including [ms-swift](https://github.com/modelscope/ms-swift), [verl](https://github.com/volcengine/verl), [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent), and [SGLang](https://github.com/sgl-project/sglang).
