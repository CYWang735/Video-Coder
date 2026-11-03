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

## 🔍 Datasets

We construct **Video-Coder-SFT-10k** and **Video-Coder-RL-4k** for training the coding agent.

| Dataset | Size | Purpose |
| --- | --- | --- |
| Video-Coder-SFT-10k | 10,000 | Supervised learning from agentic coding trajectories |
| Video-Coder-RL-4k | 4,000 | Agentic reinforcement learning with reference videos |

The data construction pipeline collects diverse text prompts, generates multi-turn coding trajectories with the Video-Coder Harness, synthesizes ground-truth reference videos from prompts and draft videos, and applies quality filtering.

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

## 📐 Getting Started

~~~bash
git clone https://github.com/CYWang735/Video-Coder.git
cd Video-Coder
~~~

| Component | Location | Description |
| --- | --- | --- |
| Supervised fine-tuning | [Video-Coder-SFT](./Video-Coder-SFT/) | SFT code based on ms-swift |
| Agentic reinforcement learning | [Video-Coder-RL](./Video-Coder-RL/) | Harness, sandbox, rollout, and reward integration |
| Video generator | [Minimax-H3-server](./Minimax-H3-server/) | MiniMax-H3 serving and batch generation from draft videos |

## 🚀 Training

### Stage 1: Supervised Fine-Tuning

The SFT implementation is provided under [Video-Coder-SFT/ms-swift](./Video-Coder-SFT/ms-swift/). Use the SFT checkpoint to initialize the RL stage.

### Stage 2: Agentic Reinforcement Learning

Follow the [RL setup and training guide](./Video-Coder-RL/README.md) to prepare the training environment, Blender sandbox, task data, and video reward services.

The main training entry point is:

~~~bash
cd Video-Coder-RL/uni-agent
bash examples/video_coder/train_video_coder_rl.sh
~~~

Before launching, configure the SFT checkpoint, training and validation data, sandbox paths, and video generator endpoints as described in the RL guide.

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
