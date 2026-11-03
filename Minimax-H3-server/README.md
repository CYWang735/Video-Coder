# MiniMax-H3 白模视频批量生成

把**白模视频**(灰色无材质的 3D 渲染)和一段**场景描述**交给 MiniMax-H3,生成带材质、光照和声音的成片。
模型沿用白模的物体布局、运动和运镜,按场景描述决定外观。

- 输出:MP4,24 fps,**480p**(16:9 输入得到 864×480),带 32 kHz 立体声
  - "480p" 指**短边** 480(横屏即高度)。宽高比默认跟随白模(`aspect_ratio: auto`)
  - 实际宽高由 SGLang 自动计算:短边 × 宽高比,再各自取最近的 32 的倍数(模型每 32×32 像素是一个 token),
    所以比例与白模略有偏差(864×480 = 1.80,16:9 = 1.78);请求里没法直接指定宽和高
- 时长:默认与白模视频相同(必须在 4–15 秒内,超出部分会被截断)
  - 帧数会被自动补齐到模型要求的长度,成片比白模略长零点几秒(实测 5 s→5.18 s、6 s→6.58 s、7 s→7.3 s)
- 底层:[SGLang Diffusion](https://docs.sglang.io/cookbook/diffusion/MiniMax/MiniMax-H3) 部署的 MiniMax-H3 `Ref2VA` 模型

## 1. 机器要求

| 项目 | 要求 |
|---|---|
| GPU | 4 张 ≥ 80 GB 显存的 NVIDIA 卡(已在 4×H100 94GB 上验证;H200 更快) |
| 驱动 | 支持 CUDA 13(`nvidia-smi` 右上角 CUDA Version ≥ 13.0,驱动 ≥ 580) |
| 磁盘 | 模型约 144 GB + Python 环境约 20 GB |
| 内存 | 建议 ≥ 256 GB |
| 软件 | Linux、conda(或 mamba)、能访问 conda-forge / PyPI / Hugging Face |

## 2. 第一次部署(只做一次)

```bash
cd Sglang_minimaxh3
bash setup_env.sh        # 建 Python 环境到 ./env,约 10 分钟
bash download_model.sh   # 下载模型到 ./models/MiniMax-H3,约 144 GB,中断后重跑可续传
```

- 访问 Hugging Face 慢的话:`HF_ENDPOINT=https://hf-mirror.com bash download_model.sh`
- 想把环境或模型放到别的盘,先改 `config.sh` 里的 `ENV_DIR` / `MODEL_PATH`(模型目录名必须是 `MiniMax-H3`)

## 3. 跑一批视频

### 3.1 下载数据

数据放在 Hugging Face 上(公开,不用登录),约 1.3 GB,2056 个白模视频:

```bash
env/bin/hf download ChaoyangWang/minimax-data --repo-type dataset --local-dir data/minimax-data
```

- 访问慢的话同样可以加 `HF_ENDPOINT=https://hf-mirror.com`;中断后重跑可续传
- 下载完 `data/minimax-data/` 就是下面这种数据包,直接拿去跑:
  `bash run_batch.sh data/minimax-data/manifest.jsonl outputs/minimax-data`

### 3.2 数据包格式

数据包的结构如下(以后如果有别的批次,也是同样格式):

```
my_batch/
├── manifest.jsonl      # 清单,一行一个视频(批量生成只读这个文件)
├── videos/
│   ├── scene_0001.mp4  # 白模视频
│   └── ...
├── excluded.jsonl      # (可能有)整理时剔除的任务及原因,仅供参考
└── content_check.jsonl # (可能有)每个白模视频的画面/运动统计,仅供参考
```

整个文件夹可以拷到任意位置,清单里的路径是相对路径,不用改。

`manifest.jsonl` 每行一个 JSON:

```json
{"id": "scene_0001", "video": "videos/scene_0001.mp4", "scene": "场景描述(英文)"}
```

| 字段 | 必填 | 说明 |
|---|---|---|
| `video` | 是 | 白模视频路径;相对路径以 `manifest.jsonl` 所在目录为准 |
| `scene` | 是 | 场景描述,英文效果最好 |
| `id` | 否 | 输出文件名,不能重复,只用字母、数字、`_`、`-`;默认取视频文件名 |
| `seed` | 否 | 随机种子,默认 42;同样输入 + 同样种子 = 同样结果 |
| `duration` | 否 | 输出秒数(4–15),默认等于白模视频长度 |
| `start_time` | 否 | 从白模视频第几秒开始截取 |
| `aspect_ratio` | 否 | 默认 `auto`(跟随白模);可选 `16:9` `4:3` `1:1` `3:4` `9:16` `21:9` |

- `scene` 会**原样**填进 prompt 模板发给模型,不会被改写或截断,长度也没有限制。
- 白模视频建议 4–15 秒;超过 15 秒只用前 15 秒。分辨率不限,有没有声音都可以。
- 跑之前可以先检查清单(不需要 GPU,也不用起服务):
  `env/bin/python batch_generate.py --manifest my_batch/manifest.jsonl --out outputs/x --dry-run`

### 3.3 运行

先用自带示例确认一切正常(1 个视频,约 10–15 分钟):

```bash
bash run_batch.sh examples/demo/manifest.jsonl outputs/demo
```

跑完后 `outputs/demo/pawn_orbit.mp4` 应该是一个胡桃木棋子立在大理石底座上、镜头环绕一周;
`examples/demo/reference_cmp.png` 是我这边的结果(上排白模、下排生成;当时用的是 768p,现在默认 480p,细节会少一些),可以对照。

正式跑:

```bash
bash run_batch.sh data/minimax-data/manifest.jsonl outputs/minimax-data
```

这一条命令会:启动服务(加载模型 3–7 分钟)→ 逐个生成 → 全部完成后关闭服务。
建议放在 `tmux` / `screen` 里跑,断开 SSH 不会中断。进度会打印在屏幕上,同时写入 `logs/batch-*.log`。

### 3.4 结果

```
outputs/my_batch/
├── scene_0001.mp4      # 生成的视频
├── scene_0001.json     # 发给模型的完整请求 + 结果
└── results.jsonl       # 汇总:每行一个视频的 status(completed / failed)、耗时、失败原因
```

- **中断后续跑**:重新执行同一条命令即可,已经生成的 `.mp4` 会自动跳过。
- **失败的视频**:每个会自动重试 1 次;仍失败的记录在 `results.jsonl` 里(含原因),重跑同一条命令会再试这些。
- **想用不同种子再生成一版**:换一个输出目录,并加 `--seed 7`(对清单里没写 `seed` 的行生效)。

### 3.5 需要多久

4×H100 实测(50 步,**768p**;现在默认 480p,画面像素约为 768p 的 40%,应明显更快,但还没实测):

| 白模时长 | 每个视频 |
|---|---|
| 5 秒 | 约 4.8 分钟 |
| 6 秒 | 约 7 分钟 |
| 7 秒 | 约 8.4 分钟 |

- 时长越长涨得越快(计算量约与 token 数平方成正比);8 秒以上没实测过。
- 另加每次启动服务的 3–7 分钟(一批只启动一次)。
- 例:2056 个以 6 秒为主的视频,按 768p 算单个服务约 250 小时(10 天出头);480p 会少很多。

### 3.6 加速:多开服务

一个服务**一次只生成一个视频**:同时提交多个只会排队;`num_outputs_per_prompt` 也是逐个生成;
SGLang 的请求合批(`--batching-max-size`)H3 不支持。要更快只能多开服务,每个服务 4 张卡,用分片分工:

```bash
# 多台机器:每台跑同一个清单,各自取一份,输出到各自的目录
bash run_batch.sh my_batch/manifest.jsonl outputs/my_batch_s0 --shard-id 0 --num-shards 3   # 机器 0
bash run_batch.sh my_batch/manifest.jsonl outputs/my_batch_s1 --shard-id 1 --num-shards 3   # 机器 1 ...

# 一台 8 卡机器开两个服务(未实测;两条命令间隔几秒再启动)
CUDA_VISIBLE_DEVICES=0,1,2,3 PORT=30011 bash run_batch.sh m.jsonl outputs/s0 --shard-id 0 --num-shards 2
CUDA_VISIBLE_DEVICES=4,5,6,7 PORT=30021 bash run_batch.sh m.jsonl outputs/s1 --shard-id 1 --num-shards 2
```

每个分片用自己的输出目录(`results.jsonl` 不会被两个进程同时写);全部跑完后把各目录里的 `.mp4` 合到一起即可。
**分片数和编号一旦开始就不要改**,否则续跑时分到每个分片的视频会变。

## 作为 RL video reward 的服务

Video-Coder-RL 的 video reward 通过 HTTP 调用这里部署的 Ref2VA 服务(见 `../Video-Coder-RL/README.md`)。
每台 4 卡机器起一个常驻服务,训练端把所有服务的地址填进 task config 的 `similarity.ref2va.base_urls`:

```bash
bash setup_env.sh                          # 每台新机器一次
HOST=0.0.0.0 PORT=30011 NUM_GPUS=4 bash serve.sh
curl -s -o /dev/null -w '%{http_code}\n' http://<host>:30011/health   # 200 = 模型已加载完(约 3-7 分钟)
```

- 训练端在 `similarity.ref2va.shared_dir` 里写白模视频,并以 `file://` 路径交给服务,所以这个目录在服务机器上必须是同一个路径(共享存储)
- 训练时我们用了 4 个服务;服务越多,每步等 H3 出视频的时间越短

## 4. 常用设置

都在 `config.sh` 里,也可以在命令前临时指定,例如 `NUM_GPUS=8 bash run_batch.sh ...`

| 设置 | 默认 | 说明 |
|---|---|---|
| `NUM_GPUS` | 4 | 用几张卡 |
| `TOPOLOGY` | auto | 并行方式:H200 用 `ulysses4`,其它用 `tp2u2`;显存不够时改 `tp4` |
| `PORT` | 30011 | 服务端口;同一台机器跑两个服务时要错开 |
| `CUDA_VISIBLE_DEVICES` | 全部 | 指定用哪几张卡,如 `0,1,2,3` |

`run_batch.sh` 后面还能加 `batch_generate.py` 的选项(`python batch_generate.py -h` 查看全部):

| 选项 | 说明 |
|---|---|
| `--seed N` | 默认种子 |
| `--steps N` | 去噪步数,默认 50(官方配方);减少会更快但质量下降 |
| `--short-edge N` | 输出短边,默认 480(16:9 → 864×480)。官方只验证过 768(16:9 → 1344×768),其它值服务日志里会有一条 `outside the verified configuration` 警告,属正常 |
| `--shard-id i --num-shards n` | 把清单分成 n 份只跑第 i 份,多台机器分工用 |
| `--dry-run` | 只打印第一个请求,不生成;用来检查清单 |

## 5. 常见问题

| 现象 | 处理 |
|---|---|
| `port 30011 is already in use` | 上一次的服务还在:`pkill -f "sglang serve"`,或换 `PORT=30021` |
| 服务启动失败 | 看 `logs/server-*.log` 最后几十行 |
| `CUDA driver version is insufficient` | 驱动太旧,需要支持 CUDA 13 |
| 日志里出现 `nvcc` / `Failed to build JIT module` | 装 CUDA 13 toolkit 后 `export CUDA_HOME=/usr/local/cuda-13.0` 再跑 |
| `CUDA out of memory` | `TOPOLOGY=tp4 bash run_batch.sh ...` |
| 生成的画面和白模差别很大 | 正常现象之一,见下一节 |

## 6. 关于生成效果(看结果前建议读一下)

- 模型把白模当作**参考**而不是逐帧编辑,运动和镜头大体跟随白模,但不保证每帧对齐。
- 外观完全由 `scene` 决定:描述里没写材质 / 背景时,成片可能几乎还是白模的样子;只写了 "chess pawn" 时,模型可能自己加一整盘棋子。
  写清材质、光照、背景,并说明 "No other objects" 时效果好得多,示例 `examples/demo/manifest.jsonl` 就是这种写法。
- 本项目没有"参考强度"之类的参数可调,影响结果的主要是 `scene` 的写法和 `seed`。
- prompt 模板在 `batch_generate.py` 的 `DEFAULT_TEMPLATE`,不要随意改动,改了请告诉我。

## 7. 文件说明

| 文件 | 作用 |
|---|---|
| `run_batch.sh` | **日常只用这个**:起服务 → 批量生成 → 关服务 |
| `config.sh` | 路径、GPU、端口等设置 |
| `setup_env.sh` / `download_model.sh` | 第一次部署用 |
| `serve.sh` | 单独启动服务(调试用,前台运行) |
| `batch_generate.py` | 批量客户端(需要服务已启动) |
| `tools/memfd_compat.py` | 环境兼容补丁,由 `setup_env.sh` 自动安装 |
| `tools/build_manifest_from_tasks.py` | 把 Video-Coder 任务目录(`<task>/task.json` + `<task>/final_run/output.mp4`)整理成数据包 |
| `examples/demo/` | 示例数据包 |
| `cluster/helma/` | 原作者在 NHR Helma 集群上的配置,其它机器用不到 |
| `env/`、`models/`、`data/`、`.cache/`、`.server/`、`logs/`、`outputs/` | 运行时生成;`.server/` 里是服务端的视频副本,可随时删除 |
