# Helma(NHR@FAU)上的用法

这里是在 Helma 集群上的配置,在其它机器上用不到。主流程的脚本和项目根目录是同一套,
`site_env.sh` 通过 `H3_SITE_ENV` 接入 `config.sh`,改动的是以下几项:

- 环境、缓存、服务工作目录都放在节点本地 `/tmp/$USER`(作业结束即清空):`/home/hpc` 空间不多,`/hnvme` 超出了文件数配额
- 走 `proxy.nhr.fau.de` 代理;conda 来自 `module load python/3.12-conda`;nvcc 来自 spack 的 cuda 13.0.2
- `HOME` 指到 `/tmp`,因为 SGLang 的 JIT 缓存会直接写 `~/.cache`(在 /hnvme 上)
- 模型用已下载在 `/hnvme/workspace/v189bd13-VOpera/models/MiniMax-H3` 的那份;它的 `Ref2VA/processor`
  是断链,所以在 `/tmp` 搭一个覆盖目录,processor 用 `model_patch/` 里从 HF 取回的文件

## 在已有交互作业里用

新节点上第一次会先建环境(约 10 分钟)。

```bash
export H3_SITE_ENV=$PWD/cluster/helma/site_env.sh
srun --jobid=<JOBID> --overlap -N1 -n1 bash setup_env.sh                       # 每个新节点一次
srun --jobid=<JOBID> --overlap -N1 -n1 --gres=gpu:4 bash run_batch.sh <manifest> <out>
```

`examples/` 里是之前在这里跑过的测试清单(绝对路径,只在本账号下可用)。
