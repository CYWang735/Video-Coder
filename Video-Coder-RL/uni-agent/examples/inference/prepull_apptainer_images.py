"""Pre-pull the Apptainer SIF images a dataset needs, before an inference/training run.

Run on the node that will run the sandboxes (SIFs are cached node-locally by
default), with a proxy if the node needs one to reach the registry::

    python examples/inference/prepull_apptainer_images.py \\
        --data-path ~/data/swe_agent/swe_bench_verified.parquet \\
        --task-config examples/quickstart/inference/task_config_claude_code_apptainer.yaml \\
        --limit 50

Each row's image is resolved through its task's ``sandbox`` config (``image_map``,
``sif_dir``, ``max_concurrent_pulls``, ...), exactly as the run will, so the run
then starts every sandbox from cache instead of pulling inside
``SANDBOX_STARTUP_TIMEOUT``.
"""

import argparse
import asyncio
import time

import pandas as pd
import yaml

from uni_agent.sandbox import SandboxConfig, build_sandbox


def _sandbox_configs(data_path: str, task_config: str, limit: int | None) -> dict[str, SandboxConfig]:
    """Map each distinct image to the SandboxConfig its task would build it with."""
    tasks = {t["name"]: t.get("sandbox") or {} for t in yaml.safe_load(open(task_config))}
    rows = pd.read_parquet(data_path)["extra_info"]
    configs: dict[str, SandboxConfig] = {}
    for extra_info in rows[:limit] if limit else rows:
        task = extra_info["tools_kwargs"]["task"]
        sandbox = {**tasks.get(task["name"], {}), **(task.get("sandbox") or {})}
        if sandbox.get("provider") != "apptainer":
            continue
        config = SandboxConfig(**sandbox)
        configs.setdefault(config.image, config)
    return configs


async def _pull_all(configs: dict[str, SandboxConfig]) -> int:
    done = failed = 0
    total = len(configs)

    async def pull(config: SandboxConfig) -> None:
        nonlocal done, failed
        t = time.time()
        try:
            sif = await build_sandbox(config)._resolve_sif()
            status = f"ok   {time.time() - t:6.0f}s  {sif}"
        except Exception as exc:  # report and keep going; the run will retry this image
            failed += 1
            status = f"FAIL {time.time() - t:6.0f}s  {config.image}: {exc}"
        done += 1
        print(f"[{done}/{total}] {status}", flush=True)

    # Concurrency is bounded inside _resolve_sif by the provider's pull slots.
    await asyncio.gather(*(pull(c) for c in configs.values()))
    return failed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--task-config", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Only the first N dataset rows.")
    args = parser.parse_args()

    configs = _sandbox_configs(args.data_path, args.task_config, args.limit)
    print(f"{len(configs)} distinct apptainer images to ensure", flush=True)
    failed = asyncio.run(_pull_all(configs))
    print(f"done: {len(configs) - failed} ok, {failed} failed", flush=True)
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
