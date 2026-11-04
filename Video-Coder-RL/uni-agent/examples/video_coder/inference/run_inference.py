"""Video-Coder inference: examples/inference/parallel_infer_verl.py with the RL rollout engine settings (Qwen3.5-9B).

Same arguments as parallel_infer_verl.py. On top of its config this sets what train_qwen3p5_fsdp.sh sets for the
rollout engine, so evaluation samples the policy the way training did: prompt length 8k, chunked prefill over the
whole context, vLLM mamba_cache_mode=align (Qwen3.5 GatedDeltaNet prefix caching) and async_scheduling off,
max_num_seqs 20, multi-turn on.
"""

import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "inference"))
import parallel_infer_verl as piv  # noqa: E402

PROMPT_LENGTH = 8192
piv.DEFAULT_PROMPT_LENGTH = PROMPT_LENGTH  # read by init_config for rollout.prompt_length / data.max_prompt_length

_init_config = piv.init_config


def init_config(args, **kwargs):
    config = _init_config(args, **kwargs)
    total = PROMPT_LENGTH + args.response_length
    for key, value in {
        "actor_rollout_ref.rollout.enable_chunked_prefill": True,
        "actor_rollout_ref.rollout.max_num_batched_tokens": total,
        "actor_rollout_ref.rollout.max_num_seqs": int(os.getenv("ROLLOUT_MAX_NUM_SEQS", 20)),
        "actor_rollout_ref.rollout.enforce_eager": False,
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "actor_rollout_ref.rollout.multi_turn.max_parallel_calls": 1,
        "actor_rollout_ref.rollout.engine_kwargs.vllm.mamba_cache_mode": "align",
        "actor_rollout_ref.rollout.engine_kwargs.vllm.async_scheduling": False,
        "data.trust_remote_code": True,
    }.items():
        OmegaConf.update(config, key, value, force_add=True)
    return config


piv.init_config = init_config

if __name__ == "__main__":
    piv.main()
