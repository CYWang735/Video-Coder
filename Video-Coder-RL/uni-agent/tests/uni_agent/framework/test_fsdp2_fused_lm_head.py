"""verl's fused Qwen3.5 PPO head under FSDP2, where lm_head's weight is used without calling lm_head.

forward_with_{torch,triton}_backend in verl/models/transformers/qwen3_5.py gather lm_head.weight themselves
(_full_lm_head_weight), so FSDP2 neither gathers that weight nor reduces its gradient. Against an unsharded
model run on each rank's own batch: the output matches, and the gradients of lm_head and of a regular
FSDP-managed layer are both the average over the ranks (FSDP2's convention). With CPUOffloadPolicy, which
keeps the shards on the CPU, and without.

    torchrun --nproc_per_node=2 -m pytest -q tests/uni_agent/framework/test_fsdp2_fused_lm_head.py
"""

import os

import pytest
import torch
import torch.distributed as dist

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.level1,
    pytest.mark.skipif("LOCAL_RANK" not in os.environ, reason="needs torchrun and 2+ GPUs"),
]

VOCAB, HIDDEN, TOKENS = 1024, 64, 48


@pytest.fixture(scope="module", autouse=True)
def _process_group():
    dist.init_process_group("cpu:gloo,cuda:nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    yield
    dist.destroy_process_group()


class TinyLM(torch.nn.Module):
    """One layer and an output head used as verl's fused head uses it: lm_head.forward never runs."""

    def __init__(self):
        super().__init__()
        self.body = torch.nn.Linear(HIDDEN, HIDDEN)
        self.lm_head = torch.nn.Linear(HIDDEN, VOCAB, bias=False)

    def forward(self, x):
        from verl.models.transformers.qwen3_5 import _full_lm_head_weight

        hidden = self.body(x)
        weight = _full_lm_head_weight(self.lm_head, hidden.device)
        return hidden.to(weight.dtype) @ weight.t()


def _full(tensor):
    from torch.distributed.tensor import DTensor

    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor


@pytest.mark.parametrize("cpu_offload", [True, False], ids=["cpu-offload", "on-gpu"])
def test_fused_head_matches_unsharded_model(cpu_offload):
    from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    torch.manual_seed(0)  # the same weights on every rank
    ref = TinyLM().cuda()
    model = TinyLM()
    model.load_state_dict(ref.state_dict())
    if not cpu_offload:
        model.cuda()
    kwargs = {"offload_policy": CPUOffloadPolicy(pin_memory=True)} if cpu_offload else {}
    for module in (model.body, model.lm_head, model):
        fully_shard(module, **kwargs)

    # each rank its own batch, as in data-parallel training
    x = torch.randn(TOKENS, HIDDEN, device="cuda", generator=torch.Generator("cuda").manual_seed(100 + dist.get_rank()))
    out = model(x)
    out.pow(2).mean().backward()
    out_ref = ref(x)
    out_ref.pow(2).mean().backward()
    torch.testing.assert_close(out, out_ref)

    for name in ("body.weight", "lm_head.weight"):
        expected = ref.get_parameter(name).grad.clone()
        dist.all_reduce(expected, op=dist.ReduceOp.AVG)
        grad = model.get_parameter(name).grad
        assert isinstance(grad, DTensor), name
        assert grad.to_local().device.type == ("cpu" if cpu_offload else "cuda"), name
        torch.testing.assert_close(_full(grad).cuda(), expected, msg=lambda m, n=name: f"{n} gradient: {m}")
