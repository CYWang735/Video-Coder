"""mRoPE position_ids survive the TransferQueue round trip for any per-rank batch size.

The framework packs per-sample ``(4, seq_len)`` position ids as a jagged NestedTensor ragged on the
sequence (``_ragged_idx=2``). TransferQueue re-packs the per-sample tensors with
``torch.nested.as_nested_tensor(..., layout=torch.jagged)``, which can only be ragged on dim 0: for
samples of different lengths that fails and it falls back to a strided layout, but a single sample (or
samples of equal length) comes back jagged on the 4 mRoPE rows. ``maybe_fix_3d_position_ids`` must
restore the sequence layout in that case instead of reading row offsets as sequence lengths.
"""

from __future__ import annotations

import pickle

import pytest
import torch
from tensordict import TensorDict

from verl.utils import tensordict_utils as tu


def _position_ids(length: int) -> torch.Tensor:
    return torch.arange(length).view(1, length).expand(4, length).clone()  # (4, seq_len)


def _tq_pack(samples: list[torch.Tensor]) -> torch.Tensor:
    """What TransferQueue's storage does with a list of per-sample tensors."""
    try:
        from transfer_queue.storage.managers.simple_storage_manager import AsyncSimpleStorageManager

        pack = AsyncSimpleStorageManager._pack_field_values
    except (ImportError, AttributeError):  # same packing, if TransferQueue is absent or renamed it
        try:
            return torch.nested.as_nested_tensor(samples, layout=torch.jagged)
        except (RuntimeError, TypeError):
            return torch.nested.as_nested_tensor(samples, layout=torch.strided)
    return pack(samples)


def _per_sample(td: TensorDict) -> list[tuple[int, ...]]:
    return [tuple(chunk["position_ids"].values().shape) for chunk in tu.chunk_tensordict(td, len(td))]


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("lengths", [[37], [21, 21], [21, 50, 9]], ids=["single", "equal-lengths", "ragged"])
def test_tq_repacked_position_ids_keep_their_sequence_length(lengths):
    samples = [_position_ids(length) for length in lengths]
    td = TensorDict({"position_ids": _tq_pack(samples)}, batch_size=[len(lengths)])

    tu.maybe_fix_3d_position_ids(td)
    tu.maybe_fix_3d_position_ids(td)  # idempotent: workers and engines each call it on the same batch

    assert _per_sample(td) == [(4, length) for length in lengths]
    for chunk, sample in zip(tu.chunk_tensordict(td, len(td)), samples, strict=True):
        assert torch.equal(chunk["position_ids"].values(), sample)


@pytest.mark.cpu
@pytest.mark.level0
@pytest.mark.parametrize("lengths", [[37], [21, 50]])
def test_pickled_position_ids_only_need_the_ragged_index(lengths):
    """The original case: a correctly packed tensor whose _ragged_idx was lost in pickling."""
    td = TensorDict(
        {"position_ids": tu.nested_tensor_from_tensor_list([_position_ids(n) for n in lengths], ragged_idx=2)},
        batch_size=[len(lengths)],
    )
    td = pickle.loads(pickle.dumps(td))

    tu.maybe_fix_3d_position_ids(td)

    position_ids = td["position_ids"]
    assert position_ids._ragged_idx == 2
    assert tuple(position_ids.values().shape) == (4, sum(lengths))
    assert _per_sample(td) == [(4, n) for n in lengths]
