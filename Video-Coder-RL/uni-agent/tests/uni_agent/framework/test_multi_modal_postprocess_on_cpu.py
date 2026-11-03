from __future__ import annotations

import pytest
import torch

from tests.uni_agent.support import FakeProcessor


@pytest.mark.cpu
@pytest.mark.level0
def test_compute_multi_modal_inputs_returns_empty_dict_without_processor():
    """Without a processor, multimodal postprocess is a no-op and returns an
    empty field map even when raw image data is present."""
    from uni_agent.framework.multi_modal_postprocess import compute_multi_modal_inputs

    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)

    assert compute_multi_modal_inputs(None, input_ids, {"images": ["image://a.png"]}) == {}


@pytest.mark.cpu
@pytest.mark.level0
def test_compute_multi_modal_inputs_returns_image_tensors_and_images_seqlens():
    """Image inputs are converted through the processor while generated
    ``input_ids`` and ``attention_mask`` are stripped from the TQ payload."""
    from uni_agent.framework.multi_modal_postprocess import compute_multi_modal_inputs

    processor = FakeProcessor()
    input_ids = torch.tensor([[11, processor.image_token_id, 12]], dtype=torch.long)

    multi_modal_inputs = compute_multi_modal_inputs(
        processor,
        input_ids,
        {"images": ["image://a.png"]},
        {"max_pixels": 1024},
    )

    assert "input_ids" not in multi_modal_inputs
    assert "attention_mask" not in multi_modal_inputs
    assert tuple(multi_modal_inputs["pixel_values"].shape) == (1, 3, 2, 2)
    assert multi_modal_inputs["image_grid_thw"].tolist() == [[1, 2, 3]]
    assert multi_modal_inputs["images_seqlens"].tolist() == [6]
    assert "mm_token_type_ids" in multi_modal_inputs
    assert processor.last_processor_call["extra_kwargs"] == {"max_pixels": 1024}


@pytest.mark.cpu
@pytest.mark.level0
def test_compute_position_ids_text_only_sample_uses_mrope_layout():
    """With an M-RoPE processor, a text-only sample gets the same (1, 4, L) layout as image
    samples (sequential positions in every row), so a batch mixing image and text-only
    trajectories stacks into one tensor; get_rope_index is not needed for it."""
    from uni_agent.framework.multi_modal_postprocess import compute_position_ids

    processor = FakeProcessor()
    input_ids = torch.tensor([[7, 8, 9, 10]], dtype=torch.long)
    attention_mask = torch.tensor([[0, 1, 1, 1]], dtype=torch.long)

    position_ids = compute_position_ids(processor, input_ids, attention_mask, {})

    assert tuple(position_ids.shape) == (1, 4, 4)
    assert position_ids[0, 0].tolist() == [1, 0, 1, 2]  # text row, as in the image path
    assert position_ids[0, 1:].tolist() == [[0, 0, 1, 2]] * 3  # what get_rope_index gives text
    assert processor.last_get_rope_index_call is None


@pytest.mark.cpu
@pytest.mark.level0
def test_compute_position_ids_text_only_sample_tolerates_placeholder_ids():
    """A text-only trajectory can contain image/video placeholder ids as plain text (e.g. a
    tool output printing "<|image_pad|>"); they are positioned as text, not as images
    without a grid."""
    from uni_agent.framework.multi_modal_postprocess import compute_position_ids

    processor = FakeProcessor()
    input_ids = torch.tensor([[7, processor.image_token_id, processor.video_token_id, 10]], dtype=torch.long)

    position_ids = compute_position_ids(processor, input_ids, torch.ones_like(input_ids), {})

    assert position_ids[0].tolist() == [[0, 1, 2, 3]] * 4
    assert processor.last_get_rope_index_call is None


@pytest.mark.cpu
@pytest.mark.level0
def test_compute_position_ids_keeps_1d_without_mrope_processor():
    """Without a processor, or with one that has no get_rope_index, ids stay 1D."""
    from uni_agent.framework.multi_modal_postprocess import compute_position_ids

    input_ids = torch.tensor([[7, 8, 9]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    assert compute_position_ids(None, input_ids, attention_mask, {}).tolist() == [[0, 1, 2]]
    assert compute_position_ids(object(), input_ids, attention_mask, {}).tolist() == [[0, 1, 2]]


@pytest.mark.cpu
@pytest.mark.level0
def test_compute_position_ids_returns_multimodal_shape_with_processor():
    """With multimodal processor outputs, position ids combine the text row
    with the processor-provided vision RoPE rows."""
    from uni_agent.framework.multi_modal_postprocess import compute_position_ids

    processor = FakeProcessor()
    input_ids = torch.tensor(
        [[11, processor.image_token_id, processor.video_token_id, 12]],
        dtype=torch.long,
    )
    attention_mask = torch.ones_like(input_ids)
    multi_modal_inputs = {
        "image_grid_thw": torch.tensor([[1, 2, 3]], dtype=torch.long),
        "video_grid_thw": torch.tensor([[1, 3, 4]], dtype=torch.long),
        "mm_token_type_ids": torch.ones_like(input_ids),
    }

    position_ids = compute_position_ids(processor, input_ids, attention_mask, multi_modal_inputs)

    assert tuple(position_ids.shape) == (1, 4, 4)
    assert position_ids[0, 0].tolist() == [0, 1, 2, 3]
    assert processor.last_get_rope_index_call["mm_token_type_ids"].tolist() == [[0, 1, 2, 0]]
