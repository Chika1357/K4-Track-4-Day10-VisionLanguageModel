import pytest
import torch

from lab.data import GroundingCollator, encode_example


class FakeProcessor:
    """Known tokens let us inspect loss boundaries without downloading a model."""
    def __init__(self, corrupt_prefix=False):
        self.corrupt_prefix = corrupt_prefix

    def apply_chat_template(self, messages, add_generation_prompt, **kwargs):
        tokens = [101, 102, 103] if add_generation_prompt else [101, 102, 103, 201, 202, 299]
        if not add_generation_prompt and self.corrupt_prefix:
            tokens[0] = 999
        return {"input_ids": torch.tensor([tokens]),
                "attention_mask": torch.ones((1, len(tokens)), dtype=torch.long),
                "mm_token_type_ids": torch.tensor([[0, 1, 0] + [0]*(len(tokens)-3)]),
                "pixel_values": torch.ones((4, 3)),
                "image_grid_thw": torch.tensor([[1, 2, 2]])}


def test_only_answer_and_eos_are_supervised():
    result = encode_example(None, "prompt", "[]", FakeProcessor())
    assert result["labels"].tolist() == [-100, -100, -100, 201, 202, 299]
    assert result["input_ids"].tolist() == [101, 102, 103, 201, 202, 299]
    assert result["mm_token_type_ids"].tolist() == [0, 1, 0, 0, 0, 0]


def test_prefix_mismatch_stops_training():
    with pytest.raises(ValueError, match="prompt prefix"):
        encode_example(None, "prompt", "[]", FakeProcessor(corrupt_prefix=True))


def test_collator_masks_padding_and_concatenates_image_grids():
    first = encode_example(None, "prompt", "[]", FakeProcessor())
    second = {key: value.clone() for key, value in first.items()}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
        second[key] = second[key][:5]
    second["pixel_values"] = torch.full((4, 3), 2.0)
    combined = GroundingCollator(17)([first, second])
    assert combined["input_ids"].shape == (2, 6)
    assert combined["input_ids"][1, -1].item() == 17
    assert combined["attention_mask"][1, -1].item() == 0
    assert combined["labels"][1, -1].item() == -100
    assert combined["mm_token_type_ids"][1, -1].item() == 0
    assert combined["pixel_values"].shape == (8, 3)
    assert combined["image_grid_thw"].tolist() == [[1, 2, 2], [1, 2, 2]]
    assert torch.equal(combined["pixel_values"][4:], second["pixel_values"])
