import pytest

from lab.modeling import SKIP_QUANTIZATION_MODULES


def test_installed_transformers_excludes_nested_visual_modules():
    pytest.importorskip("transformers")
    from transformers.integrations.bitsandbytes import should_convert_module
    for name in ("model.visual.blocks.0.attn.qkv", "model.visual.blocks.0.attn.proj",
                 "model.visual.merger.linear_fc1", "model.visual.deepstack_merger_list.0.linear_fc1", "lm_head"):
        assert not should_convert_module(name, list(SKIP_QUANTIZATION_MODULES)), name
    assert should_convert_module("model.language_model.layers.0.self_attn.q_proj",
                                 list(SKIP_QUANTIZATION_MODULES))
