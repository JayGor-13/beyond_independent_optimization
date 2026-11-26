"""Tiny real Qwen3-VL models exercise the actual pinned Transformers interfaces."""
import copy
from types import SimpleNamespace

import pytest
import torch

from adapters import Adapter
from core import compress
from loading import unpack_experts, expanded_weights, placement, device_for


def tiny_qwen():
    from transformers import Qwen3VLMoeConfig, Qwen3VLMoeForConditionalGeneration
    config = Qwen3VLMoeConfig(
        text_config={"vocab_size": 100, "hidden_size": 16, "intermediate_size": 32,
                     "moe_intermediate_size": 8, "num_hidden_layers": 2, "num_attention_heads": 2,
                     "num_key_value_heads": 1, "head_dim": 8, "num_experts": 4, "num_experts_per_tok": 2,
                     "rope_scaling": {"rope_type": "default", "mrope_section": [2, 1, 1], "mrope_interleaved": True}},
        vision_config={"depth": 2, "hidden_size": 16, "intermediate_size": 32, "num_heads": 2,
                       "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "out_hidden_size": 16, "num_position_embeddings": 16, "deepstack_visual_indexes": [0, 1]},
        image_token_id=91, video_token_id=94, vision_start_token_id=90, vision_end_token_id=92)
    config._attn_implementation = "eager"
    return Qwen3VLMoeForConditionalGeneration(config).eval()


class FakeTokenizer:
    def decode(self, tokens, **kwargs):
        return ",".join(map(str, tokens))


def adapter_for(model):
    adapter = Adapter.__new__(Adapter)
    adapter.model, adapter.family, adapter.dtype = model, "qwen3vl", torch.float32
    adapter.decoder, adapter.head = model.model.language_model, model.lm_head
    adapter.embedding = adapter.decoder.get_input_embeddings()
    adapter.device = torch.device("cpu")
    adapter.blocks = [(name, b) for name, b in adapter.decoder.named_modules() if hasattr(b, "top_k")]
    adapter.n_experts, adapter.top_k = 4, 2
    adapter.tokenizer = FakeTokenizer()
    return adapter


def prepared_batch(model):
    ids = torch.tensor([[1, 90, 91, 91, 91, 91, 92, 3]])
    raw = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
           "pixel_values": torch.randn(16, 24), "image_grid_thw": torch.tensor([[1, 4, 4]])}
    # Exercise the real prepare/interception path with a local processor fixture.
    class Batch(dict):
        def to(self, device):
            return self
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"
        def __call__(self, **kwargs):
            return Batch(raw)
    adapter = adapter_for(model)
    adapter.processor = Processor()
    batch = adapter.prepare([object()], "Question")
    return adapter, batch


def test_real_qwen_native_equivalence_and_cache_decode():
    torch.manual_seed(42)
    model = tiny_qwen()
    adapter, batch = prepared_batch(model)
    assert adapter.check_native(batch)["next_token_matches"]
    batch["eos"] = [-1]
    topk, weights, names, text = adapter.evaluate(batch, 3)
    assert topk.shape == (2, 4, 2) and len(text.split(",")) == 3
    assert len(names) == 2
    torch.testing.assert_close(torch.tensor(weights.sum(-1)), torch.ones(2, 4))
    # Native generate must match the custom cached decoder, including mRoPE.
    with torch.inference_mode():
        native = model.generate(**batch["native"], max_new_tokens=3, do_sample=False,
                                eos_token_id=None, pad_token_id=0)
    assert text == ",".join(map(str, native[0, batch["input_ids"].shape[1]:].tolist()))


def test_real_qwen_pruning_keeps_deepstack_and_runs_decode():
    adapter, batch = prepared_batch(tiny_qwen())
    pruned = compress(batch, "l2", 0.5, 0)
    topk, weights, _, text = adapter.evaluate(pruned, 2)
    assert topk.shape == (2, 2, 2) and text
    assert len(pruned["deepstack"]) == 2 and all(len(x) == 2 for x in pruned["deepstack"])
    assert all(len(b.gate._forward_hooks) == 0 for _, b in adapter.blocks)


def test_real_qwen_split_experts_preserves_model_logits():
    model = tiny_qwen()
    adapter, batch = prepared_batch(model)
    with torch.inference_mode():
        before = model(**batch["native"], logits_to_keep=1).logits
    state = {k: v.clone() for k, v in model.state_dict().items()}
    packed = unpack_experts(model)
    converted = {name: value for k, tensor in state.items() for name, value in expanded_weights(k, tensor, packed)}
    model.load_state_dict(converted, assign=True, strict=True)
    model.eval()
    with torch.inference_mode():
        after = model(**batch["native"], logits_to_keep=1).logits
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)
    mapping = placement(model, "qwen3vl", 2)
    assert all(device_for(name, mapping) in (0, 1) for name, _ in model.named_parameters())


def test_capture_hooks_removed_on_error():
    adapter, batch = prepared_batch(tiny_qwen())
    batch["visual_mask"] = batch["visual_mask"][:-1]
    with pytest.raises((ValueError, IndexError, RuntimeError)):
        adapter.evaluate(batch, 1)
    assert all(len(b.gate._forward_hooks) == 0 for _, b in adapter.blocks)
