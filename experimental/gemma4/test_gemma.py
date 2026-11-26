"""Gemma integration tests: actual pinned model code, small random weights."""
import io
import json
import struct
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import transformers

if transformers.__version__ != "5.5.0":
    pytest.skip("Gemma uses its separate Transformers 5.5.0 environment", allow_module_level=True)

from core import compress
from gemma_adapter import GemmaAdapter
from gemma_loading import GemmaSplitExperts, HubRangeReader, range_weights, read_header, split_experts
from gemma_loading import serial_vision, weight_dtype
from gemma_numerics import trace_finite, require_finite


def test_vision_precision_policy_and_serial_frame_order():
    model = tiny_gemma()
    _, batch = fixture(model, 2)
    raw = batch["native"]
    with torch.inference_mode():
        expected = model(**raw, logits_to_keep=1).logits
        serial_vision(model)
        actual = model(**raw, logits_to_keep=1).logits
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert weight_dtype("model.vision_tower.encoder.layers.0.mlp.weight", torch.float16) == torch.float32
    assert weight_dtype("model.embed_vision.embedding_projection.weight", torch.float16) == torch.float32
    assert weight_dtype("model.language_model.layers.0.router.scale", torch.float16) == torch.float16
    assert weight_dtype("model.vision_tower.std_bias", torch.bfloat16) == torch.bfloat16


def test_nonfinite_trace_names_first_failure_and_removes_hooks(tmp_path):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False), torch.nn.Identity()).half()
    with torch.no_grad():
        model[0].weight.fill_(40000)
    path = tmp_path / "numerics.jsonl"
    with pytest.raises(FloatingPointError, match="first non-finite output at 0"):
        with trace_finite(model, "native", path):
            model(torch.ones(1, 2).half())
    record = json.loads(path.read_text())
    assert record["stage"] == "native" and record["inf_count"] == 2
    assert all(not module._forward_hooks for module in model.modules())
    with pytest.raises(FloatingPointError, match="generation_logits"):
        require_finite(torch.tensor([float('nan')]), "evaluate", "generation_logits")


def test_fp32_vision_keeps_large_intermediates_finite_with_half_text():
    # Force a finite but FP16-inexpressible vision activation. This exercises the
    # actual vision path and native FP32-to-text boundary, not a mocked decoder.
    model = tiny_gemma().half()
    model.model.vision_tower.float()
    model.model.embed_vision.float()
    original = model.model.vision_tower.patch_embedder.forward
    def large_embedding(*args, **kwargs):
        return original(*args, **kwargs) + 70000
    model.model.vision_tower.patch_embedder.forward = large_embedding
    serial_vision(model)
    adapter, batch = fixture(model, 2)
    assert batch["embeddings"].dtype == torch.float16
    assert adapter.check_native(batch)["next_token_matches"]
    assert all(not module._forward_hooks for module in model.modules())


def tiny_gemma():
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    config = Gemma4Config(
        text_config={"vocab_size": 100, "hidden_size": 16, "intermediate_size": 24,
                     "hidden_size_per_layer_input": 0, "moe_intermediate_size": 8,
                     "enable_moe_block": True, "num_experts": 4, "top_k_experts": 2,
                     "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1,
                     "num_global_key_value_heads": 1, "head_dim": 8, "global_head_dim": 8,
                     "attention_k_eq_v": True, "sliding_window": 16,
                     "layer_types": ["sliding_attention", "full_attention"],
                     "use_bidirectional_attention": "vision", "final_logit_softcapping": 30.0,
                     "rope_parameters": {"full_attention": {"rope_type": "default", "rope_theta": 10000.0},
                                         "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0}}},
        vision_config={"hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
                       "num_attention_heads": 2, "num_key_value_heads": 2, "head_dim": 8,
                       "patch_size": 2, "pooling_kernel_size": 3, "position_embedding_size": 32,
                       "standardize": True},
        image_token_id=91, video_token_id=94, audio_token_id=95, boi_token_id=90,
        eoi_token_id=92, eos_token_id=[1, 2], audio_config=None)
    config._attn_implementation = "eager"
    return Gemma4ForConditionalGeneration(config).eval()


def fixture(model, images=1):
    adapter = GemmaAdapter.__new__(GemmaAdapter)
    adapter.model, adapter.dtype, adapter.family = model, torch.float32, "gemma4"
    adapter.decoder = model.model.language_model
    adapter.embedding = adapter.decoder.get_input_embeddings()
    adapter.device = torch.device("cpu")
    adapter.n_experts, adapter.top_k = 4, 2
    adapter.blocks = [(n, m) for n, m in adapter.decoder.named_modules() if type(m).__name__ == "Gemma4TextRouter"]
    adapter.tokenizer = SimpleNamespace(decode=lambda tokens, **kwargs: ",".join(map(str, tokens)))
    ids = torch.tensor([[3] + [90, 91, 91, 91, 91, 92, 4] * images + [5]])
    grid = torch.stack(torch.meshgrid(torch.arange(6), torch.arange(6), indexing="ij"), -1).reshape(36, 2)
    raw = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
           "mm_token_type_ids": ids.eq(91).long(), "pixel_values": torch.randn(images, 36, 12),
           "image_position_ids": grid[None].expand(images, -1, -1)}
    class Batch(dict):
        def to(self, device):
            return self
    adapter.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: "prompt",
                                         __call__=lambda **kwargs: Batch(raw))
    class Processor:
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "prompt"
        def __call__(self, **kwargs):
            assert len(kwargs["images"][0]) == images
            return Batch(raw)
    adapter.processor = Processor()
    return adapter, adapter.prepare([object()] * images, "Question", temporal=images > 1)


def test_gemma_native_logits_routes_and_cached_generation():
    torch.manual_seed(27)
    model = tiny_gemma()
    adapter, batch = fixture(model, 2)
    assert adapter.check_native(batch)["next_token_matches"]
    batch["eos"] = [-1]
    routes, weights, names, answer = adapter.evaluate(batch, 3)
    assert routes.shape == (2, 8, 2)
    np.testing.assert_allclose(weights.sum(-1), 1, atol=1e-6)
    assert len(names) == 2
    assert batch["frame_ids"][batch["visual_mask"]].tolist() == [0]*4 + [1]*4
    with torch.inference_mode():
        native = model.generate(**batch["native"], max_new_tokens=3, do_sample=False,
                                eos_token_id=None, pad_token_id=0)
    assert answer == ",".join(map(str, native[0, batch["input_ids"].shape[1]:].tolist()))


def test_gemma_pruning_preserves_modality_alignment_and_runs():
    adapter, batch = fixture(tiny_gemma())
    for method in ["l2", "random"]:
        pruned = compress(batch, method, 0.5, 3)
        assert pruned["mm_token_type_ids"].eq(1).sum() == 2
        assert torch.equal(pruned["mm_token_type_ids"][0].bool(), pruned["visual_mask"])
        ids, _, _, answer = adapter.evaluate(pruned, 2)
        assert ids.shape == (2, 2, 2) and answer


def test_gemma_split_experts_preserve_scaled_routing_output():
    model = tiny_gemma()
    original = model.model.language_model.layers[0].experts
    replacement = GemmaSplitExperts(original)
    with torch.no_grad():
        for expert in range(4):
            replacement.gate_up[expert].weight.copy_(original.gate_up_proj[expert])
            replacement.down[expert].weight.copy_(original.down_proj[expert])
    hidden = torch.randn(5, 16)
    indices = torch.tensor([[0, 1], [2, 1], [3, 2], [1, 3], [0, 2]])
    weights = torch.rand(5, 2) * 3
    torch.testing.assert_close(replacement(hidden, indices, weights), original(hidden, indices, weights))


def test_gemma_route_mass_excludes_learned_output_scaling():
    model = tiny_gemma()
    adapter, batch = fixture(model)
    reference = {}
    hooks = []
    for name, router in adapter.blocks:
        with torch.no_grad():
            router.per_expert_scale.copy_(torch.tensor([0.1, 2., 3., 4.]))
        hooks.append(router.register_forward_hook(lambda m, a, o, name=name: reference.update({name: o})))
    try:
        ids, weights, names, _ = adapter.evaluate(batch, 0)
    finally:
        for hook in hooks:
            hook.remove()
    for layer, name in enumerate(names):
        probabilities, scaled_weights, selected = reference[name]
        expected = probabilities.gather(-1, selected).float()
        expected /= expected.sum(-1, keepdim=True)
        np.testing.assert_allclose(weights[layer], expected[batch["visual_mask"]].numpy(), atol=1e-6)


class MemoryReader(io.BytesIO):
    def __init__(self, data):
        super().__init__(data)
        self.size = len(data)

    def read(self, size):
        return bytearray(super().read(size))


def test_gemma_range_tensor_orientation_and_header_validation():
    from safetensors.torch import save
    original = {"block.experts.gate_up_proj": torch.randn(4, 16, 8).bfloat16(),
                "block.experts.down_proj": torch.randn(4, 8, 8).bfloat16(), "norm": torch.ones(8)}
    data = save(original)
    reader = MemoryReader(data)
    header, offset = read_header(reader)
    result = dict(range_weights(reader, header, offset, ["block.experts"]))
    for expert in range(4):
        torch.testing.assert_close(result[f"block.experts.gate_up.{expert}.weight"], original["block.experts.gate_up_proj"][expert])
        torch.testing.assert_close(result[f"block.experts.down.{expert}.weight"], original["block.experts.down_proj"][expert])
    with pytest.raises(ValueError, match="size disagree"):
        read_header(MemoryReader(data[:-1]))


def test_http_reader_bounds_cache_and_rejects_ignored_range():
    payload = bytes(range(100))
    calls = []
    class Response:
        def __init__(self, start, end, status=206):
            self.status_code = status
            self.headers = {"Content-Range": f"bytes {start}-{end}/{len(payload)}"}
            self.raw = io.BytesIO(payload[start:end+1])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def raise_for_status(self): pass
    def get(url, headers, **kwargs):
        start, end = map(int, headers["Range"][6:].split("-"))
        calls.append((start, end))
        return Response(start, end)
    reader = HubRangeReader.__new__(HubRangeReader)
    reader.size, reader.chunk_bytes, reader.position, reader.cache_start, reader.cache = 100, 16, 0, 0, b""
    reader.session, reader.location, reader.headers = SimpleNamespace(get=get), "test", {}
    assert bytes(reader.read(35)) == payload[:35]
    reader.seek(17)
    assert bytes(reader.read(20)) == payload[17:37]
    assert max(end-start+1 for start, end in calls) <= 16
    reader.cache = b""
    reader.session.get = lambda *args, **kwargs: Response(0, 15, status=200)
    with pytest.raises(ValueError, match="refusing"):
        reader.read(1)


@pytest.mark.parametrize("cuda_mode", [False, pytest.param(True, marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required for complete Gemma NF4 loading/dispatch"))])
def test_gemma_full_loading_path_preserves_weights_buffers_and_tied_head(tmp_path, monkeypatch, cuda_mode):
    import accelerate
    import huggingface_hub
    import gemma_loading
    from accelerate.utils import set_module_tensor_to_device
    from safetensors.torch import save
    from transformers import AutoConfig
    model = tiny_gemma()
    state = {k: v.contiguous() for k, v in model.state_dict().items() if k != "lm_head.weight"}
    payload = save(state)
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"weight_map": {key: "shard.safetensors" for key in state}}))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: str(index))
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *a, **kw: model.config)
    monkeypatch.setattr(gemma_loading, "HubRangeReader", lambda *a, **kw: MemoryReader(payload))
    if not cuda_mode:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (100 * 1024**3, 100 * 1024**3))
        monkeypatch.setattr(torch.cuda, "memory_allocated", lambda device: 0)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(accelerate, "dispatch_model", lambda model, **kwargs: model)
        monkeypatch.setattr(gemma_loading, "put_weight", lambda model, name, value, mapping, dtype, quantize:
                            set_module_tensor_to_device(model, name, "cpu", value=value, dtype=dtype))
    loaded, info = gemma_loading.load_gemma(torch.float16 if cuda_mode else torch.float32,
                                          quantize=cuda_mode, scratch=tmp_path / "scratch")
    assert loaded.lm_head.weight is loaded.model.language_model.embed_tokens.weight
    assert not any(p.is_meta for p in loaded.parameters())
    assert info["packed_expert_modules_split"] == 2
    _, batch = fixture(model, 2)
    with torch.inference_mode():
        expected = model(**batch["native"], logits_to_keep=1).logits
        raw = {k: v.to("cuda:0") if cuda_mode else v for k, v in batch["native"].items()}
        actual = loaded(**raw, logits_to_keep=1).logits.float().cpu()
    if cuda_mode:
        assert torch.isfinite(actual).all()
        assert float((actual - expected).abs().mean()) < 0.05
    else:
        torch.testing.assert_close(actual, expected)
    assert [p.name for p in (tmp_path / "scratch").iterdir()] == ["gemma4_loading_memory.jsonl"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Gemma NF4 validation")
def test_gemma_nf4_experts_run_on_cuda():
    import bitsandbytes as bnb
    from loading import put_weight
    model = tiny_gemma()
    source = model.model.language_model.layers[0].experts
    replacement = GemmaSplitExperts(source)
    holder = torch.nn.Module()
    holder.experts = replacement
    for expert in range(4):
        for target, weights in [("gate_up", source.gate_up_proj), ("down", source.down_proj)]:
            # Prefix mimics real decoder weights for the expert quantization rule.
            wrapper = torch.nn.Module()
            wrapper.block = holder
            put_weight(wrapper, f"block.experts.{target}.{expert}.weight", weights[expert].detach(),
                       {"block": 0}, torch.float16, True)
    assert all(isinstance(m, bnb.nn.Linear4bit) for m in replacement.gate_up)
    hidden = torch.randn(5, 16, device="cuda", dtype=torch.float16)
    ids = torch.tensor([[0, 1], [2, 1], [3, 2], [1, 3], [0, 2]], device="cuda")
    weights = torch.rand(5, 2, device="cuda", dtype=torch.float16)
    with torch.no_grad():
        actual = replacement(hidden, ids, weights).float().cpu()
        expected = source(hidden.float().cpu(), ids.cpu(), weights.float().cpu())
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.2)
