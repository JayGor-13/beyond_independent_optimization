"""CPU checks with real tiny Qwen3.5 models; no model/dataset downloads."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from data import normalize_row, prompt_content
from protocol import conditions, example_seed, lock_json, score_answer, write_json
from routing import paired_bootstrap, routing_metrics, select_visual_ids


def tiny_model():
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration
    config = Qwen3_5MoeConfig(
        image_token_id=90, vision_start_token_id=91, vision_end_token_id=92, video_token_id=93,
        text_config={"vocab_size": 128, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "linear_num_key_heads": 2, "linear_num_value_heads": 4,
                     "linear_key_head_dim": 8, "linear_value_head_dim": 8,
                     "moe_intermediate_size": 16, "shared_expert_intermediate_size": 16,
                     "num_experts": 4, "num_experts_per_tok": 2, "eos_token_id": 127,
                     "layer_types": ["linear_attention", "full_attention"],
                     "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0,
                                         "partial_rotary_factor": 1.0, "mrope_section": [1, 1, 2]}},
        vision_config={"depth": 1, "hidden_size": 32, "num_heads": 4, "intermediate_size": 64,
                       "out_hidden_size": 32, "patch_size": 2, "temporal_patch_size": 2,
                       "spatial_merge_size": 2, "num_position_embeddings": 16})
    config._attn_implementation = "sdpa"
    config._experts_implementation = "eager"
    torch.manual_seed(17)
    return Qwen3_5MoeForConditionalGeneration(config).eval()


class TinyProcessor:
    """Real visual forward with fixed synthetic pixels; no tokenizer downloads."""
    tokenizer = SimpleNamespace(eos_token_id=127, decode=lambda tokens, **kwargs: "A")

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is False
        return "synthetic prompt"

    def __call__(self, **kwargs):
        from transformers.feature_extraction_utils import BatchFeature
        generator = torch.Generator().manual_seed(29)
        return BatchFeature({"input_ids": torch.tensor([[1, 91, 90, 90, 90, 90, 92, 4, 5]]),
                             "attention_mask": torch.ones(1, 9, dtype=torch.long),
                             "mm_token_type_ids": torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0, 0]]),
                             "image_grid_thw": torch.tensor([[1, 4, 4]]),
                             "pixel_values": torch.randn(16, 24, generator=generator)})


def science_item():
    return normalize_row("scienceqa", {"question": "Which option?", "choices": ["yes", "no"],
                                       "answer": 0, "image": Image.new("RGB", (8, 8))}, 7)


def test_token_identity_alignment_and_set_metrics():
    baseline = np.array([[[0, 1], [1, 2], [2, 3], [0, 2]]])
    compressed = np.array([[[2, 1], [0, 2]]])
    metrics = routing_metrics(baseline, compressed, np.array([1, 3]), 4)
    assert metrics["racr"] == 0
    assert metrics["top1_change"] == 0.5
    assert metrics["exact_topk_change"] == 0
    wrong = compressed.copy()
    wrong[0, 0] = [0, 3]
    assert routing_metrics(baseline, wrong, np.array([1, 3]), 4)["racr"] == 0.5
    with pytest.raises(ValueError):
        routing_metrics(baseline, compressed, np.array([3, 1]), 4)


def test_per_image_budgets_stable_ties_and_random_reproducibility():
    norms, images = np.ones(8), np.repeat([0, 1], 4)
    np.testing.assert_array_equal(select_visual_ids(norms, images, "l2", 0.5, 9), [0, 1, 4, 5])
    first = select_visual_ids(norms, images, "random", 0.5, example_seed("a", 2))
    np.testing.assert_array_equal(first, select_visual_ids(norms, images, "random", 0.5, example_seed("a", 2)))
    assert len(set(images[first])) == 2
    np.testing.assert_array_equal(select_visual_ids(norms, images, "random", 1, 19), np.arange(8))
    with pytest.raises(ValueError):
        select_visual_ids([np.nan], [0], "l2", 0.5, 1)


def test_ten_option_scoring_does_not_extract_letters_from_reasoning():
    item = {"choices": [str(i) for i in range(10)], "target_index": 9}
    assert score_answer("(J)", item)["correct"]
    assert not score_answer("A seems wrong; the answer might be J", item)["valid_answer"]
    assert not score_answer("<think>J", item)["valid_answer"]
    assert score_answer("<think>analysis</think>Answer: J", item)["correct"]


def test_mathvista_numeric_precision_and_invalid_responses():
    item = {"choices": [], "target": "1.23", "answer_type": "float", "precision": 2}
    assert score_answer("1.234", item)["correct"]
    assert not score_answer("The chart shows 1.234 units", item)["valid_answer"]
    assert not score_answer("nan", item)["valid_answer"]
    integer = {**item, "target": "4", "answer_type": "integer"}
    assert score_answer("4.0", integer)["correct"]


def test_mmmu_images_are_interleaved_and_gold_text_never_enters_prompt():
    row = {"id": "source", "question": "Compare <image 2> with <image 1>.",
           "options": repr(["yes", "no"]), "answer": "B", "explanation": "SECRET_GOLD_EXPLANATION",
           "image_1": Image.new("RGB", (8, 8), "red"), "image_2": Image.new("RGB", (8, 8), "blue")}
    item, images = normalize_row("mmmu_pro", row, 2)
    content = prompt_content(item, images)
    ordered = [part["image"] for part in content if part["type"] == "image"]
    assert ordered[0].getpixel((0, 0)) == (0, 0, 255)
    assert ordered[1].getpixel((0, 0)) == (255, 0, 0)
    assert "SECRET_GOLD_EXPLANATION" not in str(content)


def test_resume_rejects_configuration_changes(tmp_path):
    lock_json(tmp_path / "config.json", {"precision": "bf16"})
    with pytest.raises(ValueError):
        lock_json(tmp_path / "config.json", {"precision": "nf4"})
    assert len(conditions([0.75, 0.5, 0.25], [0, 1, 2])) == 15
    with pytest.raises(ValueError):
        conditions([0.500001, 0.500002], [0])


def test_paired_bootstrap_uses_question_differences():
    assert paired_bootstrap(np.zeros(20), 100) == (0, 0, 0)
    mean, low, high = paired_bootstrap([1, 1, 0, 0], 500)
    assert mean == 0.5 and low <= mean <= high


def test_real_tiny_native_equivalence_pruning_and_cache_generation():
    from adapter import Adapter, prune_batch
    adapter = Adapter(tiny_model(), TinyProcessor())
    item, images = science_item()
    batch = adapter.prepare(item, images)
    assert adapter.check_native(batch)["all_visual_routes_match"]
    baseline = adapter.evaluate(batch, 3)
    assert baseline["routes"].shape == (2, 4, 2)
    with torch.inference_mode():
        native_generated = adapter.model.generate(**batch["native"], max_new_tokens=3, do_sample=False,
                                                  pad_token_id=127)
    assert native_generated[0, batch["native"]["input_ids"].shape[1]:].tolist() == baseline["generated_token_ids"]
    identity = prune_batch(batch, conditions([0.5], [0])[1], item["id"])
    np.testing.assert_array_equal(adapter.evaluate(identity, 3)["routes"], baseline["routes"])
    compressed = prune_batch(batch, conditions([0.5], [0])[3], item["id"])
    assert int(compressed["visual_mask"].sum()) == 2
    # Original native coordinates survive pruning; they are not regenerated from shorter length.
    original_visual = batch["position_ids"][..., batch["visual_mask"]]
    actual_visual = compressed["position_ids"][..., compressed["visual_mask"]]
    torch.testing.assert_close(actual_visual, original_visual[..., compressed["visual_ids"]])
    assert adapter.evaluate(compressed, 3)["routes"].shape == (2, 2, 2)


def test_split_experts_match_official_packed_experts():
    from loading import split_experts
    original = tiny_model()
    modified = copy.deepcopy(original)
    state = original.state_dict()
    names = split_experts(modified)
    for name, module in modified.named_modules():
        if name not in names:
            continue
        for expert in range(module.num_experts):
            module.gate_up[expert].weight = torch.nn.Parameter(state[name + ".gate_up_proj"][expert].clone())
            module.down[expert].weight = torch.nn.Parameter(state[name + ".down_proj"][expert].clone())
    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3, 4]])
        before = original(input_ids=ids, logits_to_keep=1).logits
        after = modified(input_ids=ids, logits_to_keep=1).logits
    torch.testing.assert_close(before, after, atol=1e-6, rtol=1e-5)


def test_complete_synthetic_experiment_analysis_and_incomplete_rejection(tmp_path):
    from adapter import Adapter
    from analyze import read_complete_run, summarize_run
    from run import run_benchmark
    matrix = conditions([0.5], [0, 1])
    rows = [{"question": "Which option?", "choices": ["yes", "no"], "answer": answer,
             "image": Image.new("RGB", (8, 8))} for answer in [0, 1]]
    manifest = {"indices": [0, 1], "benchmark": "scienceqa"}
    config = {"schema": 1, "examples": 2, "benchmark": "scienceqa", "conditions": matrix,
              "route_storage": "full", "repo": "synthetic-only", "revision": "0" * 40,
              "precision": "float32_test_only", "phase": "smoke"}
    args = SimpleNamespace(model="synthetic", route_storage="full", conditions=matrix, smoke=True, max_new_tokens=2)
    adapter = Adapter(tiny_model(), TinyProcessor())
    write_json(tmp_path / "run_config.json", config)
    write_json(tmp_path / "manifest.json", manifest)
    write_json(tmp_path / "loading_info.json", {"n_experts": 4})
    run_benchmark(args, adapter, tmp_path, rows, manifest, config)
    summary = summarize_run(tmp_path, bootstrap_samples=100, verify_routes=True)
    random_mean = next(row for row in summary if row["method"] == "random_seed_mean")
    assert random_mean["examples"] == 2
    assert (tmp_path / "summaries" / "summary.csv").exists()
    # A missing condition cannot silently shrink the paired evaluation set.
    (tmp_path / "conditions" / matrix[-1]["name"] / "scienceqa_000001.json").unlink()
    with pytest.raises(ValueError, match="Incomplete"):
        read_complete_run(tmp_path)


def checkpoint_fixture(tmp_path, monkeypatch):
    import shutil
    import huggingface_hub
    from safetensors.torch import save_file
    from transformers import AutoConfig
    reference = tiny_model()
    state = reference.state_dict()
    source = tmp_path / "checkpoint"
    source.mkdir()
    # Use two shards, ensuring the loader can release each temporary download.
    keys, weight_map = sorted(state), {}
    for number, subset in enumerate((keys[::2], keys[1::2])):
        filename = f"shard{number}.safetensors"
        save_file({key: state[key].contiguous() for key in subset}, source / filename)
        weight_map.update({key: filename for key in subset})
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))

    def download(repo, filename, revision=None, local_dir=None):
        if local_dir:
            destination = type(source)(local_dir) / filename
            shutil.copyfile(source / filename, destination)
            return str(destination)
        return str(source / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: copy.deepcopy(reference.config))
    return reference


@pytest.mark.parametrize("precision", ["bf16", "expert-nf4"])
def test_meta_shard_loader_without_network_or_cuda(tmp_path, monkeypatch, precision):
    import accelerate
    import loading
    from accelerate.utils import set_module_tensor_to_device
    reference = checkpoint_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda device: (100 * 1024**3, 100 * 1024**3))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "CPU test double")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda device: SimpleNamespace(total_memory=100 * 1024**3))
    monkeypatch.setattr(accelerate, "dispatch_model", lambda model, **kwargs: model)
    monkeypatch.setattr(loading, "put_weight", lambda model, name, value, mapping, quantize:
                        set_module_tensor_to_device(model, name, "cpu", value=value, dtype=torch.bfloat16))
    model, info = loading.load_model("synthetic", "0" * 40, precision, tmp_path / "scratch")
    assert not any(parameter.is_meta for parameter in model.parameters())
    assert not list((tmp_path / "scratch").iterdir())
    assert info["quantized_expert_collections"] == (2 if precision == "expert-nf4" else 0)
    reference.to(torch.bfloat16)
    with torch.inference_mode():
        ids = torch.tensor([[1, 2, 3]])
        actual = model(input_ids=ids, logits_to_keep=1).logits
        expected = reference(input_ids=ids, logits_to_keep=1).logits
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA/NF4 loading requires the user's GPU")
def test_cuda_nf4_shard_loader_and_router_instrumentation(tmp_path, monkeypatch):
    import bitsandbytes as bnb
    from adapter import Adapter
    from loading import load_model
    checkpoint_fixture(tmp_path, monkeypatch)
    model, info = load_model("synthetic", "0" * 40, "expert-nf4", tmp_path / "scratch", reserve_gib=0.01)
    quantized = [module for module in model.modules() if isinstance(module, bnb.nn.Linear4bit)]
    assert len(quantized) == 16 and all(module.weight.quant_state is not None for module in quantized)
    adapter = Adapter(model, TinyProcessor())
    item, images = science_item()
    batch = adapter.prepare(item, images)
    assert adapter.check_native(batch)["all_visual_routes_match"]
    assert adapter.evaluate(batch, 2)["routes"].shape == (2, 4, 2)
    assert not list((tmp_path / "scratch").iterdir())
