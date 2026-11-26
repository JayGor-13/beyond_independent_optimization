import json
from pathlib import Path

import numpy as np
import pytest
import torch

from core import aggregate_frames, compress, cross_indices, frame_groups, lock_json, nextqa_manifest
from loading import SplitExperts, expanded_weights, unpack_experts


def sample_batch():
    return {"embeddings": torch.arange(24).reshape(1, 6, 4).float(),
            "input_ids": torch.arange(6)[None], "attention_mask": torch.ones(1, 6),
            "visual_mask": torch.tensor([False, True, True, False, True, True]),
            "frame_ids": torch.tensor([-1, 0, 0, -1, 1, 1]),
            "position_ids": torch.arange(18).reshape(3, 1, 6),
            "deepstack": [torch.arange(16).reshape(4, 4).float()]}


def test_pruning_preserves_text_identity_positions_and_deepstack():
    batch = sample_batch()
    result = compress(batch, "l2", 0.5, 0)
    assert result["visual_ids"].tolist() == [2, 3]
    assert result["input_ids"].tolist() == [[0, 3, 4, 5]]
    torch.testing.assert_close(result["deepstack"][0], batch["deepstack"][0][[2, 3]])
    torch.testing.assert_close(result["position_ids"], batch["position_ids"][:, :, [0, 3, 4, 5]])
    assert batch["embeddings"].shape[1] == 6


def test_baseline_is_identity_and_random_is_reproducible():
    batch = sample_batch()
    result = compress(batch, "baseline", 1, 0)
    torch.testing.assert_close(result["embeddings"], batch["embeddings"])
    assert result["visual_ids"].tolist() == [0, 1, 2, 3]
    first = compress(batch, "random", 0.5, 72)
    second = compress(batch, "random", 0.5, 72)
    torch.testing.assert_close(first["visual_ids"], second["visual_ids"])


def test_frame_mapping_and_empty_rejection():
    batch = sample_batch()
    torch.testing.assert_close(frame_groups(batch["visual_mask"], 2), batch["frame_ids"])
    with pytest.raises(ValueError):
        frame_groups(batch["visual_mask"], 3)
    with pytest.raises(ValueError):
        frame_groups(torch.zeros(4, dtype=torch.bool), 1)


def test_cross_clip_never_uses_target_even_for_smoke():
    for size in (2, 3, 8, 256):
        for target in range(size):
            donors = cross_indices(target, size, 8)
            assert len(donors) == 8 and all(0 <= d < size and d != target for d in donors)
    with pytest.raises(ValueError):
        cross_indices(0, 1, 8)


def test_frame_mass_and_counts():
    topk = np.array([[[0, 1], [1, 2], [0, 2]]])
    weights = np.array([[[0.75, 0.25], [0.5, 0.5], [0.4, 0.6]]])
    mass, counts, tokens = aggregate_frames(topk, weights, np.array([0, 0, 1]), 3, 2)
    np.testing.assert_allclose(mass[:, 0], [[0.375, 0.375, 0.25], [0.4, 0, 0.6]])
    np.testing.assert_array_equal(counts[:, 0], [[1, 2, 1], [1, 0, 1]])
    assert tokens.tolist() == [2, 1]
    with pytest.raises(ValueError):
        aggregate_frames(topk, weights, np.zeros(3), 3, 2)


def test_resume_rejects_precision_or_manifest_changes(tmp_path):
    path = tmp_path / "config.json"
    lock_json(path, {"precision": "expert-nf4"})
    lock_json(path, {"precision": "expert-nf4"})
    with pytest.raises(ValueError):
        lock_json(path, {"precision": "float16"})
    assert json.loads(path.read_text())["precision"] == "expert-nf4"


def test_nextqa_unique_sampling_and_exhaustion():
    dataset = [{"video": str(i), "qid": i, "type": ["TN", "CW", "DC"][i % 3],
                "question": "q", "answer": 0, **{f"a{j}": str(j) for j in range(5)}} for i in range(12)]
    manifest = nextqa_manifest(dataset, {str(i) for i in range(12)}, 6, 42)
    assert len({row["video_id"] for row in manifest}) == 6
    assert [row["category"] for row in manifest] == ["temporal", "causal", "descriptive"] * 2
    assert manifest == nextqa_manifest(dataset, {str(i) for i in range(12)}, 6, 42)
    with pytest.raises(ValueError):
        nextqa_manifest(dataset, {"0"}, 2, 42)


def test_packed_expert_conversion_matches_original_math():
    torch.manual_seed(5)
    count, hidden, intermediate = 4, 8, 6
    gate_up = torch.randn(count, hidden, 2 * intermediate)
    down = torch.randn(count, intermediate, hidden)
    module = SplitExperts(count, hidden, intermediate, torch.nn.functional.silu)
    for key, value in expanded_weights("experts.gate_up_proj", gate_up, ["experts"]):
        module.get_parameter(key.removeprefix("experts.")).data.copy_(value)
    for key, value in expanded_weights("experts.down_proj", down, ["experts"]):
        module.get_parameter(key.removeprefix("experts.")).data.copy_(value)
    x = torch.randn(1, 7, hidden)
    probabilities = torch.randn(7, count).softmax(-1)
    weights, indices = probabilities.topk(2, dim=-1)
    weights /= weights.sum(-1, keepdim=True)
    full = torch.zeros_like(probabilities).scatter_(1, indices, weights)
    reference = torch.zeros_like(x)
    for e in range(count):
        gate, up = (x @ gate_up[e]).chunk(2, -1)
        reference += (torch.nn.functional.silu(gate) * up) @ down[e] * full[:, e][None, :, None]
    torch.testing.assert_close(module(x, full, indices), reference)


def test_missing_completion_refuses_analysis(tmp_path):
    from analyze import validate
    (tmp_path / "manifests").mkdir()
    (tmp_path / "run_config.json").write_text(json.dumps({"limit": 1, "experiment": 1}))
    (tmp_path / "manifests" / "examples.json").write_text('[{"id":"a"}]')
    with pytest.raises(ValueError, match="incomplete"):
        validate(tmp_path)
