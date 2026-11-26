import types
import unittest

import numpy as np
import torch

import analyze
import analyze_fastmmoe
from model_compat import patch_eager_attention
from run_fastmmoe import RouteRecorder
from run_internvl35 import capture_routes as capture_internvl_routes


class FakeGate:
    n_routed_experts = 4


class FakeMoE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = FakeGate()

    def moe_infer(self, x, topk_ids, topk_weight):
        return x

    def forward(self, hidden, images_seq_mask_without_separator=None):
        topk = torch.tensor([[0, 1], [1, 2], [2, 3]], device=hidden.device)
        weights = torch.full((3, 2), 0.5, device=hidden.device)
        output = self.moe_infer(hidden.reshape(-1, 2), topk, weights).reshape_as(hidden)
        return output, images_seq_mask_without_separator, hidden


class FakeLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = FakeMoE()

    def forward(self, hidden, images_seq_mask_without_separator=None):
        return self.mlp(hidden, images_seq_mask_without_separator=images_seq_mask_without_separator)


class FakeAttention(torch.nn.Module):
    def forward(self, hidden, output_attentions=False, **kwargs):
        weights = torch.ones(1, 1, 2, 2) if output_attentions else None
        return hidden, weights, None


class FakeAttentionLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = FakeAttention()

    def forward(self, hidden, sparse=False):
        output, weights, cache, relation = self.self_attn(
            hidden,
            return_logits=sparse,
        )
        return output, weights, cache, relation


class FakeInternVLRouter(torch.nn.Module):
    top_k = 2
    num_experts = 4

    def forward(self, hidden):
        logits = torch.tensor(
            [[4.0, 3.0, 2.0, 1.0], [1.0, 2.0, 4.0, 3.0], [3.0, 1.0, 4.0, 2.0]],
            device=hidden.device,
        )
        values, indices = logits.softmax(-1).topk(self.top_k, dim=-1)
        return logits, values, indices


class FakeInternVLExperts(torch.nn.Module):
    num_experts = 4


class FakeInternVLBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = FakeInternVLRouter()
        self.experts = FakeInternVLExperts()

    def forward(self, hidden):
        self.gate(hidden.reshape(-1, hidden.shape[-1]))
        return hidden


class FakeInternVLDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.block = FakeInternVLBlock()

    def forward(self, inputs_embeds, **kwargs):
        return self.block(inputs_embeds)


class FakeInternVLLanguageModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInternVLDecoder()


class MetricTests(unittest.TestCase):
    def test_preprune_aggregate(self):
        rows = [
            {
                "condition_correct": 1,
                "accuracy_delta": 0,
                "actual_retention": 0.5,
                "racr_example": 0.25,
                "top1_flip": 0.5,
                "exact_topk_change": 0.75,
                "units": 4,
                "racr_sum": 1.0,
                "baseline_counts": np.array([2, 2]),
                "condition_counts": np.array([3, 1]),
            },
            {
                "condition_correct": 0,
                "accuracy_delta": -1,
                "actual_retention": 0.5,
                "racr_example": 0.5,
                "top1_flip": 0.5,
                "exact_topk_change": 1.0,
                "units": 4,
                "racr_sum": 2.0,
                "baseline_counts": np.array([2, 2]),
                "condition_counts": np.array([2, 2]),
            },
        ]
        result = analyze.aggregate(rows)
        self.assertEqual(result["examples"], 2)
        self.assertAlmostEqual(result["accuracy_delta"], -0.5)
        self.assertAlmostEqual(result["racr_example"], 0.375)

    def test_distribution_distances(self):
        first = np.array([4, 0])
        second = np.array([0, 4])
        self.assertAlmostEqual(analyze_fastmmoe.total_variation(first, first), 0.0)
        self.assertAlmostEqual(analyze_fastmmoe.total_variation(first, second), 1.0)
        self.assertAlmostEqual(analyze_fastmmoe.jensen_shannon(first, first), 0.0)

    def test_fastmmoe_route_recorder(self):
        layer = FakeLayer()
        model = types.SimpleNamespace(
            language=types.SimpleNamespace(
                model=types.SimpleNamespace(layers=torch.nn.ModuleList([layer]))
            )
        )
        recorder = RouteRecorder(model)
        recorder.start()
        hidden = torch.ones(1, 3, 2)
        mask = torch.tensor([[True, True, False]])
        layer(hidden, images_seq_mask_without_separator=mask)
        recorder.stop()
        arrays = recorder.arrays()
        self.assertEqual(arrays["visual_tokens"].tolist(), [2])
        self.assertEqual(int(arrays["expert_counts"].sum()), 4)

    def test_eager_attention_compatibility(self):
        layer = FakeAttentionLayer()
        model = types.SimpleNamespace(
            language=types.SimpleNamespace(
                model=types.SimpleNamespace(layers=torch.nn.ModuleList([layer]))
            )
        )
        patch_eager_attention(model)
        hidden = torch.ones(1, 2, 2)
        regular = layer(hidden)
        sparse = layer(hidden, sparse=True)
        self.assertEqual(len(regular), 4)
        self.assertIsNone(regular[3])
        self.assertEqual(len(sparse), 4)
        self.assertEqual(tuple(sparse[3].shape), (1, 1, 2, 2))

        patch_eager_attention(model)
        self.assertEqual(len(layer.self_attn(hidden)), 4)

    def test_internvl_router_capture_with_structured_gate_output(self):
        model = types.SimpleNamespace(language_model=FakeInternVLLanguageModel())
        condition = {
            "embeddings": torch.ones(1, 3, 2),
            "attention_mask": torch.ones(1, 3, dtype=torch.long),
            "visual_mask": torch.tensor([True, False, True]),
        }
        routes, layer_names, n_experts = capture_internvl_routes(model, condition)
        self.assertEqual(routes.shape, (1, 2, 2))
        self.assertEqual(layer_names.tolist(), ["model.block"])
        self.assertEqual(n_experts, 4)
        self.assertEqual(routes[0].tolist(), [[0, 1], [2, 0]])


if __name__ == "__main__":
    unittest.main()
