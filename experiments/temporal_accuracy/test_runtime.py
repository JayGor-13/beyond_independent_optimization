import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from run import capture_routes_and_logits, sample_frames, score_answer_choices


class MoEGate(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_routed_experts = 6

    def forward(self, hidden):
        batch, sequence, _ = hidden.shape
        first = torch.arange(sequence, device=hidden.device) % self.n_routed_experts
        second = (first + 1) % self.n_routed_experts
        topk = torch.stack([first, second], dim=-1).repeat(batch, 1, 1).reshape(-1, 2)
        weights = torch.full_like(topk, 0.5, dtype=hidden.dtype)
        return topk, weights, None


class FakeLanguage(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = MoEGate()
        self.embedding = torch.nn.Embedding(20, 4)

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, inputs_embeds, attention_mask, use_cache, return_dict):
        self.gate(inputs_embeds)
        logits = torch.zeros(inputs_embeds.shape[0], inputs_embeds.shape[1], 20)
        logits[:, -1, 1:6] = torch.tensor([1.0, 2.0, 5.0, 3.0, 4.0])
        return types.SimpleNamespace(logits=logits)


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.language = FakeLanguage()


class RuntimeTests(unittest.TestCase):
    def test_single_prefill_captures_routes_and_scores_choices(self):
        model = FakeModel()
        batch = {
            "embeddings": torch.ones(1, 6, 4),
            "attention_mask": torch.ones(1, 6, dtype=torch.long),
            "frame_ids": torch.tensor([0, 0, 0, 1, 1, 1]),
        }
        mass, counts, layer_names, tokens, topk, logits = capture_routes_and_logits(model, batch, 2)
        self.assertEqual(mass.shape, (2, 1, 6))
        self.assertEqual(counts.shape, (2, 1, 6))
        self.assertEqual(layer_names.tolist(), ["language.gate"])
        self.assertEqual(tokens.tolist(), [3, 3])
        self.assertEqual(topk, 2)
        self.assertTrue(np.allclose(mass.sum(axis=-1), 1))

        token_ids = {letter: [index + 1] for index, letter in enumerate("ABCDE")}
        scores, probabilities, predicted = score_answer_choices(model, batch, logits, token_ids)
        self.assertEqual(predicted, "C")
        self.assertGreater(scores["C"], scores["E"])
        self.assertAlmostEqual(sum(probabilities.values()), 1.0)

    def test_missing_target_uses_nearest_unique_decodable_frame(self):
        class FakeCapture:
            def __init__(self, _):
                self.position = 0

            def isOpened(self):
                return True

            def get(self, property_id):
                if property_id == 7:  # cv2.CAP_PROP_FRAME_COUNT
                    return 10
                if property_id == 1:  # cv2.CAP_PROP_POS_FRAMES
                    return self.position
                return 0

            def set(self, property_id, value):
                if property_id == 1:
                    self.position = int(value)
                return True

            def read(self):
                if self.position == 9:
                    return False, None
                frame = np.zeros((4, 4, 3), dtype=np.uint8)
                self.position += 1
                return True, frame

            def release(self):
                pass

        with patch("run.cv2.VideoCapture", FakeCapture), self.assertWarns(RuntimeWarning):
            frames, actual, requested, recoveries = sample_frames("fake.mp4", 2)
        self.assertEqual(len(frames), 2)
        self.assertEqual(requested.tolist(), [0, 9])
        self.assertEqual(actual.tolist(), [0, 8])
        self.assertEqual(recoveries, [{"requested_index": 9, "actual_index": 8, "offset": -1}])


if __name__ == "__main__":
    unittest.main()
