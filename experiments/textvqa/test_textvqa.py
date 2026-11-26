import unittest

import numpy as np

import analyze
from textvqa_metrics import extract_answer, normalize_answer, vqa_soft_score


class TextVQAMetricTests(unittest.TestCase):
    def test_answer_cleanup(self):
        self.assertEqual(extract_answer("Answer: New York\nExplanation"), "New York")
        self.assertEqual(normalize_answer("The two cats."), "2 cats")

    def test_vqa_consensus_scores(self):
        for matches, expected in [(1, 0.3), (2, 0.6), (3, 0.9), (4, 1.0), (5, 1.0), (10, 1.0)]:
            answers = ["cat"] * matches + ["dog"] * (10 - matches)
            self.assertAlmostEqual(vqa_soft_score("cat", answers), expected)

    def test_textvqa_aggregate(self):
        rows = [
            {
                "condition_score": 0.9, "accuracy_delta": -0.1,
                "actual_retention": 0.5, "racr_example": 0.2,
                "top1_flip": 0.3, "exact_topk_change": 0.4,
                "units": 10, "racr_sum": 2.0,
                "baseline_counts": np.array([5, 5]),
                "condition_counts": np.array([6, 4]),
            },
            {
                "condition_score": 0.3, "accuracy_delta": 0.3,
                "actual_retention": 0.5, "racr_example": 0.4,
                "top1_flip": 0.5, "exact_topk_change": 0.6,
                "units": 10, "racr_sum": 4.0,
                "baseline_counts": np.array([5, 5]),
                "condition_counts": np.array([5, 5]),
            },
        ]
        result = analyze.aggregate(rows)
        self.assertAlmostEqual(result["accuracy"], 0.6)
        self.assertAlmostEqual(result["accuracy_delta"], 0.1)
        self.assertAlmostEqual(result["racr_example"], 0.3)


if __name__ == "__main__":
    unittest.main()
