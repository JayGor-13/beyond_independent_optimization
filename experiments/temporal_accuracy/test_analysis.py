import unittest

import numpy as np
import pandas as pd

from analyze import add_baseline_deltas, association_table


class PairedAnalysisTests(unittest.TestCase):
    def synthetic_runs(self):
        rows = []
        conditions = [
            ("chronological", None, 0.70, 2.0, True),
            ("mild", 0, 0.65, 1.5, True),
            ("mild", 1, 0.63, 1.3, True),
            ("medium", 0, 0.55, 0.5, True),
            ("medium", 1, 0.53, 0.2, True),
            ("severe", 0, 0.40, -1.0, False),
            ("severe", 1, 0.38, -1.2, False),
        ]
        for item_index in range(8):
            for condition, replicate, trc8, margin, correct in conditions:
                rows.append(
                    {
                        "id": f"q{item_index}",
                        "category": "temporal" if item_index < 6 else "descriptive",
                        "condition": condition,
                        "replicate": replicate,
                        "trc8": trc8 + item_index * 0.001,
                        "weighted_trc": trc8 + 0.1 + item_index * 0.001,
                        "correct_option_margin": margin + item_index * 0.01,
                        "correct": correct,
                    }
                )
        return pd.DataFrame(rows)

    def test_baseline_deltas_are_paired_by_question(self):
        with_deltas = add_baseline_deltas(self.synthetic_runs())
        baseline = with_deltas[with_deltas["condition"] == "chronological"]
        self.assertTrue(np.allclose(baseline["delta_trc8"], 0))
        self.assertTrue(np.allclose(baseline["delta_margin"], 0))

    def test_positive_synthetic_association_is_recovered(self):
        with_deltas = add_baseline_deltas(self.synthetic_runs())
        table = association_table(
            with_deltas,
            samples=50,
            randomizations=50,
            rng=np.random.default_rng(7),
        )
        primary = table[
            (table["category"] == "temporal")
            & (table["routing_metric"] == "trc8")
            & (table["performance_outcome"] == "margin")
        ].iloc[0]
        self.assertGreater(primary["pearson_r"], 0.95)
        self.assertGreater(primary["pearson_ci95_low"], 0)


if __name__ == "__main__":
    unittest.main()
