import unittest

import numpy as np

from core import adjacency_retention, generate_permutation_bank, layer_metrics, normalized_kendall_distance


class PermutationTests(unittest.TestCase):
    def test_eight_frame_bank_has_expected_structure(self):
        bank = generate_permutation_bank(8, seed=1234, repetitions=2)
        self.assertEqual(len(bank), 8)
        self.assertEqual(bank[0]["condition"], "chronological")
        self.assertEqual(bank[-1]["condition"], "reverse")
        self.assertEqual(len({tuple(record["order"]) for record in bank}), len(bank))
        by_condition = {name: [record for record in bank if record["condition"] == name] for name in {
            "chronological", "mild", "medium", "severe", "reverse"
        }}
        self.assertEqual(len(by_condition["mild"]), 2)
        self.assertEqual(len(by_condition["medium"]), 2)
        self.assertEqual(len(by_condition["severe"]), 2)
        self.assertLess(max(row["kendall_distance"] for row in by_condition["mild"]), 0.26)
        self.assertGreater(min(row["kendall_distance"] for row in by_condition["severe"]), 0.59)
        mild_retention = np.mean([row["adjacency_retention"] for row in by_condition["mild"]])
        medium_retention = np.mean([row["adjacency_retention"] for row in by_condition["medium"]])
        severe_retention = np.mean([row["adjacency_retention"] for row in by_condition["severe"]])
        self.assertGreater(mild_retention, medium_retention)
        self.assertGreater(medium_retention, severe_retention)

    def test_geometry_endpoints(self):
        self.assertEqual(normalized_kendall_distance([0, 1, 2, 3]), 0.0)
        self.assertEqual(normalized_kendall_distance([3, 2, 1, 0]), 1.0)
        self.assertEqual(adjacency_retention([0, 1, 2, 3]), 1.0)
        self.assertEqual(adjacency_retention([3, 2, 1, 0]), 1.0)


class MetricTests(unittest.TestCase):
    def test_identical_adjacent_routing_has_unit_trc(self):
        mass = np.zeros((3, 2, 20), dtype=np.float32)
        mass[:, :, :8] = 1 / 8
        counts = (mass > 0).astype(np.int32)
        rows = layer_metrics(mass, counts)
        for row in rows:
            self.assertAlmostEqual(row["trc8"], 1.0)
            self.assertAlmostEqual(row["weighted_trc"], 1.0)

    def test_disjoint_routing_has_zero_weighted_trc(self):
        mass = np.zeros((2, 1, 16), dtype=np.float32)
        mass[0, 0, :8] = 1 / 8
        mass[1, 0, 8:] = 1 / 8
        counts = (mass > 0).astype(np.int32)
        row = layer_metrics(mass, counts)[0]
        self.assertAlmostEqual(row["trc8"], 0.0)
        self.assertAlmostEqual(row["weighted_trc"], 0.0)


if __name__ == "__main__":
    unittest.main()
