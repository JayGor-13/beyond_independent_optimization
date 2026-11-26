import json
import tempfile
import unittest
from pathlib import Path

from core import CONDITIONS, donor_items, ensure_config, summarize, validate_manifest


def sample():
    manifest = [dict(id=f"nextqa_{i:05d}", dataset_index=i, video_id=str(i), type="TC",
                     category="temporal", choices=list("ABCDE"), answer=0) for i in range(256)]
    reference = [dict(id=m["id"], video_id=m["video_id"], question_type="TC",
                      category="temporal", correct="True") for m in manifest]
    return manifest, reference


class Checks(unittest.TestCase):
    def test_exact_sample_required(self):
        manifest, reference = sample()
        validate_manifest(manifest, reference)
        manifest[-1]["video_id"] = "different"
        with self.assertRaises(ValueError):
            validate_manifest(manifest, reference)

    def test_pilot_uses_full_donor_ring(self):
        manifest, _ = sample()
        self.assertEqual([m["dataset_index"] for m in donor_items(manifest, 31)], list(range(32, 40)))
        self.assertEqual([m["dataset_index"] for m in donor_items(manifest, 255)], list(range(8)))

    def test_configuration_change_blocks_resume(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            path = Path(directory)/"config.json"
            ensure_config(path, {"model": "first"})
            ensure_config(path, {"model": "first"})
            with self.assertRaises(ValueError):
                ensure_config(path, {"model": "second"})

    def test_pilot_not_paper_ready(self):
        manifest, reference = sample()
        expected = validate_manifest(manifest, reference)
        records = [dict(id=m["id"], condition=c, predicted="A", correct=True)
                   for m in manifest[:32] for c in CONDITIONS]
        summary, checks = summarize(records, expected)
        self.assertEqual(summary[0]["accuracy_percent"], 100)
        self.assertFalse(checks["eligible_for_paper_review"])

    def test_full_run_detects_baseline_disagreement(self):
        manifest, reference = sample()
        expected = validate_manifest(manifest, reference)
        records = [dict(id=m["id"], condition=c, predicted="A", correct=True)
                   for m in manifest for c in CONDITIONS]
        self.assertTrue(summarize(records, expected)[1]["eligible_for_paper_review"])
        records[0]["correct"] = False
        checks = summarize(records, expected)[1]
        self.assertFalse(checks["eligible_for_paper_review"])
        self.assertEqual(checks["chronological_correctness_mismatches"], [manifest[0]["id"]])


if __name__ == "__main__":
    unittest.main()
