import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core import CONDITIONS, write_json
from run import experiment1, experiment2


class FixtureAdapter:
    n_experts, top_k = 4, 2

    def prepare(self, images, text, temporal=False):
        frame_ids = torch.tensor([v for i in range(len(images)) for v in (-1, i, i)])
        mask = frame_ids >= 0
        return {"native": {}, "embeddings": torch.randn(1, len(mask), 4),
                "visual_mask": mask, "frame_ids": frame_ids,
                "attention_mask": torch.ones(1, len(mask)), "input_ids": torch.zeros(1, len(mask), dtype=torch.long)}

    def check_native(self, batch):
        return {"next_token_matches": True, "max_absolute_logit_error": 0}

    def evaluate(self, batch, max_new_tokens):
        count = int(batch["visual_mask"].sum())
        routes = np.tile([0, 1], (2, count, 1))
        weights = np.tile([0.75, 0.25], (2, count, 1))
        return routes, weights, np.asarray(["layer0", "layer1"]), "A"


@pytest.mark.parametrize("retentions", [[0.5], [1.0, 0.25, 0.50, 0.75, 0.70]])
def test_runner_outputs_feed_both_complete_analyses(tmp_path, monkeypatch, retentions):
    import run
    import huggingface_hub
    args = SimpleNamespace(model="qwen3vl", retentions=retentions, random_seeds=[0],
                           frames=2, max_new_tokens=2, scratch=tmp_path / "scratch")
    config = {"limit": 2, "retentions": retentions, "random_seeds": [0], "frames": 2,
              "model": "fixture", "model_revision": "fixture", "precision": "float16",
              "protocol": "fixture", "video_revision": "fixture", "seed": 42}
    dataset = [{"image": object(), "question": "q", "choices": ["a", "b"], "answer": i} for i in range(2)]
    manifest = [{"id": f"example{i}", "dataset_index": i, "video_id": str(i), "qid": i,
                 "type": "TN", "category": "temporal", "question": "q", "choices": ["a", "b"], "answer": i}
                for i in range(2)]
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **kw: "fixture.mp4")
    monkeypatch.setattr(run, "sample_frames", lambda path, count: ([object()] * count, np.arange(count)))
    adapter = FixtureAdapter()
    for experiment, function in [(1, experiment1), (2, experiment2)]:
        root = tmp_path / "qwen3vl" / f"experiment_{experiment}_n2"
        write_json(root / "run_config.json", {**config, "experiment": experiment})
        write_json(root / "manifests" / "examples.json", manifest)
        function(args, adapter, (root, config, dataset, manifest, {"0": "0.mp4", "1": "1.mp4"}))
        # Resume must not evaluate completed records again.
        old = adapter.evaluate
        adapter.evaluate = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Resume recomputed output"))
        function(args, adapter, (root, config, dataset, manifest, {"0": "0.mp4", "1": "1.mp4"}))
        adapter.evaluate = old
    subprocess.run([sys.executable, str(Path(__file__).with_name("analyze.py")), "--model", "qwen3vl",
                    "--output-root", str(tmp_path), "--limit", "2", "--bootstrap", "10", "--permutations", "2"], check=True)
    import pandas as pd
    summary = pd.read_csv(tmp_path / "qwen3vl/experiment_1_n2/summaries/summary.csv")
    expected = {"baseline"}
    for retention in retentions:
        expected.update({f"l2_r{round(retention * 100)}", f"random_r{round(retention * 100)}_s0"})
    assert set(summary["condition"]) == expected and (summary["model"] == "fixture").all()
    if 1.0 in retentions:
        full = summary[summary["condition"].isin(["l2_r100", "random_r100_s0"])]
        assert (full["actual_retention"] == 1.0).all()
        assert (full["accuracy_delta"] == 0.0).all()
        assert (full["top1_flip"] == 0.0).all()
    accuracy = pd.read_csv(tmp_path / "qwen3vl/experiment_2_n2/summaries/condition_accuracy.csv")
    assert set(accuracy["condition"]) == set(CONDITIONS)
    assert (accuracy["accuracy"] == 0.5).all()
