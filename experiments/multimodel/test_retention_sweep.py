"""CPU checks for the InternVL retention sweep and Kaggle notebook commands."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from build_package import notebook
from core import compress
from run import INTERNVL_RETENTIONS, parse_args


@pytest.mark.parametrize("model, extra, expected", [
    ("internvl35", [], INTERNVL_RETENTIONS),
    ("internvl35", ["--retentions", "0.5"], [0.5]),
    ("qwen3vl", [], [0.5]),
    ("internvl35", ["--retentions", "1.0"], [1.0]),
])
def test_retention_defaults_and_overrides(monkeypatch, model, extra, expected):
    monkeypatch.setattr(sys, "argv", ["run.py", "--model", model, *extra])
    assert parse_args().retentions == expected


@pytest.mark.parametrize("values", [["0"], ["-0.1"], ["1.01"], ["nan"], ["0.5", "0.5"]])
def test_invalid_retentions_are_rejected(monkeypatch, values):
    monkeypatch.setattr(sys, "argv", ["run.py", "--model", "internvl35", "--retentions", *values])
    with pytest.raises(SystemExit):
        parse_args()


@pytest.mark.parametrize("method", ["l2", "random"])
@pytest.mark.parametrize("retention, expected_count", [(1.0, 256), (0.25, 64), (0.5, 128), (0.75, 192), (0.70, 179)])
def test_sweep_keeps_expected_tokens_and_text(method, retention, expected_count):
    mask = torch.tensor([False, *([True] * 256), False])
    batch = {"embeddings": torch.arange(258 * 4).reshape(1, 258, 4).float(),
             "input_ids": torch.arange(258)[None], "attention_mask": torch.ones(1, 258),
             "visual_mask": mask, "frame_ids": torch.where(mask, 0, -1),
             "position_ids": torch.arange(258)[None],
             "deepstack": [torch.arange(256 * 4).reshape(256, 4).float()]}
    result = compress(batch, method, retention, 42)
    assert int(result["visual_mask"].sum()) == expected_count
    assert result["input_ids"][0, [0, -1]].tolist() == [0, 257]
    assert result["visual_ids"].tolist() == sorted(result["visual_ids"].tolist())
    if retention == 1:
        for key in ("embeddings", "input_ids", "attention_mask", "visual_mask", "frame_ids", "position_ids"):
            torch.testing.assert_close(result[key], batch[key])
        torch.testing.assert_close(result["deepstack"][0], batch["deepstack"][0])
        assert result["visual_ids"].tolist() == list(range(256))
    if method == "random":
        torch.testing.assert_close(result["visual_ids"], compress(batch, method, retention, 42)["visual_ids"])


def test_generated_notebook_matches_saved_and_runs_all_retentions(tmp_path):
    saved = json.loads(Path(__file__).with_name("InternVL35_Kaggle.ipynb").read_text())
    generated = notebook("internvl35", "InternVL3.5-30B-A3B")
    for index, entry in enumerate(generated["cells"]):
        entry["id"] = f"internvl35-{index:02d}"
    assert saved == generated
    commands = []
    state = {"CODE": tmp_path, "sys": sys, "subprocess": SimpleNamespace(run=lambda cmd, **kw: commands.append(cmd))}
    # Execute configuration assignments without setup or any model downloads.
    setup = "".join(saved["cells"][1]["source"])
    config = setup[setup.index("MODEL ="):setup.index("assert 'INTERNVL_RETENTIONS'")]
    exec(config, {"Path": Path}, state)
    for index in (3, 6):
        exec("".join(saved["cells"][index]["source"]), state)
    assert len(commands) == 4
    for command in commands:
        assert command[command.index("--output-root") + 1].endswith("internvl35_retention_sweep_results")
        if Path(command[1]).name == "run.py":
            start = command.index("--retentions") + 1
            stop = command.index("--random-seeds")
            assert list(map(float, command[start:stop])) == INTERNVL_RETENTIONS
            assert command[stop + 1] == "0"
