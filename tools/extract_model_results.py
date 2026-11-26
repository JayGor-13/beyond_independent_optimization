"""Extract full-run notebook evidence; never use screenshot or smoke-test values."""

import argparse
import hashlib
import io
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INTERNVL = "InternVL3.5-30B-A3B"
DEEPSEEK = "DeepSeek-VL2-Tiny"


def extract(notebook, output):
    raw = notebook.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    tables, full_streams, locations = [], [], []
    for index, cell in enumerate(data["cells"]):
        source = "".join(cell.get("source", []))
        # These guards exclude the separate two-example smoke tables.
        if "experiment_{experiment}_n256" in source:
            for number, item in enumerate(cell.get("outputs", [])):
                html = item.get("data", {}).get("text/html")
                if html:
                    frame = pd.read_html(io.StringIO("".join(html)))[0]
                    frame = frame.loc[
                        :, ~frame.columns.astype(str).str.startswith("Unnamed:")
                    ]
                    if "..." in frame:
                        frame = frame.drop(columns="...")
                    tables.append(frame)
                    locations.append({"cell_index": index, "output_index": number})
        if "'--limit', '256'" in source and "run.py" in source:
            if any(
                item.get("output_type") == "error" for item in cell.get("outputs", [])
            ):
                raise ValueError("The full-run cell contains an error.")
            full_streams.append(
                "".join(
                    "".join(item.get("text", [])) for item in cell.get("outputs", [])
                )
            )
    stream = "\n".join(full_streams)
    for experiment in (1, 2):
        if f"Completed experiment {experiment}:" not in stream:
            raise ValueError(f"No completion evidence for experiment {experiment}.")
    exp1 = next(t for t in tables if "target_retention" in t).copy()
    trc = next(
        t
        for t in tables
        if "weighted_trc" in t.get("metric", pd.Series(dtype=str)).values
    ).copy()
    accuracy = next(
        t for t in tables if "accuracy_delta_from_chronological" in t
    ).copy()
    assert (
        set(exp1["examples"]) == set(trc["clips"]) == set(accuracy["examples"]) == {256}
    )
    assert set(exp1["precision"]) == {"expert-nf4"}
    # Pandas display truncates the middle columns; recover example-averaged RACR
    # and Top-1 from the analysis log, at its displayed FOUR-decimal precision.
    rows = re.findall(
        r"^(baseline|l2_r50|random_r50_s0)\t.*?RACR=([\d.]+)\ttop1_flip=([\d.]+)$",
        stream,
        re.M,
    )
    metrics = {condition: (float(racr), float(flip)) for condition, racr, flip in rows}
    if set(metrics) != set(exp1["condition"]):
        raise ValueError("Missing printed full-run routing metrics.")
    exp1["racr_example"] = exp1["condition"].map(lambda name: metrics[name][0])
    exp1["top1_flip"] = exp1["condition"].map(lambda name: metrics[name][1])
    exp1["routing_source_decimals"] = 4
    exp1["model_label"] = trc["model_label"] = accuracy["model_label"] = INTERNVL

    paired_rows, relation_rows = [], []
    for line in stream.splitlines():
        cells = line.split()
        if len(cells) == 7 and cells[0].startswith("chronological_minus_"):
            paired_rows.append(cells)
        if (
            len(cells) == 11
            and cells[0] == "all"
            and cells[1] in ("trc8", "weighted_trc")
        ):
            relation_rows.append(cells)
    if len(paired_rows) != 8 or len(relation_rows) != 2:
        raise ValueError("Incomplete paired-control or accuracy-relation evidence.")
    paired = pd.DataFrame(
        paired_rows,
        columns=[
            "comparison",
            "metric",
            "clips",
            "mean_difference",
            "median_difference",
            "ci95_low",
            "ci95_high",
        ],
    )
    relation = pd.DataFrame(
        relation_rows,
        columns=[
            "layer",
            "metric",
            "examples",
            "correct_examples",
            "wrong_examples",
            "mean_correct",
            "mean_wrong",
            "correct_minus_wrong",
            "point_biserial_r",
            "difference_ci95_low",
            "difference_ci95_high",
        ],
    )
    # Prefer the completed retention sweep over the older notebook's 50%-only table.
    sweep = ROOT / "results/published/internvl/retention_sweep"
    sweep_sources = []
    if (sweep / "summaries/summary.csv").exists():
        config = json.loads((sweep / "run_config.json").read_text(encoding="utf-8"))
        complete = json.loads((sweep / "complete.json").read_text(encoding="utf-8"))
        exp1 = pd.read_csv(sweep / "summaries/summary.csv")
        expected = {"baseline"} | {
            f"{method}_r{round(r * 100)}{suffix}"
            for r in (1.0, 0.25, 0.5, 0.75, 0.7)
            for method, suffix in (("l2", ""), ("random", "_s0"))
        }
        if (
            set(exp1.condition) != expected
            or len(exp1) != len(expected)
            or set(complete["conditions"]) != expected
            or complete["examples"] != 256
            or set(exp1.examples) != {256}
            or set(exp1.precision) != {"expert-nf4"}
            or set(exp1.model) != {config["model"]}
            or set(exp1.model_revision) != {config["model_revision"]}
        ):
            raise ValueError(
                "Incomplete or inconsistent InternVL retention-sweep summary."
            )
        exp1["model_label"] = INTERNVL
        sweep_sources = [
            sweep / name
            for name in (
                "summaries/summary.csv",
                "summaries/per_example.csv",
                "run_config.json",
                "complete.json",
                "native_equivalence.json",
                "manifests/examples.json",
            )
        ]
    deep_exp_path = ROOT / "results/published/deepseek/compression_routing/summary.csv"
    deep_trc_path = ROOT / "results/published/deepseek/temporal_routing/summary.csv"
    deep_exp = pd.read_csv(deep_exp_path)
    deep_exp["model_label"], deep_exp["precision"] = DEEPSEEK, "float16"
    deep_trc = pd.read_csv(deep_trc_path)
    deep_trc["model_label"], deep_trc["precision"] = DEEPSEEK, "float16"
    output.mkdir(parents=True, exist_ok=True)
    for name, table in [
        ("internvl_exp1_summary", exp1),
        ("internvl_trc_summary", trc),
        ("internvl_condition_accuracy", accuracy),
        ("internvl_control_differences", paired),
        ("internvl_accuracy_relation", relation),
        ("exp1_comparison", pd.concat([deep_exp, exp1], ignore_index=True)),
        ("trc_comparison", pd.concat([deep_trc, trc], ignore_index=True)),
    ]:
        table.to_csv(output / f"{name}.csv", index=False)
    sources = [notebook, deep_exp_path, deep_trc_path, *sweep_sources]
    manifest = {
        "sources": [
            {
                "path": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sources
        ],
        "notebook_table_locations_zero_based": locations,
        "internvl_checkpoint": str(exp1["model"].iloc[0]),
        "internvl_revision": str(exp1["model_revision"].iloc[0]),
        "internvl_precision": "expert-nf4",
        "precision_note": (
            "InternVL Exp1 uses the exact saved full-sweep CSV values. Exp2 remains extracted from six-decimal notebook tables and printed analysis logs."
            if sweep_sources
            else "Notebook displays six-decimal tables and four-decimal Exp1 RACR/Top-1 logs; raw InternVL outputs are not present locally."
        ),
        "comparison_note": (
            "Exp1: DeepSeek n=1000 at 25/50/70/75% retention; InternVL n=256 at 25/50/70/75/100%, with separate L2/random identity controls at 100%. "
            if sweep_sources
            else "Exp1: DeepSeek n=1000 at 25/50/70/75% retention, InternVL n=256 at 50% only. "
        )
        + "Exp2: n=256 each; paired cross-model manifests unavailable. DeepSeek correctness is recorded only for chronological inputs; InternVL has four answer-generating controls. Router permutation is a routing-only diagnostic. Precision, preprocessing and expert counts differ. No third model results supplied.",
    }
    (output / "provenance.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Extracted two completed 256-example InternVL runs to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--notebook",
        type=Path,
        default=ROOT / "results/evidence/internvl_experiments_1_2.ipynb",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts/reproduction/tables"
    )
    args = parser.parse_args()
    extract(args.notebook.resolve(), args.output_dir.resolve())
