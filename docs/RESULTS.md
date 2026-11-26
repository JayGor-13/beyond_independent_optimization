# Included evidence and provenance

## Current measured results

| Model / experiment | Retained evidence | Scope |
|---|---|---|
| DeepSeek-VL2-Tiny / compression | [Summary and plots](../results/published/deepseek/compression_routing) | ScienceQA, n=1,000; 25%, 50%, 70%, 75% retention |
| InternVL3.5-30B-A3B / compression | [Saved sweep outputs](../results/published/internvl/retention_sweep) | ScienceQA, n=256; all four pruning budgets plus 100% controls; expert-NF4 |
| DeepSeek-VL2-Tiny / temporal routing | [Clip-level/layer-level CSVs](../results/published/deepseek/temporal_routing) | NExT-QA, n=256; frame controls and routing-only permutation |
| InternVL3.5-30B-A3B / temporal routing | [Executed evidence notebook](../results/evidence/internvl_experiments_1_2.ipynb) | n=256; retained notebook tables and completion logs |

[Consolidated paper values](../results/paper_tables) are derived artifacts. The original values are retained unchanged under `results/published/` and `results/evidence/`. The executed notebook intentionally retains outputs because they supply evidence absent from separate result files; other runnable notebooks have cleared outputs.

## Reproduction boundary

`python tools/reproduce_paper.py` extracts tables from the executed notebook, prefers the later exact InternVL sweep CSVs, and regenerates the two empirical comparison tables and figures. It does not recreate unavailable raw predictions/routes or rerun models. The InternVL temporal results therefore retain the notebook's displayed precision. Its historical output notebook is an evidence archive, not the current launcher.

The historical DeepSeek temporal data records answer correctness for chronological inputs only. Missing control accuracies remain unmeasured; router permutation remains routing-only. Do not fill these cells using a pilot or a new checkpoint without reconciling question IDs, donor order and checkpoint/runtime provenance.

## Hashes and source selection

[The source manifest](source_manifest.json) maps original workspace-relative paths to release paths and records source SHA-256 values. Repository checks enforce unchanged bytes for original published data and evidence. Windows-style paths in the old consolidated provenance are superseded by paths regenerated with the relocated extraction script.

The canonical multimodel runner comes from the supplied `internvl_original_retention_update/kaggle_new_models` package, which supports the full retention matrix. The older root-level `kaggle_new_models` runner only defaults to the 50% comparison. These versions are not silently combined. The code selection does not independently prove that the saved historical outputs were produced by the release's current runtime; retain the supplied configuration and precision caveats.

No real-data measurements for the new Qwen3.5/Qwen3.6 revision experiments are included. The revision directory contains implementation and synthetic tests only. Likewise, no measured Qwen3-VL, TextVQA, graded-perturbation or Gemma results have been inferred from the presence of their scripts.
