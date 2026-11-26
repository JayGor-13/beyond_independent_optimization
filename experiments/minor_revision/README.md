# Additional experiments for the TMLR minor revision

This standalone package extends the **Section 4.2 compression-to-routing experiment** to a 122B-class multimodal MoE and harder benchmarks. It does not edit the manuscript or existing experiment code/results.

The action editor suggested larger models and broader, more complex multimodal benchmarks. The editor did not prescribe a particular model, dataset, precision, or sample count. The choices below are our proposed response.

## Models and hardware

| CLI model | Checkpoint | Recommended role | Precision on one H200 |
|---|---|---|---|
| `qwen35_122b` | `Qwen/Qwen3.5-122B-A10B` | Main larger-model experiment | `expert-nf4` |
| `qwen35_35b` | `Qwen/Qwen3.5-35B-A3B` | Smaller model in the same family; full BF16 replication | `bf16` |
| `qwen36_35b` | `Qwen/Qwen3.6-35B-A3B` | Optional newer checkpoint | `bf16` |

The 122B and 35B names describe approximately 122B/10B-active and 35B/3B-active **language** parameters; vision and other checkpoint components add storage. Qwen3.5 combines Gated DeltaNet, attention, and routed/shared experts. Its different architecture is part of the generalization test, not an isolated parameter-scaling intervention.

An H200 has 141 GB memory. The 122B language weights alone require approximately 244 GB in BF16. Use **one H200 with expert-NF4**, or suitable multiple GPUs for BF16. Full-checkpoint memory and runtime remain subject to the GPU smoke test. The loader estimates weights plus a 12 GiB per-GPU runtime reserve and stops before downloading shards if placement is too large.

`expert-nf4` explicitly quantizes only the **routed expert matrices** to NF4 with double quantization and BF16 computation. Routers, shared experts, vision, attention, embeddings, and the output head retain BF16 weights. Both unpruned and pruned conditions use the same loaded weights and precision. This supports a claim about the evaluated quantized model; it is not evidence of BF16/NF4 equivalence.

**Select the H200 explicitly.** The loader partitions layers equally across visible GPUs; exposing a 24 GB RTX 4090 alongside the H200 can make that placement fail. Use the 4090 for independent small checks rather than assuming its memory pools with the H200.

## Benchmarks

| CLI benchmark | Public source/configuration | Default full-run coverage |
|---|---|---|
| `scienceqa` | `derek-thomas/ScienceQA`, `test`, image-containing rows | 1,000 examples; seed 20260818 |
| `mmmu_pro` | `MMMU/MMMU_Pro`, `standard (10 options)`, `test` | Full split; currently 1,730 questions |
| `mathvista` | `AI4Math/MathVista`, `testmini` | Full split; currently 1,000 questions |

Every run records the actual sample count and immutable dataset commit. MMMU-Pro Standard-10 is evaluated separately; this package does not claim the combined Standard/Vision MMMU-Pro score.

MMMU-Pro image references are interleaved in their original question/option positions, and all images are retained in the problem. Choices support A–J and beyond. ScienceQA hints/lectures/solutions and MMMU-Pro explanations are excluded from prompts. MathVista uses the question, choices, unit, and requested numeric precision; the gold answer never enters the prompt.

## Files

- `setup.sh`, `requirements.txt`: separate Python environment and pinned model/data dependencies.
- `run_h200.sh`: convenient smoke/full/data-plan commands.
- `run.py`: resumable experiment runner; loads a model once for all requested benchmarks.
- `loading.py`: shard-wise loading and explicit packed-expert quantization.
- `adapter.py`: native visual preparation, pruning, router capture, and greedy decoding.
- `data.py`, `protocol.py`, `routing.py`: benchmark handling, scoring, reproducibility, and metrics.
- `analyze.py`: completeness validation, paired confidence intervals, and CSV/Markdown reports.
- `test_experiments.py`: CPU integration tests and an automatically selected CUDA/NF4 test.

## 1. Set up on the GPU server

Copy this entire directory to the server. Run from this directory, for example:

```bash
cd "/path/to/repository/experiments/minor_revision"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
```

Set the actual H200 index. The example assumes it is GPU 0:

```bash
export CUDA_VISIBLE_DEVICES=0
TMLR_PYTHON=python3.12 bash setup.sh
```

Python 3.12+, Internet access to Hugging Face, and a CUDA driver compatible with the installed PyTorch wheel are required. Setup creates `.venv` and does not change the existing experiments' environments. The local CPU checks were developed with PyTorch 2.9.1 and Transformers 5.5.0. Torchvision 0.24.1 is required by the native processor even for image-only use.

If your server needs a specific CUDA wheel, create the environment and install the matching PyTorch/Torchvision builds first; then install the remaining requirements and run tests. For example, use the official PyTorch installation selector for the server's driver. Do not independently upgrade Transformers: this adapter and its packed expert layout are pinned to 5.5.0.

Storage: checkpoint shards are downloaded into a fresh temporary directory and removed one at a time. Quantized inference still downloads the original BF16 source weights—roughly 250 GB for a 122B-model load. Plan sufficient network transfer, host RAM for the largest shard/tensor, and scratch space for at least the largest shard. Raw routing outputs can also be large; estimate their size from the smoke run. `--route-storage metrics` retains predictions and per-example/per-layer metrics but omits raw expert IDs, reducing independent auditability. The default is `full`.

The instrumented eager/split-expert path is not a throughput-optimized serving engine. Use a pilot to estimate wall time before committing to the full matrix; do not infer runtime from a hosted model's advertised token rate.

Use a server data disk by setting these before running:

```bash
export TMLR_OUTPUT="/path/to/data/tmlr_minor_revision_results"
export TMLR_SCRATCH="/path/to/data/tmlr_model_shards"
```

When these variables are omitted, outputs and scratch are placed in this directory.

## 2. Prepare a plan without model weights

```bash
bash run_h200.sh plan
```

This verifies the model commit pinned in `protocol.py`, resolves and saves dataset commit SHAs, prepares the datasets/manifests, and prints the actual condition counts. It does **not** download model weight shards. The same `revisions.json` is reused by all models and on resume. Full benchmark datasets may be downloaded at this stage.

## 3. Run and inspect the 122B smoke test

```bash
bash run_h200.sh smoke
.venv/bin/python analyze.py --output-root "${TMLR_OUTPUT:-outputs}" --include-smoke --verify-routes
```

Smoke mode uses **two questions per benchmark** in separate `smoke/` directories. It checks native/adapted unpruned logits, exact visual routes, 100% identity controls, all pruning conditions, answer generation, saving, and analysis. Local tiny-model tests do not establish that the actual checkpoint fits or loads successfully on your GPU.

Inspect `loading_info.json`, `native_equivalence/*.json`, the baseline raw answers/valid-answer flags, and `RESULTS.md`. Resolve loading errors, identity mismatches, invalid answer formatting, or unexpected baseline behavior before a full run. Two-example smoke accuracies are debugging outputs, not paper evidence.

## 4. Run the main 122B experiment

```bash
bash run_h200.sh full
```

The default matrix is **15 conditions per question**:

- One native unpruned baseline.
- L2 and random 100% retention identity controls.
- L2 selection at 75%, 50%, and 25% retention.
- Random selection at each budget with seeds 0, 1, and 2.

With the current benchmark sizes, the default covers 3,730 questions per model. Actual counts are recorded by the scripts. Routing is collected only during the prompt prefill; generated answer tokens are excluded.

For a smaller operational pilot, use a separate output root:

```bash
TMLR_OUTPUT="/path/to/data/tmlr_pilot" bash run_h200.sh full --limit 32
```

Use the full run's default root again for the paper experiment. Changing sample limits, token budgets, precision, seeds, source code, dependencies, or other recorded settings in an existing run is rejected.

## 5. Run the BF16 replication

The same-family 35B checkpoint is the default second model:

```bash
TMLR_MODEL=qwen35_35b TMLR_PRECISION=bf16 bash run_h200.sh smoke
TMLR_MODEL=qwen35_35b TMLR_PRECISION=bf16 bash run_h200.sh full
```

It uses exactly the same pinned datasets and sample manifests as the 122B run. Absolute differences between these two models also involve architecture width/depth, training, and precision; do not attribute those differences solely to parameter count.

Optional newer-model replication:

```bash
TMLR_MODEL=qwen36_35b TMLR_PRECISION=bf16 bash run_h200.sh smoke
TMLR_MODEL=qwen36_35b TMLR_PRECISION=bf16 bash run_h200.sh full
```

Optional precision control: repeat `qwen35_35b` with `TMLR_PRECISION=expert-nf4`. If multiple sufficiently large GPUs are available, `qwen35_122b` with `TMLR_PRECISION=bf16` is also supported by explicit layer placement. Each precision has its own output directory.

## 6. Resume and analyze

Rerun the **identical command** after interruption. Per-example conditions are saved atomically. Completed records are reused; a recomputed baseline must match its saved routing hash and answer before pending conditions are added. One process lock prevents simultaneous writers to the same model/precision/phase.

When the requested full runs are complete:

```bash
.venv/bin/python analyze.py --output-root "${TMLR_OUTPUT:-outputs}"
```

For an independent recalculation of saved route metrics, add `--verify-routes`. It requires raw route files and may be expensive on full runs. Analysis fails on incomplete condition matrices; it never silently reports a smaller paired subset. Analyze separate output roots if one contains unfinished experiments you do not yet want to summarize.

```text
outputs/
  revisions.json
  manifests/
  qwen35_122b/expert-nf4/full/<benchmark>/
    run_config.json
    manifest.json
    loading_info.json
    native_equivalence/*.json
    conditions/<condition>/<example>.json
    conditions/<condition>/<example>.npz
    complete.json
    summaries/
      summary.csv
      layerwise.csv
      l2_random_contrasts.csv
      analysis_metadata.json
  qwen35_35b/bf16/full/<benchmark>/...
  combined_summary.csv
  RESULTS.md
```

Keep the entire output tree, particularly configuration/manifests and raw routing files. For manuscript tables, use the individual model/benchmark rows in `summary.csv`, the paired L2–random contrasts, and layerwise confidence intervals. Smoke results are excluded from the default analysis.

## Measurement and interpretation

Pruning occurs after the vision encoder/projector and before the text decoder. At each budget, selection is **per image**, preserving at least one token per image and recording actual token retention. L2 ranks post-projector activation norms with stable tie handling. Random masks use deterministic example-specific seeds. Original multimodal positional coordinates are retained; image markers and all text tokens remain. This is an identity-preserving diagnostic protocol, not a reproduction of a named pruning algorithm. It differs from earlier InternVL position reindexing and from a global multi-image pruning budget.

For each retained token/layer, compare its compressed expert set with the baseline set at its **original visual-token ID**. Report:

- **RACR:** one minus the fraction of baseline Top-k assignments retained.
- **Top-1 change:** fraction whose first-ranked expert changes.
- **Exact Top-k change:** fraction whose selected expert set changes.
- **Load CV difference:** expert assignment count CV after pruning minus baseline CV, both computed on the **same retained tokens**, including zero-load experts.
- Task accuracy, valid-answer rate, actual retention, and layerwise metrics.

RACR and other routing statistics are averaged within each question, then across questions. Confidence intervals use 2,000 bootstrap resamples of paired questions, never individual tokens/layers. Accuracy differences are in **percentage points**, not relative percent. Individual random seeds are reported; pooled random rows first average seeds **within each question** before bootstrapping. These intervals describe question-sampling uncertainty conditional on the tested seeds, not uncertainty over all possible random masks. Models and benchmarks are never pooled into one apparent sample population.

Greedy decoding requests direct answers with thinking disabled. Multiple-choice parsing accepts a single valid letter or a unique exact option text; ambiguous explanations are invalid. MathVista uses its published integer/float precision normalization with strict direct-answer extraction. There is no LLM judge, fuzzy option matching, or automatic selection of a number from a reasoning trace. Treat these as **our controlled task accuracies**, not automatically comparable official leaderboard scores. Inspect the saved valid-answer rates and truncation flags.

The primary claim to test is: **visual-token pruning changes downstream expert assignments for retained tokens across larger multimodal MoEs and more demanding tasks.** Report the observed outcomes, including null findings. Routing drift plus accuracy loss does not establish that routing drift caused the loss. These scripts do not measure optimized inference latency or validate a routing-stability objective.

## Validation status and sources

Local validation: **12 tests passed; 1 CUDA/NF4 test skipped** because CUDA is unavailable on the local Mac. The supplied tests use real tiny, randomly initialized Qwen3.5 models on CPU, synthetic image inputs, synthetic checkpoint shards, and synthetic experiment outputs. They verify retained-token alignment, multi-image selection, ten-option/numeric scoring, native-forward equivalence, cached greedy generation against native `generate`, packed-expert conversion, shard loading, resumability checks, and analysis completeness. The separate tiny CUDA/NF4 integration test runs automatically when CUDA is available. No full-size GPU experiment has been executed as part of preparing this package.

The actual public configurations, processors/tokenizers, and checkpoint indexes of all three supported models were also checked without downloading weight shards. Their image processors returned the required `input_ids`, `attention_mask`, `pixel_values`, `image_grid_thw`, and `mm_token_type_ids` fields for a synthetic image. The checked commits are pinned in `protocol.py`; see `VALIDATION.md` for the distinction between completed checks and GPU checks still pending.

Primary references checked for implementation:

- [Qwen3.5-122B-A10B model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B).
- [Qwen3.5-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B).
- [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B).
- [Pinned Transformers 5.5.0 implementation](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py).
- [MMMU-Pro dataset and revisions](https://huggingface.co/datasets/MMMU/MMMU_Pro).
- [MathVista dataset](https://huggingface.co/datasets/AI4Math/MathVista) and [official answer normalization](https://github.com/lupantech/MathVista/blob/main/evaluation/calculate_score.py).
- [NVIDIA H200 memory specifications](https://docs.nvidia.com/enterprise-reference-architectures/hgx-ai-factory-h100-h200-b200/latest/components.html).
