# Experiment 1 implementation record

This document records the scientific and implementation decisions for the compression-routing experiment.

## Question

The primary question is whether visual-token compression changes the routing behavior of downstream MoE layers. Accuracy is measured concurrently, but the experiment does not assume that every routing change is harmful or that routing change is the only cause of an answer change.

## Fixed evaluation set

- Dataset: `derek-thomas/ScienceQA`, test split.
- Sample: 256 image-containing examples selected using seed `20260818`.
- Manifest: `manifests/scienceqa_256_seed20260818.json`.
- Prompt: image, question, answer choices, and an instruction to return one option letter.
- ScienceQA hints are excluded unless explicitly requested.
- Decoding: greedy, maximum eight new tokens.

The same manifest, prompts, model checkpoint, and decoding settings are used for every condition.

All current outputs use schema version 2 under `/kaggle/working/compression_routing_exp1`. This separate root prevents route files from older pilot implementations from being silently reused.

## Part A: identity-preserving pruning

`run.py` performs pruning before the language model.

Conditions:

- uncompressed baseline;
- embedding-L2 selection at 75%, 50%, and 25% retention;
- random selection at the same retention ratios with random seeds 0, 1, and 2.

Each visual token receives an ordinal ID before pruning. A retained token keeps that ID, allowing its expert assignments to be compared with the same token in the uncompressed run.

Primary metrics:

- paired accuracy change;
- actual retained-token ratio;
- macro-average routing assignment change rate (RACR);
- Top-1 expert-flip rate;
- complete Top-k set-change rate;
- matched-cohort expert-load CV change.

`analyze.py` produces example-level values and 2,000-sample paired bootstrap confidence intervals. Random pruning is kept as three distinct conditions; seed aggregation, if desired, is performed only after all raw seed-level results have been retained.

## Part B: official FastMMoE

Source: `https://github.com/MindVLA-Team/FastMMoE`

Pinned commit: `b355b15a46601be24e440e1449df82de08e3684d`

The setup script checks out this exact commit and installs `FastMMoE/DeepSeek-VL2` without copying or reimplementing the algorithm.

The official DeepSeek configuration uses:

- token compression at language layers 2, 5, and 8;
- hybrid attention/routing-similarity scoring;
- MLERP merging;
- routing-similarity window size 3;
- `BASE_ALPHA=0.3`;
- `MERGE_RATIO=0.05`;
- cumulative schedules `0.91^3`, `0.79^3`, and `0.63^3` for approximately 75%, 50%, and 25% final retention;
- visual-expert reduction from layer 2 with factor 0.5 in the full framework.

Two modes are intentionally separated:

1. `fastmmoe_prune` disables expert reduction and isolates progressive routing-aware token compression.
2. `fastmmoe_full` enables the paper's token compression and dynamic expert reduction together.

Without this separation, an observed routing/load change could not be attributed specifically to token compression.

## Why FastMMoE does not use RACR

FastMMoE merges groups of tokens into new representations. A merged token does not have a one-to-one identity in the uncompressed sequence. Assigning it an arbitrary original-token ID would make token-level RACR scientifically invalid.

The FastMMoE runner therefore records, at every MoE layer:

- visual-token count;
- active expert-assignment count;
- tokens received by each routed expert;
- routing mass received by each routed expert.

`analyze_fastmmoe.py` compares normalized expert traffic against an uncompressed run using:

- total-variation distance;
- Jensen-Shannon divergence;
- load-CV change;
- active-assignment retention;
- actual final visual-token retention;
- paired accuracy change.

These metrics remain defined when tokens are progressively merged.

## Routing capture

The L2/random runner captures Top-k gate outputs during a dedicated prefill pass and saves the selected expert IDs for each retained visual token and layer.

The FastMMoE runner wraps each MoE module's inference call during a dedicated prefill pass. This records the final assignments after FastMMoE has applied token merging and, for `fastmmoe_full`, expert reduction. Sentinel expert IDs used for disabled routes are excluded from active traffic.

Shared experts are not included because they are always active and are not part of the sparse Top-k router decision.

## Kaggle compatibility fixes

The pinned FastMMoE fork uses its dynamic decoder for both the standard DeepSeek wrapper and the FastMMoE wrapper. Under eager attention, its attention modules return three values while that decoder always expects a fourth routing-relation tensor. Both runners apply the shared adapter in `model_compat.py`: ordinary layers receive a fourth `None` value and compression layers receive the computed attention-probability matrix. This preserves the hybrid attention/routing signal without requiring a locally compiled `flash-attn` package on Kaggle.

The fork's standard DeepSeek wrapper also returns `(embeddings, separator mask, visual mask)` from `prepare_inputs_embeds`. The identity-preserving runner explicitly unpacks this tuple and uses the separator-free visual mask. Generation is performed through the language model using the compressed embeddings, avoiding the wrapper's image-input branch.

## Output separation

Identity-preserving conditions are stored under `conditions/`. Official FastMMoE conditions are stored under `fastmmoe_conditions/` because their route files have different valid schemas.

The two headline result tables are:

1. `summaries/summary.csv`: token-level routing changes for L2 and random pruning.
2. `summaries/fastmmoe/summary.csv`: progressive token reduction and expert-traffic redistribution for FastMMoE.

## InternVL3.5 cross-model replication

`run_internvl35.py` is a separate adapter for `OpenGVLab/InternVL3_5-30B-A3B`, pinned to model revision `ba1762d830e2aeddb3272b66ba9e0abf7e69f626`. The smaller InternVL3.5 variants are dense; the 30B-A3B checkpoint is used because routed Qwen3-MoE layers are necessary for RACR and expert-flip measurements.

The adapter repeats the identity-preserving baseline/L2/random protocol, not the progressive FastMMoE protocol. It:

- builds the same deterministic ScienceQA subset from seed `20260818`;
- uses InternVL's dynamic-resolution image tiling with at most six tiles;
- inserts the projected InternVL visual features into the language-model prompt;
- removes visual positions before the Qwen3-MoE decoder while preserving ordinal token IDs;
- records each routed layer's Top-k expert IDs from its gate output; and
- saves the same prediction and route fields consumed by `analyze.py`.

InternVL outputs live under `/kaggle/working/compression_routing_exp1_internvl35`, so neither run configuration nor partial results can be confused with DeepSeek outputs. The separate `requirements-internvl35.txt` is also intentional: official InternVL3.5 requires `transformers>=4.52.1`, which conflicts with the pinned DeepSeek environment.

The checkpoint contains about 30.8B total parameters and its BF16 files occupy about 61.7 GB. A 16-bit run needs an 80 GB-class GPU or an adequate multi-GPU configuration. The Kaggle path uses `--load-in-4bit` with NF4, nested quantization, FP16 computation, `device_map="auto"` across two T4 GPUs, and one 448-pixel image tile to control activation memory. Because this changes the numerical model, results from that path must be labeled as an InternVL3.5 4-bit replication. They should not be pooled numerically with the DeepSeek FP16 estimates.

## Verification before reporting results

For each complete run, verify:

- all conditions contain 256 predictions and 256 route files;
- actual retention is close to the requested retention;
- baseline accuracy is identical within each comparison family;
- each route file contains every expected MoE layer;
- all confidence intervals were generated from paired example resampling;
- the recorded FastMMoE commit matches the pinned commit;
- the recorded InternVL model revision matches the pinned revision;
- InternVL and DeepSeek outputs come from the same manifest IDs;
- smoke-test rows are not reported as experimental evidence.
