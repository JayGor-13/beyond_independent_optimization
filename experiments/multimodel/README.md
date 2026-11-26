# InternVL3.5 / Qwen3-VL compression and temporal-routing replications

This is the canonical retention-sweep package selected for the repository. InternVL supports the full saved paper condition matrix, including 100% identity controls. Historical measured evidence is in `../../results/`, separate from new outputs.

## From a repository checkout

Use a separate CUDA environment with `requirements.txt` (Transformers 4.57.6). The CPU test profile is in `../../environments/requirements-tests-multimodel.txt`; it deliberately omits GPU-only bitsandbytes dependencies.

```bash
python experiments/multimodel/run.py --model internvl35 --experiments 1 --smoke --output-root artifacts/internvl --scratch artifacts/model_shards
python experiments/multimodel/analyze.py --model internvl35 --experiments 1 --smoke --output-root artifacts/internvl
# After the smoke check, repeat without --smoke for 256 examples.
python experiments/multimodel/run.py --model internvl35 --experiments 1 --limit 256 --output-root artifacts/internvl --scratch artifacts/model_shards
python experiments/multimodel/analyze.py --model internvl35 --experiments 1 --limit 256 --output-root artifacts/internvl
```

For both experiments, omit `--experiments 1`; for Qwen, use `--model qwen3vl`. Explicit retention/seed overrides are recorded and require a new output directory.

To prepare Kaggle, run `python tools/build_kaggle_bundle.py multimodel` from the repository root and upload `dist/multimodel_kaggle.zip`. Its internal folder remains `kaggle_new_models` to match the notebook's input discovery.

The detailed Kaggle protocol follows. Commands inside Kaggle refer to that upload-folder alias.

# Kaggle protocol

This self-contained package runs **only**:

1. `internvl35`: `OpenGVLab/InternVL3_5-30B-A3B`
2. `qwen3vl`: `Qwen/Qwen3-VL-30B-A3B-Instruct`

Both checkpoints are pinned to immutable revisions in `core.py`. Existing DeepSeek code and results are unchanged. The notebooks run a two-example smoke test first, followed by the 256-example experiment. The package contains implementation and local tests, not measured results from these full checkpoints.

## Start on Kaggle

1. Upload `multimodel_kaggle.zip` as a Kaggle Dataset and attach it to a notebook. The extracted input should contain `kaggle_new_models/run.py`.
2. Import `InternVL35_Kaggle.ipynb` or `Qwen3VL_Kaggle.ipynb`. Use **one model per notebook/session**.
3. Enable Internet and select **GPU T4 x2**. A single 16 GB GPU is insufficient for the default loading strategy. The loader checks estimated weight memory plus a 2 GiB reserve per GPU before downloading weights; activation memory still depends on the input.
4. Run the setup and smoke-test cells. Setup also runs local tests, including a tiny CUDA/NF4 checkpoint-loading test, before the full checkpoint download. Inspect the smoke results and `native_equivalence.json` before executing the full-run cell.
5. Download the result ZIP from `/kaggle/working` before ending the session. A fresh Kaggle session does not preserve the old working directory automatically.

No DeepSeek setup or FastMMoE repository is required. Installation and experiment execution use subprocesses so a previously imported notebook library cannot retain an old Transformers version. Each notebook runs both experiments on its selected model and then analyzes them.

## Commands

In a Kaggle notebook, find the uploaded package once:

```python
from pathlib import Path
import subprocess, sys
matches = list(Path('/kaggle/input').rglob('kaggle_new_models/run.py'))
assert len(matches) == 1, f'Expected one package, found {matches}'
CODE = matches[0].parent
subprocess.run(['bash', str(CODE / 'setup_kaggle.sh')], check=True)
MODEL = 'internvl35'  # use 'qwen3vl' in the second notebook
```

Smoke test, both experiments, separate output folders:

```python
subprocess.run([sys.executable, str(CODE / 'run.py'), '--model', MODEL, '--smoke'], check=True)
subprocess.run([sys.executable, str(CODE / 'analyze.py'), '--model', MODEL, '--smoke'], check=True)
```

Full run, both experiments:

```python
subprocess.run([sys.executable, str(CODE / 'run.py'), '--model', MODEL, '--limit', '256'], check=True)
subprocess.run([sys.executable, str(CODE / 'analyze.py'), '--model', MODEL, '--limit', '256'], check=True)
```

Add `--experiments 1` or `--experiments 2` to both commands to run only one experiment. InternVL defaults to the full sweep: baseline plus L2/random at 100%, 25%, 50%, 75%, and 70% retention (11 conditions), with seed 0. Qwen defaults to baseline plus L2/random at 50%. For a larger matrix, append `--retentions 0.75 0.5 0.25 --random-seeds 0 1 2` to the runner and choose a new `--output-root` for both runner and analyzer. Do not change settings in a directory with existing results.

Full BF16 requires larger GPUs and `--precision bfloat16`. FP16 is also available. The default is **expert-NF4**: routed expert matrices are quantized to NF4 with double quantization and FP16 computation; router gates, attention, vision, embeddings and output heads remain FP16. This is a quantized-model replication and must be reported as such.

## What the experiments measure

**Experiment 1:** ScienceQA test examples with images, seed 20260818, no extra hint text. Visual-token pruning occurs before the text decoder. Retained original visual-token IDs are saved for paired RACR, Top-1 flips, Top-k set changes and load-CV analysis. Qwen's DeepStack features are pruned using the same retained IDs, and its native multidimensional position coordinates are preserved. InternVL follows the existing runner's sequential position reindexing. Record this positional-policy difference when interpreting cross-model differences. No inference-speed claim should be made from this instrumentation.

**Experiment 2:** NExT-QA test set, seed 20260819, one question per unique video, eight sampled frames. Both models receive frames as **separate images with frame labels**, preserving an identifiable visual-token block for every frame. This is a sampled-frame, multi-image replication; it does not claim to test Qwen's native video-token pathway. Conditions are chronological, shuffled, duplicated middle frame, and cross-clip. Prompt wording is held fixed across controls. Cross-clip frames cycle through other sampled videos, never the target video, including the two-video smoke test. This corrects target-video leakage in the old runner's small-sample wraparound behavior; record the control policy when comparing historical outputs.

TRC routes come only from the prompt forward pass, not autoregressive answer tokens. Weighted TRC uses the routers' selected and renormalized probabilities. Accuracy is evaluated for every condition, and the existing TRC–accuracy relation uses chronological answers. Shuffling uses the same deterministic permutation across models for each example.

## Loading, validation and resume

- Official weight shards are downloaded to a fresh temporary directory, loaded and then removed one at a time. No persistent full-size model checkpoint is written. Expect roughly 62 GB of network transfer per model load, even for quantized inference; allow at least 12 GiB of free working disk plus room for datasets/results. Existing completed runs skip model loading.
- Qwen's packed experts are split into equivalent per-expert linear layers before quantization. This makes expert quantization explicit and avoids allocating all experts' activations for every token. The split implementation is checked against the original floating-point expert computation in local tests.
- Model/data revisions, library versions, preprocessing, precision and control settings are recorded in `run_config.json`. Resuming with different settings fails. Two models sharing an output root must use identical stored sample manifests and data revisions.
- The first pending example of each experiment checks native unpruned logits against the adapter and checks the next-token prediction. Failure stops the run before recording that example. This check does not establish quantized/full-precision equivalence.
- Outputs are written one example/condition at a time using atomic file replacement. Rerunning the same command reuses complete records; it does not mix smoke and full results.
- Analysis validates the complete requested condition matrix. It refuses incomplete outputs instead of silently computing results from a subset.

## Results to download

```text
/kaggle/working/new_model_results/
  shared_manifests/
  internvl35/                         # or qwen3vl/
    experiment_1_smoke/
    experiment_2_smoke/
    experiment_1_n256/
      run_config.json
      loading_info.json
      native_equivalence.json
      manifests/examples.json
      conditions/*/{predictions,routes}/
      summaries/summary.csv
      summaries/per_example.csv
    experiment_2_n256/
      run_config.json
      loading_info.json
      native_equivalence.json
      manifests/examples.json
      routes/
      predictions/                   # chronological; original analysis schema
      condition_predictions/         # all four conditions
      summaries/summary.csv
      summaries/control_differences.csv
      summaries/accuracy_relation.csv
      summaries/condition_accuracy.csv
```

For the paper, use Experiment 1 `summary.csv` and Experiment 2 `summary.csv`, `control_differences.csv`, `accuracy_relation.csv`, and `condition_accuracy.csv`. CSVs include model, revision, precision and protocol. Keep the raw routes, predictions and manifests for auditing. Report the two models individually alongside the existing DeepSeek results; do not pool token assignments across models or present smoke outputs as evidence.

## Local validation and sources

Run `python tools/run_tests.py multimodel` with the pinned dependencies. CPU tests use small random model instances and synthetic routing fixtures; they do not download the full checkpoints. Full GPU loading, NF4 kernels, real-data accuracy and Kaggle runtime must be verified in the smoke notebooks.

Historical development validation reported **16 tests passed, 1 CUDA-only test skipped**. Current repository validation is recorded in `../../docs/VALIDATION.md`. This included the optional InternVL integration test against a local copy of its pinned official source. On Kaggle, that optional source-fixture test is skipped unless its fixture is supplied; the full-model smoke test exercises the downloaded InternVL implementation. See `VALIDATION.md`.

`analyze_exp1.py` and `analyze_exp2.py` are bundled copies of this project's existing analysis scripts, wrapped by `analyze.py` for completeness validation and model metadata.

- [InternVL checkpoint and model card](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B)
- [Qwen checkpoint and model card](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
- [Pinned Qwen3-VL implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py)
