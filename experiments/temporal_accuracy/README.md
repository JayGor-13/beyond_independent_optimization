# Experiment 3: TRC and answer performance

This package runs a paired temporal-perturbation experiment with DeepSeek-VL2-Tiny on NExT-QA. The same eight frames and question are evaluated chronologically and under graded reorderings. The analysis asks whether within-question changes in TRC track changes in correct-option margin and correctness.

It is designed for **one Kaggle T4 GPU**. InternVL is not used.

## Kaggle settings

1. Create a Kaggle notebook.
2. Select **GPU T4** as the accelerator.
3. Turn **Internet on**. Internet is needed for Python packages, NExT-QA videos, and the DeepSeek checkpoint unless those have also been uploaded as inputs.
4. Upload `temporal_accuracy_kaggle.zip` through **Models > New Model**, then attach that model to the notebook as an input.

## 1. Extract the uploaded experiment

Run this notebook cell:

```python
!mkdir -p /kaggle/working/Experiment-3
!unzip -qo "$(find /kaggle/input -type f -iname 'temporal_accuracy_kaggle.zip' | sort -V | tail -n 1)" -d /kaggle/working/Experiment-3

EXP3 = "/kaggle/working/Experiment-3/experiment_3_TRC_accuracy"
```

## 2. Install dependencies and DeepSeek-VL2 code

```python
!bash "{EXP3}/setup_kaggle.sh"
```

The setup script first looks for uploaded DeepSeek-VL2 source code under `/kaggle/input`. If it is absent, it clones the official repository. The runner similarly looks for an uploaded DeepSeek-VL2-Tiny checkpoint; if none is found, it downloads `deepseek-ai/deepseek-vl2-tiny` from Hugging Face.

## 3. Smoke test

The smoke test evaluates three questions with chronological and one mild ordering. It verifies model loading, route capture, option scoring, saving, and analysis:

```python
!python "{EXP3}/run.py" \
  --limit 3 \
  --permutation-repetitions 1 \
  --conditions chronological mild \
  --output-dir /kaggle/working/trc_accuracy_smoke

!python "{EXP3}/analyze.py" \
  --output-dir /kaggle/working/trc_accuracy_smoke \
  --allow-incomplete \
  --bootstrap 200 \
  --randomizations 200
```

## 4. Full 256-question experiment

Use a clean output directory after the smoke test:

```python
!python "{EXP3}/run.py" \
  --limit 256 \
  --permutation-repetitions 2 \
  --output-dir /kaggle/working/trc_accuracy_outputs
```

The complete design has eight orderings per question: one chronological, two mild, two medium, two severe, and one reverse. This produces 2,048 resumable question-condition runs.

If a notebook stops, rerun the identical command. Existing complete route/prediction pairs are skipped.

If an MP4 reports an unavailable frame, the runner substitutes the nearest unique decodable frame and records the requested index, actual index, and offset in the prediction. It does not terminate the experiment for an isolated decode failure.

Then analyze:

```python
!python "{EXP3}/analyze.py" \
  --output-dir /kaggle/working/trc_accuracy_outputs \
  --bootstrap 2000 \
  --randomizations 2000
```

Package the results for download:

```python
import shutil
shutil.make_archive(
    "/kaggle/working/Experiment-3-results",
    "zip",
    "/kaggle/working",
    "trc_accuracy_outputs",
)
print("Created /kaggle/working/Experiment-3-results.zip")
```

## Optional: exclude Experiment 2 videos

For a strictly held-out confirmatory sample, attach the Experiment 2 output containing its `nextqa_manifest.json`, locate it, and pass it to the runner:

```python
OLD_MANIFEST = "/kaggle/input/REPLACE_WITH_ATTACHED_PATH/nextqa_manifest.json"

!python "{EXP3}/run.py" \
  --limit 256 \
  --permutation-repetitions 2 \
  --exclude-manifest "{OLD_MANIFEST}" \
  --output-dir /kaggle/working/trc_accuracy_outputs_heldout
```

An output directory is tied to its scientific configuration. Use a new directory when changing the limit, seed, exclusion manifest, frame count, or permutation repetitions.

## Optional: shard the work

The runner supports `--num-shards N --shard-index K`. All shards must use the same output directory and manifest. This is useful if several commands share one writable Kaggle session:

```python
for shard in range(4):
    !python "{EXP3}/run.py" \
      --limit 256 \
      --num-shards 4 \
      --shard-index {shard} \
      --output-dir /kaggle/working/trc_accuracy_outputs
```

## Outputs

```text
/kaggle/working/trc_accuracy_outputs/
  run_config.json
  manifests/
    nextqa_manifest.json
    permutations.json
  predictions/
  routes/
  summaries/
    per_run_layer.csv
    per_run.csv
    condition_summary.csv
    paired_differences.csv
    within_question_association.csv
    accuracy_flips.csv
    permutation_summary.csv
    analysis_metadata.json
    RESULTS_SUMMARY.md
  figures/
    temporal_delta_trc8_vs_margin.png
    condition_trc_margin_accuracy.png
```

The primary result is the temporal-question/TRC@8/margin row in `within_question_association.csv`. Complete reversal is descriptive only and is not pooled into the primary association.

This remains multi-image inference over sampled video frames, not native-video inference.
