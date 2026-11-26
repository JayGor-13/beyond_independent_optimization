# TRC experiment

For **InternVL3.5 and Qwen3-VL**, use the self-contained [Kaggle new-model package](../kaggle_new_models/README.md). Its notebooks run both experiments on the selected new model, with sampled-frame routing and separate results.

This is a minimal Kaggle experiment for measuring Temporal Routing Consistency in DeepSeek-VL2-Tiny on NExT-QA.

The annotation table references some videos that are absent from the raw-video repository. Manifest construction checks the repository file list first and samples only questions whose `.mp4` files are available.

It runs four real eight-frame inputs for every question:

- chronological frames;
- the same frames shuffled;
- one frame duplicated eight times;
- frames from a different video.

The analysis also creates an expert-identity permutation baseline averaged over 100 random permutations. It reports original union TRC, TRC@4/8/16, weighted TRC, routing entropy, expert-load CV, video-QA accuracy, and the difference in TRC between correct and incorrect answers.

## Kaggle

Attach the uploaded project through Kaggle's **Input > Models** panel, enable Internet, and select a GPU. The code runs directly from the read-only input location. Only the DeepSeek repository and results are created under `/kaggle/working`.

Run setup:

```python
!bash "$(find /kaggle/input -type f -path '*/experiment_2_TRC/setup_kaggle.sh' | sort -V | tail -n 1)"
```

Run a two-question smoke test:

```python
!python "$(find /kaggle/input -type f -path '*/experiment_2_TRC/run.py' | sort -V | tail -n 1)" --limit 2
!python "$(find /kaggle/input -type f -path '*/experiment_2_TRC/analyze.py' | sort -V | tail -n 1)"
```

After the smoke test, run the planned 20-question control experiment:

```python
!python "$(find /kaggle/input -type f -path '*/experiment_2_TRC/run.py' | sort -V | tail -n 1)" --limit 20
!python "$(find /kaggle/input -type f -path '*/experiment_2_TRC/analyze.py' | sort -V | tail -n 1)"
```

For the preliminary TRC-accuracy analysis, use at least 256 unique videos:

```python
!python "$(find /kaggle/input -type f -path '*/experiment_2_TRC/run.py' | sort -V | tail -n 1)" --limit 256
!python "$(find /kaggle/input -type f -path '*/experiment_2_TRC/analyze.py' | sort -V | tail -n 1)"
```

Use a new output directory when changing the sample size because the manifest is tied to `--limit`:

```python
!python "$(find /kaggle/input -type f -path '*/experiment_2_TRC/run.py' | sort -V | tail -n 1)" --limit 256 --output-dir /kaggle/working/trc_outputs_256
!python "$(find /kaggle/input -type f -path '*/experiment_2_TRC/analyze.py' | sort -V | tail -n 1)" --output-dir /kaggle/working/trc_outputs_256
```

## Outputs

```text
/kaggle/working/trc_outputs/
  manifests/
  predictions/
  routes/
  summaries/
    trc_per_clip_layer.csv
    trc_per_clip.csv
    summary.csv
    control_differences.csv
    accuracy_relation.csv
```

`summary.csv` describes each routing condition. `control_differences.csv` contains paired chronological-minus-control differences and their bootstrap confidence intervals. `accuracy_relation.csv` reports mean TRC for correct and incorrect questions, their difference, its bootstrap confidence interval, and point-biserial correlation. A confidence interval spanning zero does not support a reliable difference.

This DeepSeek run is multi-image inference over sampled frames, not native video inference. Generalization requires a second experiment on a native-video MoE model.
