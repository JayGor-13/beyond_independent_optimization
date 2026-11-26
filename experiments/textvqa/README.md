# Experiment 1 replication: TextVQA

## Research question

Does visual-token pruning change downstream MoE expert assignments and TextVQA performance, and does L2-based retention preserve routing and task score better than random retention at the same token budget?

This is a separate replication of Experiment 1. It does not overwrite the ScienceQA run.

## Design

| Component | Setting |
|---|---|
| Model | `deepseek-ai/deepseek-vl2-tiny` |
| Dataset | `facebook/textvqa`, validation split |
| Planned sample | 1,000 questions sampled with a fixed seed |
| Input | Image and question only; no supplied OCR tokens |
| Conditions | Unpruned baseline, L2 retention, random retention |
| Retention | 75%, 70%, 50%, and 25% of visual tokens |
| Random control | Seed 0 by default, matching the completed ScienceQA matrix |
| Task metric | Official VQA-style soft accuracy over 10 human answers |
| Routing metrics | RACR, Top-1 expert flips, complete Top-k changes, load-CV change |
| Uncertainty | 2,000 paired example-bootstrap samples, 95% intervals |

TextVQA contains questions that require reading scene text. Its validation set has reference answers, while the test answers are hidden, so this experiment uses the validation split. The prompt intentionally excludes dataset-provided OCR tokens: supplying the answer-bearing text directly to the language model would weaken the visual-token pruning intervention.

## Kaggle package

Upload `textvqa_kaggle.zip` through **Kaggle > Inputs > Models**, enable Internet, and select a T4 GPU. The archive contains both this directory and the shared `compression_routing_kaggle` instrumentation used by the original experiment.

Run setup:

```python
!bash "$(find /kaggle/input -type f -path '*/experiment_1_textvqa/setup_kaggle.sh' | sort -V | tail -n 1)"
```

Run a two-question smoke test in a separate output directory:

```python
!python "$(find /kaggle/input -type f -path '*/experiment_1_textvqa/run.py' | sort -V | tail -n 1)" --limit 2 --random-seeds 0 --output-dir /kaggle/working/textvqa_smoke
!python "$(find /kaggle/input -type f -path '*/experiment_1_textvqa/analyze.py' | sort -V | tail -n 1)" --expected-examples 2 --output-dir /kaggle/working/textvqa_smoke
```

Run the planned 1,000-question experiment:

```python
!python "$(find /kaggle/input -type f -path '*/experiment_1_textvqa/run.py' | sort -V | tail -n 1)" --limit 1000 --random-seeds 0
!python "$(find /kaggle/input -type f -path '*/experiment_1_textvqa/analyze.py' | sort -V | tail -n 1)" --expected-examples 1000
```

The run is resumable. Re-running the same command skips question-condition pairs whose prediction and routing files already exist. Do not change the sample limit, dataset split, prompt, or generation settings inside an existing output directory.

## Outputs

```text
/kaggle/working/compression_routing_exp1_textvqa/
  run_config.json
  manifests/
  conditions/
    baseline/
    l2_r*/
    random_r*_s0/
  summaries/
    summary.csv
    per_example.csv
```

`summary.csv` is intentionally compatible with the Experiment 1 plotting workflow. Here, the `accuracy` column is mean TextVQA VQA soft accuracy, not multiple-choice exact accuracy.

## Allowed conclusion

If the controls behave consistently, the experiment can show that visual-token selection affects downstream MoE routing on an OCR-heavy VQA task and can compare L2 and random retention at matched budgets. It cannot establish that routing changes cause score changes.

