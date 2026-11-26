# Quick Kaggle follow-up: DeepSeek control accuracies

Import `DeepSeek_Control_Accuracy_Kaggle.ipynb` into Kaggle. It embeds the code and
the original 256 reference question/video IDs. Enable a GPU accelerator and Internet.
No additional code dataset is required.

Version `2026-10-13-py311-arrow17-fix` fixes the tokenizers build failure on Kaggle's Python
3.13 kernel. Setup creates a separate Python 3.11 environment under
`/kaggle/working/deepseek_accuracy_py311`, installs CUDA 12.1 PyTorch 2.5.1 and the
original Transformers 4.41.0/tokenizers 0.19.1 stack, then checks imports and CUDA.
Inference uses that environment's interpreter; the notebook kernel stays unchanged.
The first setup downloads CUDA wheels and can take several minutes. Libraries are
installed from wheels; setup does not attempt a Rust tokenizers source build.
PyArrow 17.0.0 is pinned with NumPy 1.26.4; leaving PyArrow unpinned allowed a
newer release requiring NumPy 2 to break the datasets import.

If the previous setup has already installed the GPU libraries but fails importing
PyArrow, paste `Kaggle_Repair_Arrow_Cell.py` into a new cell in that same session.
It updates the extracted requirements, replaces PyArrow 26 with 17, and resumes setup.
Then run the configuration and inference cells. No session restart is needed.

For the failed old notebook, import the corrected notebook and rerun its setup and
configuration cells. No code ZIP is needed, and no inference output is deleted.

1. Set `SAMPLES=2` for a smoke test, or `32` for a pilot.
2. Set `ORIGINAL_MANIFEST` to the original `nextqa_256_seed20260819.json` if available.
   Set `MODEL`/`REVISION` to the original checkpoint if known.
3. Run the notebook. It generates chronological, duplicate, shuffled and cross-clip
   answers with the original FP16 preprocessing, prompts and greedy decoding.
   No routing forward pass or routing hooks are used.
4. After checking the pilot, change `SAMPLES=256` and rerun the configuration and
   inference cells. Completed predictions are reused, so the pilot contributes to
   the complete run. Keep the same settings and output directory.
5. Download `deepseek_control_accuracy_results.zip` from the final cell.

The complete run requires 1,024 answer generations, less the completed pilot runs.
Three conditions supply missing accuracies; chronological input is rerun to compare
each question's correctness with the original results (162/256). The code prints
observed inference timing; no fixed runtime estimate is established.

## Matching the existing table

A 32-question accuracy is a pilot estimate and cannot fill a cell labelled n=256.
The full 256 questions must match the saved question/video IDs. Cross-clip frames
always use donors from the **full original manifest**, even when running only 2 or
32 target questions. Building a new 32-question manifest would change that control.

The package contains the original chronological correctness and copies the original
Experiment 2 preprocessing and frame-selection code without changes. The original
prompt's chronological-frame wording is retained in every condition for compatibility.

If no original manifest is supplied, the original seed (20260819) and sampler rebuild
the 256-item manifest; exact IDs and video membership are checked before inference.
This is not independent evidence that the historical manifest order or model revision
matches. If matching fails, the runner stops instead of silently substituting a new
sample. If the original checkpoint revision is unknown, the resolved new revision is
recorded. Changes in runtime/checkpoint/configuration block reuse of existing outputs.

`checks.json` records completeness and per-question chronological correctness
disagreements. Inspect these plus the manifest/checkpoint provenance before merging
accuracies with the existing TRC measurements. If baseline correctness differs, the
new accuracy results must be reported as a follow-up run or reconciled with the
original setup. Matching correctness alone does not prove identical routing.

Router permutation has no new accuracy: it relabels stored expert distributions and
does not define a separate answer-generating input condition.

## Local verification

`python -m unittest discover -s experiments/control_accuracy -p test_core.py`
checks sample matching, donor selection, resume configuration and paper eligibility.
The notebook source and embedded scripts can be compiled locally without a GPU.
Actual model loading and GPU inference still require the Kaggle smoke test.

Rebuild after source changes with `python experiments/control_accuracy/build_notebook.py`.
