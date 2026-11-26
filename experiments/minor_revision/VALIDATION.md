# Validation record

Prepared on 21 November 2026. No full-checkpoint results are bundled or claimed.

## Completed locally

- Python 3.12.16, PyTorch 2.9.1, Torchvision 0.24.1, Transformers 5.5.0, Accelerate 1.12.0, NumPy 2.5.5.
- `python -m pytest test_experiments.py -q`: **12 passed, 1 skipped** (CUDA unavailable).
- Python compilation and shell syntax checks.
- Real tiny Qwen3.5 native visual forward versus the adapter: matching routes and next-token prediction.
- Cached greedy generation versus native `generate` on the unpruned tiny model.
- Identity controls, retained original positions, per-image retention, and native packed versus split experts.
- Two-shard synthetic checkpoint loading, with no uninitialized parameters and matching reference logits.
- Synthetic end-to-end run, raw-route metric recalculation, bootstrap analysis, and rejection of a missing condition.
- All non-MTP tensor names in the actual checkpoint indexes matched the pinned Transformers model definitions (1,164 tensors for 122B; 1,026 for each 35B model).
- Actual public model configs, tokenizers/processors and checkpoint indexes loaded successfully; synthetic image preprocessing returned the required native fields. This check used small metadata/tokenizer downloads, **not model weights**.

| Model alias | Checked immutable revision |
|---|---|
| `qwen35_122b` | `dc4d348443bc740c68e2d77492492c11606384d5` |
| `qwen35_35b` | `2508b2437e382c605eb0fc52b83f5545960f6b4a` |
| `qwen36_35b` | `345d467b117d1fd5eda9b879a5988de9ae5f47a7` |

## Pending on the user's GPU server

- Actual CUDA/NF4 kernel and dispatch test: included in the test suite and selected automatically when CUDA is available.
- Full-checkpoint loading, memory headroom and execution time on the selected H200(s).
- Native/adapted equivalence and 100% identity controls on the real checkpoint and each real benchmark.
- Baseline answer formatting/accuracy sanity checks and complete paired evaluations.

Use the README's smoke commands before the full experiment. Passing CPU checks or metadata checks is not a substitute for these GPU checks. Report only completed full evaluations as empirical evidence.
