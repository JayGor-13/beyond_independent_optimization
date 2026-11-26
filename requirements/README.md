# Shared dependency profiles

Run these installation commands from the repository root, in separate Python 3.12 virtual environments.

| File | Purpose | Install |
|---|---|---|
| `analysis.txt` | Rebuild the included result tables and figures on a CPU or Mac | `python -m pip install -r requirements/analysis.txt` |
| `tests-multimodel.txt` | InternVL/Qwen3-VL adapter tests and legacy metric fixtures; Transformers 4.57.6 | `python -m pip install -r requirements/tests-multimodel.txt` |
| `tests-minor.txt` | Additional Qwen3.5/Qwen3.6 adapter tests; Transformers 5.5.0 | `python -m pip install -r requirements/tests-minor.txt` |

Both test profiles include `analysis.txt`. Install them in separate environments because their model-library versions conflict. They support CPU validation; CUDA/NF4 checks skip without the required GPU runtime.

For actual model inference, use the runtime requirements in the relevant experiment directory. Those files travel with standalone Kaggle bundles and include the model-specific GPU dependencies. See the [development guide](../docs/DEVELOPMENT.md) and [experiment inventory](../docs/EXPERIMENTS.md).
