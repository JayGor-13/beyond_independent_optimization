# Gemma 4 research prototype — unresolved

These files preserve the local Gemma adapter, loading diagnostics, tests and historical notebook. They are **not an integrated supported experiment package**. The previous full-checkpoint Kaggle smoke test encountered non-finite values, and the candidate FP32-vision mitigation has not been verified on the full checkpoint.

The historical notebook expects a Gemma-enabled `kaggle_new_models` runner, whereas the canonical released multimodel runner supports InternVL/Qwen only. Do not launch it as if it were a completed working workflow. The prototype also imports shared modules from that older development package and is excluded from the supported CPU test matrix. Integration and an actual GPU smoke test are required before using it for new evidence.

`GEMMA4.md` retains the historical development notes and model/precision details. The parent [experiment inventory](../../../docs/EXPERIMENTS.md) identifies the supported workflows. No Gemma measurements are included in the reported paper results.
