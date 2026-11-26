# Validation record

Date: 28 September 2026.

## Completed locally

An isolated environment used Transformers 4.57.6, Accelerate 1.12.0, BitsAndBytes 0.49.2 and the host's CPU PyTorch. The optional InternVL fixture used official source files from revision `ba1762d830e2aeddb3272b66ba9e0abf7e69f626`.

Command: `python -m pytest kaggle_new_models -q`, with `INTERNVL_TEST_SOURCE` pointing to the parent of the downloaded `internvl_source` fixture.

Result: **16 passed, 1 skipped**. The skipped test requires CUDA.

Verified:

- Tiny real Qwen3-VL native multimodal logits match the prepared-decoder path.
- Tiny real InternVL native multimodal logits match the adapter path.
- Cached greedy decoding matches native generation in both tiny models.
- Qwen pruning preserves original token identity, mRoPE coordinates and aligned DeepStack features.
- Splitting Qwen packed expert matrices preserves floating-point model logits.
- Shard loading fills all expected meta-initialized parameters, leaves no meta parameters, and removes only its temporary shard directory.
- Frame identities, normalized routing mass and assignment counts are consistent.
- Cross-clip sampling excludes the target video even in a two-video smoke test.
- Changed configurations cannot silently reuse previous outputs.
- Completed per-condition records are skipped on resume.
- Both actual analysis entry points accept newly produced fixture outputs, and incomplete matrices are rejected.
- Python sources parse; both command-line help interfaces work; notebook cells and ZIP contents are checked during packaging.

## Still requires Kaggle

The local host has no CUDA GPU. Full checkpoint loading, GPU placement, BitsAndBytes NF4 execution, real ScienceQA/NExT-QA inference, peak memory and full-run duration have not been measured. No scientific result or accuracy value for the two new models is claimed.

Each notebook runs a small CUDA/NF4 loader test before downloading the full model. Its two-example smoke run then verifies native unpruned logits and next-token agreement before writing experiment records. Continue to the full run only after that smoke run and its analysis succeed.
