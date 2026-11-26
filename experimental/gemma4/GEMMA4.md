# Gemma 4 26B-A4B: Kaggle experiments 1 and 2

Use `Gemma4_Kaggle.ipynb` with the updated `kaggle_new_models` code package. This replaces Qwen as the second new model alongside InternVL3.5. The old Qwen implementation remains available for reproducibility, but its notebook is excluded from the new upload ZIP.

## Run

1. Upload `kaggle_new_models.zip` as a new version of the Kaggle Dataset/Model that holds the code, then select that version in the notebook's inputs. The extracted package must contain `kaggle_new_models/gemma_loading.py`.
2. Import `Gemma4_Kaggle.ipynb`. Save existing outputs before starting a fresh session. Select **T4 x2**, enable Internet, and attach only one copy of the code package.
3. Run setup. The first cell must print `2026-09-30-gemma4-vision-fp32`. Gemma installs Transformers **5.5.0**; InternVL keeps **4.57.6** in its own notebook/session. The setup preserves Kaggle's installed torch/CUDA runtime.
4. Leave the broad development suite disabled, run the new small CUDA preflight, then run the two-example smoke test and analysis. Proceed to the 256-example cells only if both native-equivalence checks and analyses succeed.
5. Download the result archive before ending the session. The new notebook writes to `/kaggle/working/gemma4_vision_fp32_results/gemma4/`, keeping this precision policy separate from previous attempts.

The pinned checkpoint is `google/gemma-4-26B-A4B-it`, revision `4d7ae4984b7db7de8f8457170b3f1a419ee76d52`. Its metadata was publicly accessible without a token during development. If Hub rate limits interrupt downloads, supply `HF_TOKEN` through Kaggle secrets/environment; never paste it into the notebook source or outputs.

## Memory and loading

The official files are **49,907,246,508 bytes** and **1,704,763,408 bytes**. Downloading the first shard to a standard Kaggle working directory is unsuitable. This loader uses validated HTTP byte ranges with **32 MiB download windows**, reads each routed expert separately, and quantizes it on its assigned GPU. It does not save either full shard. A server that ignores range requests causes a clear error before a full file is read.

All parameters still have to be transferred: about **51.6 GB per model load**. Smoke and full runs are separate processes and therefore each reload the weights. Full-run duration and Kaggle peak RAM are not measured.

Only routed experts use NF4 with double quantization and FP16 computation. The vision tower and visual projection now stay **FP32**, including normalization buffers. Shared dense text MLPs, routers, text attention, embeddings and the tied output head stay FP16. Vision processes one image at a time; all eight frame embeddings remain in the text sequence. Weight estimates are approximately **10.37 GiB on GPU 0** and **8.98 GiB on GPU 1**. These are estimates, not measured peaks; the loader requires an additional 2 GiB free per GPU before starting. Host buffers, activations, CUDA workspaces and runtime caches add memory.

The previous Kaggle run loaded successfully but failed its first native-equivalence check with NaNs. This revision targets possible FP16 overflow in the vision path; it is a candidate mitigation, not a confirmed fix. A local real-weight vision probe produced finite outputs in both FP16 and FP32 on one synthetic input, so it did not reproduce the Kaggle failure. Complete T4 validation is still required. It traces module outputs during initial visual preparation and native/adapted checks and writes the first invalid output's module, stage, dtype and shape to `/kaggle/working/model_shards/gemma4_numerics.jsonl`. It also rejects non-finite route probabilities and generation logits in later conditions. It does not clamp invalid values, loosen tolerances or bypass native equivalence. If the smoke test fails, preserve that diagnostic file and the final traceback.

Memory logs persist at `/kaggle/working/model_shards/gemma4_loading_memory.jsonl`. A partial loading failure requires restarting model loading; completed experiment records still resume normally.

## Scientific protocol

- Experiment 1: same sampled ScienceQA examples, baseline/L2/random conditions, 50% retention and random seed 0.
- Experiment 2: same NExT-QA sampling, **eight separately encoded frame images**, and chronological/shuffle/duplicate/cross-clip conditions. This does not use Gemma's native video interface.
- The image processor uses a maximum **280 visual tokens per image** by default. `--gemma-soft-tokens` permits 70/140/280/560/1120; changes are recorded in `run_config.json` and require a new output directory.
- Thinking is disabled; the common eight-token answer budget is preserved.
- Pruning operates on projected visual embeddings. Retained sequences are reindexed, matching the InternVL positional policy. Gemma's native attention-mask builder reconstructs its causal/global and sliding/bidirectional image masks after pruning. Native embedding scaling and output-logit softcapping are preserved.
- Routing capture includes all **30 routed layers**, **128 experts**, **top-8** selections. The always-on dense MLP is excluded from routed-expert statistics.
- Weighted TRC uses normalized selected router probabilities **before Gemma's learned per-expert output scales**. Those scales remain active in model inference. This probability definition is recorded explicitly; it must not be described as measuring the magnitude of scaled expert contributions.
- Native unpruned logits and next-token agreement are checked before either experiment writes its first result. This verifies the adapter path; it does not establish NF4/full-precision equivalence.

Report Gemma separately as an expert-NF4 replication with FP32 vision, its preprocessing and routing policy. Both vision dtype and per-forward image count are saved in run configurations and exported summaries. No full-model experiment results have been generated locally.

## Validation

The CPU suite in the isolated Gemma environment covers actual tiny Gemma native logits, cached generation, multi-image frame identity, both pruning methods, split-expert arithmetic, learned router-scale handling, complete loading with tied weights and persistent buffers, bounded HTTP behavior, and shared experiment/analysis logic. Two additional tests require CUDA (expert NF4 math and complete loading/dispatch); those are skipped on this CPU host. See `VALIDATION.md` for final test counts.

The full architecture was instantiated on meta at about **496 MiB process RSS**. Every official checkpoint tensor name and shape matched; all 30 expert blocks were identified. Real HTTP range requests successfully read the headers of both official shards without downloading them in full.

The actual pinned tokenizer, chat template and image processor were also checked with two images and with eight synthetic frames. The eight-frame prompt retained all eight visual blocks (266 tokens per frame for that test aspect ratio), with 2,200 total prompt tokens. The empty thinking-channel prefix was verified. These are preprocessing checks, not real-data inference results.

Full checkpoint inference, real-data accuracy, T4 NF4 execution and peak Kaggle memory remain to be verified by the notebook smoke test.

Sources: [official checkpoint](https://huggingface.co/google/gemma-4-26B-A4B-it), [pinned model code](https://github.com/huggingface/transformers/blob/v5.5.0/src/transformers/models/gemma4/modeling_gemma4.py).
