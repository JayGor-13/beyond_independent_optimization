# Experiment 1: compression and MoE routing

For the new **InternVL3.5 and Qwen3-VL runs only**, including both Experiments 1 and 2, use the self-contained [Kaggle new-model package](../../kaggle_new_models/README.md). It has separate notebooks, outputs, pinned dependencies, and explicit expert-NF4 loading.

This experiment measures how visual-token compression changes downstream MoE routing and ScienceQA accuracy.

It has two measurement paths and one cross-model replication:

1. `run.py` evaluates uncompressed, embedding-L2, and random pre-LLM pruning. These methods preserve the identity of every retained token, so exact token-level routing changes can be measured with RACR and expert-flip rates.
2. `run_fastmmoe.py` evaluates the official FastMMoE implementation. FastMMoE progressively prunes and merges tokens inside the decoder, so it is evaluated using actual token retention and normalized expert-traffic changes rather than invalid one-to-one token matching.
3. `run_internvl35.py` repeats the identity-preserving comparison on the MoE checkpoint `OpenGVLab/InternVL3_5-30B-A3B`. It uses the same ScienceQA sampling seed, prompt, pruning conditions, route schema, and analysis as the DeepSeek run.

The official source is pinned to FastMMoE commit `b355b15a46601be24e440e1449df82de08e3684d`.

## Kaggle setup

Create a Kaggle notebook, enable Internet, and select a GPU. Attach this code through Kaggle Input. The code runs directly from the read-only `/kaggle/input` mount and writes everything to `/kaggle/working/compression_routing_exp1`.

```python
!bash "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/setup_kaggle.sh' | sort -V | tail -n 1)"
```

The setup script clones the pinned official FastMMoE repository into `/kaggle/working/FastMMoE`, installs its DeepSeek package, and records the commit in the output directory.

## Part A: L2 and random pruning

Run a small implementation check. This is not a reported experiment:

```python
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run.py' | sort -V | tail -n 1)" \
  --limit 16 \
  --methods baseline l2 random \
  --retentions 0.5 \
  --random-seeds 0

!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/analyze.py' | sort -V | tail -n 1)" --expected-examples 16
```

After the check succeeds, run the complete 256-example matrix. The 256-example manifest begins with the same 16 examples, so completed smoke-test examples are reused.

```python
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run.py' | sort -V | tail -n 1)" \
  --limit 256 \
  --methods baseline l2 random \
  --retentions 0.75 0.5 0.25 \
  --random-seeds 0 1 2

!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/analyze.py' | sort -V | tail -n 1)" \
  --bootstrap-samples 2000 \
  --expected-examples 256 \
  --require-complete-matrix
```

Random seeds are separate experimental conditions such as `random_r50_s0`, `random_r50_s1`, and `random_r50_s2`.

The primary output is:

```text
/kaggle/working/compression_routing_exp1/summaries/summary.csv
```

It reports accuracy, paired accuracy change, actual retention, RACR, Top-1 flips, complete Top-k changes, expert-load CV change, and paired bootstrap confidence intervals.

## Part B: official FastMMoE

FastMMoE configuration is read when the model is constructed. Therefore, each condition is run as a separate command.

Run the three-condition implementation check:

```python
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode baseline --limit 16
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode fastmmoe_prune --retention 0.5 --limit 16
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode fastmmoe_full --retention 0.5 --limit 16
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/analyze_fastmmoe.py' | sort -V | tail -n 1)" --expected-examples 16
```

For the complete run, execute:

```python
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode baseline --limit 256

!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode fastmmoe_prune --retention 0.75 --limit 256
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode fastmmoe_prune --retention 0.5 --limit 256
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode fastmmoe_prune --retention 0.25 --limit 256

!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode fastmmoe_full --retention 0.75 --limit 256
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode fastmmoe_full --retention 0.5 --limit 256
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_fastmmoe.py' | sort -V | tail -n 1)" --mode fastmmoe_full --retention 0.25 --limit 256

!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/analyze_fastmmoe.py' | sort -V | tail -n 1)" \
  --bootstrap-samples 2000 \
  --expected-examples 256 \
  --require-complete-matrix
```

The FastMMoE summary is:

```text
/kaggle/working/compression_routing_exp1/summaries/fastmmoe/summary.csv
```

Its primary fields are:

- `actual_retention`: last measured visual-token count divided by the first-layer count;
- `assignment_retention`: active expert assignments relative to the matched uncompressed run;
- `load_tv`: total-variation distance between normalized expert-load distributions;
- `load_jsd`: Jensen-Shannon divergence between normalized expert-load distributions;
- `load_cv_delta`: change in expert-load imbalance;
- `accuracy_delta`: paired accuracy change from the uncompressed FastMMoE-code baseline.

`fastmmoe_prune` disables dynamic expert reduction and isolates progressive routing-aware token compression. `fastmmoe_full` uses the official DeepSeek settings: compression at layers 2, 5, and 8 and visual-expert reduction from layer 2 with factor 0.5.

## Cross-model replication: InternVL3.5-30B-A3B

Run InternVL3.5 in a separate notebook or fresh environment. Its official code requires `transformers>=4.52.1`, whereas the pinned DeepSeek/FastMMoE path uses `transformers==4.38.2`. Do not install both requirement files into the same running kernel.

The selected checkpoint is the 30.8B-total/3B-active MoE model. The smaller InternVL3.5 checkpoints are dense and therefore cannot provide routed-expert measurements. The model files occupy about 61.7 GB. On Kaggle, select **GPU T4 x2** and use the 4-bit command below; a single P100 cannot hold this model. The 4-bit run is a quantized-model replication and must be described that way in the paper rather than presented as a BF16 run.

Install the separate dependencies:

```python
!bash "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/setup_internvl35.sh' | sort -V | tail -n 1)"
```

Restart the kernel after installation if `transformers` was already imported. Then run the central 50%-retention replication:

```python
!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/run_internvl35.py' | sort -V | tail -n 1)" \
  --limit 256 \
  --methods baseline l2 random \
  --retentions 0.5 \
  --random-seeds 0 \
  --load-in-4bit \
  --dtype float16 \
  --max-tiles 1 \
  --max-gpu-memory 14GiB

!python "$(find /kaggle/input -type f -path '*/compression_routing_kaggle/analyze.py' | sort -V | tail -n 1)" \
  --output-dir /kaggle/working/compression_routing_exp1_internvl35 \
  --bootstrap-samples 2000 \
  --expected-examples 256
```

The checkpoint is pinned to revision `ba1762d830e2aeddb3272b66ba9e0abf7e69f626`. The Kaggle T4 x2 command fixes preprocessing to one 448-pixel tile per ScienceQA image to control activation memory; larger-GPU runs may raise `--max-tiles` to six. `run_internvl35.py` compresses the projected visual-token sequence before it enters the language model and records the Top-k Qwen3-MoE assignments for each retained token. Its results are written separately to:

```text
/kaggle/working/compression_routing_exp1_internvl35/summaries/summary.csv
```

## Output layout

```text
compression_routing_exp1/
  fastmmoe_commit.txt
  manifests/
  conditions/                 # baseline, L2, random
  fastmmoe_conditions/        # official-code baseline and FastMMoE
  summaries/
    summary.csv
    per_example.csv
    fastmmoe/
      summary.csv
      per_example.csv
      per_example_layer.csv

compression_routing_exp1_internvl35/
  run_config.json
  manifests/
  conditions/                 # InternVL baseline, L2, random
  summaries/
    summary.csv
    per_example.csv
```

Every runner saves one example at a time and skips complete prediction/route pairs when restarted.
