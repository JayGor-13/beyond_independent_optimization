# Experiment 2: Temporal Routing Consistency (TRC)

## Objective

This experiment evaluates whether Temporal Routing Consistency (TRC) is a useful and reproducible diagnostic for temporal expert routing in Mixture-of-Experts vision-language models.

The minimum claim we want to validate is:

> TRC distinguishes temporally coherent frame sequences from routing controls such as shuffled frames, unrelated frames, and permuted expert assignments.

This experiment must not assume in advance that high TRC improves video-question-answering accuracy. A relationship between TRC and task performance is a separate, stronger hypothesis that will be tested and reported independently.

## What TRC measures

For video frame \(t\), MoE layer \(l\), and expert \(e\), the router assigns visual tokens to one or more experts. TRC measures how similar these expert-routing patterns are between adjacent frames.

### Original set-based TRC

Let \(E_t^{(l)}\) be the union of routed experts activated by all visual tokens from frame \(t\) at layer \(l\). The original layer-wise TRC is:

\[
\operatorname{TRC}^{(l)} = \frac{1}{T-1}\sum_{t=1}^{T-1}
\frac{|E_t^{(l)} \cap E_{t+1}^{(l)}|}
{|E_t^{(l)} \cup E_{t+1}^{(l)}|}.
\]

TRC lies between 0 and 1. A value of 1 means adjacent frames activated identical expert sets; a value of 0 means the sets were disjoint.

The union definition can saturate when every frame contains many visual tokens and each token activates several experts. For example, most of a model's experts may appear at least once in every frame even though the distribution of routing mass changes substantially. Therefore, the original TRC will be reported for transparency but will not be the primary measure.

### Primary metric: TRC@M

First aggregate the router probabilities or normalized routing weights over the visual tokens in one frame:

\[
r_{t,l,e} =
\frac{\sum_i g_{t,l,i,e}}
{\sum_{e'}\sum_i g_{t,l,i,e'}},
\]

where \(g_{t,l,i,e}\) is the routing weight from visual token \(i\) to expert \(e\). Let \(E_{t,l;M}\) contain the \(M\) experts receiving the most aggregate routing mass. Then:

\[
\operatorname{TRC@M}^{(l)} = \frac{1}{T-1}\sum_{t=1}^{T-1}
\frac{|E_{t,l;M} \cap E_{t+1,l;M}|}
{|E_{t,l;M} \cup E_{t+1,l;M}|}.
\]

The primary setting is **M = 8**. We will also report **M = 4** and **M = 16** as sensitivity checks. Shared experts that are always active must be excluded because they artificially increase overlap.

### Weighted TRC

Weighted TRC preserves the entire frame-level routing distribution instead of converting it to a set:

\[
\operatorname{wTRC}^{(l)} = \frac{1}{T-1}\sum_{t=1}^{T-1}
\frac{\sum_e \min(r_{t,l,e},r_{t+1,l,e})}
{\sum_e \max(r_{t,l,e},r_{t+1,l,e})}.
\]

This weighted Jaccard measure is the main companion to TRC@8. Reporting both reveals whether an apparent result depends on hard top-expert selection.

## Experimental protocol

### Dataset and inputs

- Use 20 videos from the NExT-QA test set.
- Balance the sample across temporal, causal, and descriptive questions.
- Sample 8 frames uniformly from each video.
- Keep the real question and answer choices fixed while processing its frames.
- Use a fixed 384 × 384 image input and disable dynamic tiling for the frame-wise model, so token count does not vary between frames.
- Record the exact video IDs, frame indices, prompts, seed, preprocessing, and model revision in a manifest.

The initial DeepSeek-VL2 run processes frames independently. This is an adaptation of an image MoE-VLM and must not be described as native video inference. A native-video replication is included below.

### Sequence conditions

Each video produces the following conditions using the same number of frames and visual tokens:

| Condition | Construction | Purpose |
|---|---|---|
| Duplicate | Repeat one frame eight times | Deterministic positive control; expected to approach 1 |
| Chronological | Eight frames in their original order | Target temporally coherent condition |
| Shuffled | Randomly permute the same eight frames | Destroys adjacency while preserving frame content |
| Cross-clip | Assemble frames from unrelated videos | Strong negative temporal control |
| Router permutation | Permute expert identities within each layer while preserving expert frequencies | Estimates overlap not attributable to temporal identity consistency |

The qualitative expectation is `Duplicate > Chronological > Shuffled/Cross-clip`, but the measured result must be reported even if this ordering fails.

### Recorded measurements

For every model, condition, clip, and MoE layer, save:

- original union-based TRC;
- TRC@4, TRC@8, and TRC@16;
- weighted TRC;
- router entropy;
- per-expert token counts and normalized routing mass;
- expert load coefficient of variation;
- number of routed experts, router Top-k, visual-token count, and shared-expert count;
- scene-boundary versus within-scene TRC;
- video-QA correctness and confidence when the model supports the task.

Report the mean, median, interquartile range, and clip-bootstrap 95% confidence interval. Show per-layer curves as well as a clearly defined layer aggregate; do not hide layer dependence behind only one global number.

## Suitable models

| Model | Role in the experiment | Practical note |
|---|---|---|
| `deepseek-ai/deepseek-vl2-tiny` | Low-cost development model with existing router instrumentation; establishes that the implementation and controls work | Suitable for the first Kaggle run. Frame-wise evaluation is not native video inference. Exclude its always-active shared experts from TRC. |
| `OpenGVLab/InternVL3_5-30B-A3B` | Bridge to FastMMoE because FastMMoE evaluates the InternVL3.5 MoE family; also provides video preprocessing | Requires substantially more GPU memory than a Kaggle T4. Use after the small-model protocol is stable. |
| `Qwen/Qwen3-VL-30B-A3B-Instruct` | Native-video MoE replication for external validity | Strongest conceptual check that TRC is not an artifact of independent frame processing, but also a large-GPU experiment. Router extraction requires model-specific hooks. |

Recommended execution order:

1. Debug and preregister the protocol on DeepSeek-VL2-Tiny.
2. Run one native-video model, preferably Qwen3-VL-30B-A3B-Instruct.
3. Run InternVL3.5-30B-A3B when performing the direct FastMMoE comparison.

Official references:

- [NExT-QA project repository](https://github.com/doc-doc/NExT-QA)
- [DeepSeek-VL2-Tiny model](https://huggingface.co/deepseek-ai/deepseek-vl2-tiny)
- [InternVL3.5-30B-A3B model](https://huggingface.co/OpenGVLab/InternVL3_5-30B-A3B)
- [Qwen3-VL-30B-A3B-Instruct model](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct)
- [FastMMoE implementation](https://github.com/MindVLA-Team/FastMMoE)

## How TRC will be validated

Calling TRC a “good metric” requires several distinct checks:

### Level 1: Measurement validity

- Duplicate frames produce nearly identical routing and high TRC.
- TRC@M does not trivially saturate like the union metric.
- Results are reproducible under a fixed seed and model revision.
- Shared experts, padding tokens, text tokens, and visual tokens are handled explicitly.

### Level 2: Control sensitivity

- Chronological TRC is higher than shuffled and cross-clip TRC.
- Observed TRC is higher than the within-layer expert-permutation baseline.
- Scene-boundary pairs have lower TRC than within-scene pairs.
- The conclusion is stable for M = 4, 8, and 16 and under a matched visual-token budget.

Passing Levels 1 and 2 supports describing TRC as a nontrivial temporal-routing diagnostic.

### Level 3: Task relevance

Across clips and experimental conditions, test whether TRC is associated with video-QA correctness or confidence. Use clip-level confidence intervals and report effect sizes, not only a correlation p-value. Control for scene changes, token count, router entropy, and expert-load imbalance.

Passing this level supports an association between routing consistency and task performance. It still does not demonstrate that routing consistency causes higher accuracy.

### Level 4: Generality

Repeat the control-sensitive result on at least two model families, including one native-video MoE. Only then can the paper claim that TRC generalizes beyond one architecture and one frame-processing setup.

## FastMMoE extension

FastMMoE is a compression method, not a standalone model. A direct comparison should therefore use one supported MoE-VLM under matched conditions:

1. uncompressed baseline;
2. ordinary visual-token pruning;
3. FastMMoE routing-aware pruning;
4. matched token-retention ratios and identical videos, prompts, decoding, and seeds.

For each condition, compare task accuracy, retained-token ratio, latency, TRC@8, weighted TRC, router entropy, and expert-load balance. This can test whether FastMMoE preserves temporal routing structure better than ordinary pruning at comparable efficiency.

This extension must distinguish three possible outcomes:

- accuracy and TRC are both preserved;
- accuracy is preserved while TRC changes, suggesting TRC is not necessary for that setting;
- TRC is preserved while accuracy falls, showing that TRC alone is insufficient.

## Allowed conclusions

If only Levels 1 and 2 pass, use language such as:

> TRC is a control-sensitive diagnostic of adjacent-frame routing consistency in the evaluated models.

Do not claim that:

- higher TRC necessarily improves video understanding;
- lower TRC always indicates harmful routing;
- TRC is a validated training objective;
- compression-induced TRC changes cause accuracy changes;
- results from independent image frames automatically generalize to native video MoE models.

High TRC can also result from expert collapse, and low TRC can reflect a meaningful scene transition. TRC must therefore be interpreted together with routing entropy, expert-load statistics, scene boundaries, task accuracy, and the control conditions.

## Planned outputs

- `manifest.csv`: selected videos, questions, frame indices, and experimental settings;
- `routing_records/`: compact per-layer routing summaries;
- `trc_per_clip.csv`: all TRC variants and controls by clip and layer;
- `summary.csv`: aggregate statistics and bootstrap confidence intervals;
- `figures/`: per-layer condition curves, scene-boundary comparison, and TRC-versus-accuracy plots;
- `RESULTS.md`: exact findings, failed expectations, limitations, and paper-ready claim wording.

