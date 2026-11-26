# Experiment 3 implementation

This experiment tests whether **within-question changes** in temporal routing consistency track changes in multiple-choice answer quality. It uses DeepSeek-VL2-Tiny in FP16 on a single Kaggle T4 and presents eight sampled video frames as a multi-image sequence.

## Controlled intervention

Every question uses the same video, eight sampled frames, answer choices, and neutral prompt. Only the presented frame order changes. A fixed permutation bank contains one chronological ordering, two mild corruptions, two medium corruptions, two severe corruptions, and one complete reversal. Mild, medium, and severe permutations are selected before inference using normalized Kendall distance and retained unordered adjacency. The reverse condition is excluded from the primary association because it retains every unordered adjacent pair.

The primary cohort is temporal questions. Causal questions are secondary and descriptive questions are a negative-control cohort. The default 256-question manifest contains 128 temporal, 64 causal, and 64 descriptive questions from unique videos.

## T4 execution path

The runner performs one prompt prefill per question-condition pair. Forward hooks capture the routed expert IDs and weights from every `MoEGate`, while the same forward pass supplies next-token logits for option scoring. DeepSeek answer letters are normally single tokens. A teacher-forced continuation fallback handles a tokenizer that represents a letter using multiple tokens.

The model remains in FP16 and eager attention is used. Only compact frame-by-layer-by-expert routing mass and assignment counts are saved. Token-level router tensors are immediately discarded. Every route and prediction file is written independently and atomically, so rerunning the same command resumes missing work.

Some NExT-QA MP4 files report a nominal frame count whose final frames cannot be decoded. For each uniformly requested index, the sampler deterministically searches outward and uses the nearest unique decodable frame, preferring the earlier frame when distances tie. Both requested and actual indices and every recovery offset are saved. A single missing encoded frame therefore does not terminate the run or change the eight-frame input budget.

## Measurements

The primary routing measurement is TRC@8 averaged over all routed MoE layers. Weighted TRC is the prespecified robustness measurement. The primary performance outcome is the correct-option margin:

\[
\log P(y_\text{correct}) - \max_{j \ne \text{correct}} \log P(y_j).
\]

Binary correctness is secondary. The analysis subtracts each question's chronological baseline and estimates the association between change in TRC and change in performance. Confidence intervals resample complete question clusters. A second slope controls for mild, medium, and severe perturbation level. Complete reversal is reported separately.

## Scope

DeepSeek-VL2-Tiny is an image/multi-image MoE-VLM. This implementation does not perform native-video inference. The intervention supports a within-model association claim, not a causal claim that routing changes produce accuracy changes.
