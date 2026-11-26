# Implementation notes

`run.py` first intersects the NExT-QA annotations with the `.mp4` files actually present in the raw-video repository. It then downloads only the selected videos, uniformly samples eight frames, and sends all eight images with the multiple-choice question to DeepSeek-VL2-Tiny. Text between image placeholders makes each frame's visual-token block identifiable. The cross-clip control draws its eight positions from eight different videos rather than using a coherent sequence from one replacement video.

A forward hook on every `MoEGate` records routed expert IDs and weights. The script immediately reduces these token-level values into two small arrays:

- `mass[frame, layer, expert]`: normalized routing mass;
- `counts[frame, layer, expert]`: number of routed visual-token assignments.

Only routed experts are included; DeepSeek's always-active shared experts are not emitted by `MoEGate` and therefore do not inflate TRC.

`analyze.py` computes adjacent-frame Jaccard overlap for the union metric and the top 4, 8, and 16 experts by routing mass. Weighted TRC is weighted Jaccard over the complete routing-mass vectors. The router-permutation control independently relabels experts for each frame and layer while preserving every frame's routing distribution; its reported result is averaged over 100 permutations.

Accuracy is generated only for the chronological sequence. The TRC-accuracy table compares clip-level chronological TRC between correct and incorrect questions. This establishes association, not causation.
