# Experiment 1: Results, Figures, and Table Guidance

## Experimental scope

DeepSeek-VL2-Tiny was evaluated on 1,000 paired ScienceQA examples. The unpruned baseline accuracy was **71.6%**. Each pruning condition retained the indicated proportion of visual tokens. Random retention has one reported seed (`seed=0`), so it is a matched control rather than a multi-seed estimate.

## Recommended main-paper table

Use the following as the main quantitative table. Accuracy changes are paired with the unpruned prediction for the same examples; intervals are 95% bootstrap confidence intervals.

| Method | Retention | Accuracy | Δ accuracy (pp), 95% CI | RACR | Top-1 flips | Top-k changed | Δ load CV |
|---|---|---|---|---|---|---|---|
| L2 retention | 25% | 71.2 | -0.4 [-3.0, +2.1] | 29.1% | 41.1% | 84.0% | -0.061 |
| L2 retention | 50% | 69.9 | -1.7 [-4.1, +0.8] | 28.5% | 40.8% | 82.7% | -0.034 |
| L2 retention | 70% | 70.0 | -1.6 [-3.8, +0.5] | 26.5% | 38.6% | 81.4% | -0.021 |
| L2 retention | 75% | 69.2 | -2.4 [-4.5, -0.4] | 25.6% | 37.5% | 80.8% | -0.016 |
| Random retention | 25% | 60.2 | -11.4 [-14.1, -8.5] | 42.1% | 58.0% | 92.7% | +0.008 |
| Random retention | 50% | 65.1 | -6.5 [-9.0, -4.1] | 33.5% | 48.2% | 88.2% | +0.005 |
| Random retention | 70% | 66.5 | -5.1 [-7.3, -3.0] | 28.5% | 41.6% | 84.2% | +0.006 |
| Random retention | 75% | 67.8 | -3.8 [-5.9, -1.8] | 27.2% | 39.9% | 83.0% | +0.007 |

## Matched L2-versus-random comparison

Negative values in the first two columns mean L2 retention changes routing less than random retention at the same budget.

| Retention | L2 − random RACR (pp) | L2 − random Top-1 flips (pp) | L2 − random accuracy (pp) |
|---|---|---|---|
| 25% | -12.9 | -16.9 | +11.0 |
| 50% | -5.0 | -7.4 | +4.8 |
| 70% | -2.0 | -3.0 | +3.5 |
| 75% | -1.5 | -2.4 | +1.4 |

## Figures to include

1. **Figure 1 — Accuracy versus token retention.** Shows the task-performance cost as progressively fewer visual tokens are retained. Error bars are paired 95% bootstrap intervals for the accuracy change.
2. **Figure 2 — Routing disruption versus token retention.** Shows that the routing identities of retained tokens change substantially after pruning. RACR measures the fraction of baseline Top-k expert assignments not preserved; Top-1 flips isolate changes to the highest-ranked expert.
3. **Figure 3 — Matched L2 versus random pruning.** Shows the same-budget comparison. L2 consistently causes less routing reassignment than random retention, but the current results do not establish that this routing preservation causes the observed accuracy differences.

## Paper-ready interpretation

> Visual-token pruning substantially changes downstream MoE routing even among retained tokens. At matched token budgets, L2-based retention consistently yields lower routing assignment change and fewer Top-1 expert flips than random retention. For example, at 50% retention, L2 retention produces a RACR of 28.5% compared with 33.5% for random retention, while changing the dominant expert for 40.8% and 48.2% of retained token-layer units, respectively. Accuracy declines relative to the 71.6% unpruned baseline for both methods. These observations demonstrate co-occurring pruning, routing, and accuracy changes; they do not establish that routing disruption causes the accuracy loss.

## Definitions for the paper

- **Retention:** fraction of original visual tokens kept after pruning.
- **Accuracy change:** pruned accuracy minus the paired unpruned accuracy, in percentage points.
- **RACR (Routing Assignment Change Rate):** fraction of baseline Top-k expert assignments for retained token-layer units that is not preserved after pruning. Zero means the same Top-k assignments; higher values mean more reassignment.
- **Top-1 flips:** fraction of retained token-layer units whose highest-ranked expert differs from the unpruned run.
- **Top-k changed:** fraction of units whose complete Top-k expert set differs from the unpruned run.
- **Load-CV change:** change in the coefficient of variation of aggregate expert assignment counts. Negative values indicate less uneven aggregate expert load than the baseline.

## Scope note

This is a single-model, static-image QA experiment. Random pruning is currently represented by one seed, so claims about L2 versus random pruning should be described as descriptive results in this evaluated setting, not as a general result across random draws or models.
