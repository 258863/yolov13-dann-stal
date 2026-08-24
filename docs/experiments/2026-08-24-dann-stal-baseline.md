# DANN and STAL Experiment Baseline

## DANN Results

Evaluation uses `conf=0.40`. Values are the mean over three seeds.

| Method | TP | FP | FN | Miss rate |
| --- | ---: | ---: | ---: | ---: |
| No DANN | 55.67 | 1.67 | 3.33 | 5.65% |
| P5 | 56.33 | 2.00 | 2.67 | 4.52% |
| P3 | 56.67 | 3.67 | 2.33 | 3.95% |
| P2 | 58.00 | 2.00 | 1.00 | 1.69% |
| P2 + P3 | 56.67 | 2.33 | 2.33 | 3.95% |

P2 is the current best domain-alignment configuration. Compared with no
DANN, its mean FN decreases from `3.33` to `1.00`, a reduction of about 70%,
while FP increases slightly from `1.67` to `2.00`.

P2 + P3 does not improve on P2 alone. Its FN increases from `1.00` to `2.33`,
its miss rate increases from `1.69%` to `3.95%`, and its FP increases from
`2.00` to `2.33`. This is treated as negative transfer, so P4/P5 should not
be stacked onto the multi-layer branch without new evidence.

## Next STAL Comparison

The next experiment changes only the STAL switch:

| Experiment | Domain alignment | Assignment |
| --- | --- | --- |
| NoDANN-TAL | Disabled | `stal=False` |
| NoDANN-STAL | Disabled | `stal=True` |
| P2-TAL | P2 single-layer DANN | `stal=False` |
| P2-STAL | P2 single-layer DANN | `stal=True` |

The existing NoDANN-TAL and P2-TAL results may be reused only when the dataset
split, initialization, augmentation, image size, batch size, optimizer,
epochs, and evaluation code are unchanged. Otherwise all four experiments
must be rerun.

Run seed 42 first as a screening experiment. If STAL does not increase FN and
shows a useful improvement, run seeds 43 and 44 and report the three-seed
mean at `conf=0.40`. Compare TP, FP, FN, miss rate, and per-image prediction
visualizations on the same untouched validation set.
