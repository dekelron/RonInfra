# vgg19-scramble-r50-s0-in1k

`vgg19`, 50 reps/cell, best mean R² 0.924 at `features.19`.

## What this run was for

IMAGENET1K_V1 at 50 reps: the cell separating weight lineage from repetition count as the cause of the r50/r250 disagreement

## What it showed

The control for [`vgg19-r50-s0-in1k`](../vgg19-r50-s0-in1k/notes.md), and the
fourth measurement of a value the repo has never pinned down.

| Layer | mean R² | pooled R² | spacing CV |
|---|---|---|---|
| `features.0` | 0.615 | 0.615 | 1.423 |
| **`features.19`** | **0.924** | 0.879 | 0.668 |
| `classifier.3` | 0.752 | 0.531 | 4.150 |
| `logits` | 0.760 | 0.536 | 3.697 |
| `prob` | 0.760 | 0.536 | 3.699 |

The control series at `prob` now reads:

| | value |
|---|---|
| `wiki/Method.md`, documented | 0.60 |
| converted Caffe, 50 reps | 0.428 |
| `IMAGENET1K_V1`, 50 reps (this run) | **0.760** |
| `IMAGENET1K_V1`, 250 reps | 0.768 |

Which splits cleanly by checkpoint, not by sampling: the two `IMAGENET1K_V1`
values differ by 0.008 across a 5× change in reps, while the Caffe value sits
0.33 below both. Same conclusion as the trained run — lineage, not reps.

**Still do not quote 0.760 on its own.** Its spacing CV is 3.70 at `prob` and
4.15 at `classifier.3`, against 0.67–0.97 for the trained net at the same
settings. The per-frequency fits fan out rather than describing one log ladder,
so R² is summarising frequencies that disagree with each other.

### The sweep, and what it settled

Seeds 1–3 were run at identical settings. `prob` mean R²:

| seed | R² | spacing CV | trained − scrambled |
|---|---|---|---|
| 0 | 0.760 | 3.70 | 0.153 |
| 1 | **0.863** | 1.42 | 0.050 |
| 2 | 0.704 | 2.89 | 0.209 |
| 3 | **0.693** | 1.24 | 0.220 |

**Spread 0.169, sd 0.078 — so no single control value is quotable, including
0.760.** Worse for the comparison it exists to support: the learned contribution
at `prob` ranges 0.050 to 0.220 depending only on which permutation was drawn.
A one-seed control does not measure it, and the four values straddle the 0.60 in
[Method](../../wiki/Method.md) rather than confirming or refuting it.

The spacing CV moves with it, 1.24 to 4.15, and not in step with R²: seed 1 has
both the highest R² and a low CV (a genuinely straighter response), while seed 3
has the lowest R² *and* a low CV. So the scrambled net's behaviour is not one
phenomenon with a noisy estimate — different permutations differ in kind.

**Caveat on what was varied.** At the code version these ran on, `--seed` drove
both the weight permutation and the orientation/phase draws, so the sweep moved
both together. Sampling is unlikely to explain much of it: at a fixed seed,
going 50 → 250 reps moved this value only 0.760 → 0.768. But that is an
argument, not an isolation. `--scramble-seed` now exists precisely so the next
sweep can hold the draws fixed and vary only the permutation.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-scramble-r50-s0-in1k --notes IMAGENET1K_V1 at 50 reps: the cell separating weight lineage from repetition count as the cause of the r50/r250 disagreement --figures out/ --scramble
```

Code: `d446b36e387a`. Weights: torchvision vgg19 IMAGENET1K_V1.
