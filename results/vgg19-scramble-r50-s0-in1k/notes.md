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

This is one scramble seed. A sweep over seeds 1–3 at these settings is what
decides whether 0.760 is a property of the scrambled net or of this particular
permutation.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-scramble-r50-s0-in1k --notes IMAGENET1K_V1 at 50 reps: the cell separating weight lineage from repetition count as the cause of the r50/r250 disagreement --figures out/ --scramble
```

Code: `d446b36e387a`. Weights: torchvision vgg19 IMAGENET1K_V1.
