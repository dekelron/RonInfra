# vgg19-scramble-r250-s0

`vgg19`, 250 reps/cell, best mean R² 0.924 at `features.19`.

## What this run was for

The control for [`vgg19-r250-s0`](../vgg19-r250-s0/notes.md): the same
`IMAGENET1K_V1` weights loaded and verified, then permuted within each layer, so
the pair isolates learned organisation from architecture plus softmax. Same
seed, same grid, run in parallel on the other matrix job.

## What it showed

| Layer | mean R² | pooled R² | spacing CV |
|---|---|---|---|
| `features.0` | 0.604 | 0.604 | 1.440 |
| **`features.19`** | **0.924** | 0.880 | 0.659 |
| `classifier.3` | 0.760 | 0.540 | 4.073 |
| `logits` | 0.768 | 0.545 | 3.542 |
| `prob` | 0.768 | 0.545 | 3.544 |

**Do not quote the 0.768 without its spacing CV.** At the three late taps the CV
is 3.5–4.1, against 0.59–0.89 for the trained net. The per-frequency fits fan
out: high-frequency traces climb steeply while low ones stay near flat, so R²
0.768 is averaging over frequencies that behave differently, not describing an
even log ladder. R² is the wrong summary for this column — this is why rule 4
in [CLAUDE.md](../../CLAUDE.md) says to state the disagreement rather than pick
a number.

Two things worth flagging:

- **This control beats the trained net at the early and middle taps** (0.604 vs
  0.548, 0.924 vs 0.869). Whatever the log response is, at those depths it is
  not a product of learning.
- **The control now has three disagreeing values**: 0.428 here at 50 reps on
  converted Caffe weights, 0.768 at 250 reps on `IMAGENET1K_V1`, and 0.60
  documented in [Method](../../wiki/Method.md). No scramble seed has been
  repeated, so none of the three is yet known to be stable. Vary the seed before
  trusting any of them.

Cost note: this job took 116.5 min wall against 69.7 min for the same work in
run 30148332262 — identical code and grid, 1.67× slower runner. Treat the
per-run wall times in [Running](../../wiki/Running.md) as a range.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0 --notes vgg19, reps=250, seed=0, weights scrambled; GitHub-hosted runner --figures out/ --scramble
```

Code: `7067d624aaa7`. Weights: torchvision vgg19 IMAGENET1K_V1.
