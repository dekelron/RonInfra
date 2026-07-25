# vgg19-r250-s0

`vgg19`, 250 reps/cell, best mean R² 0.928 at `classifier.3`.

## What this run was for

The grid [Method](../../wiki/Method.md) actually specifies — `--reps 250`, all
14 contrasts × 8 frequencies — on the canonical torchvision checkpoint rather
than the converted Caffe weights used by
[`vgg19-r50-s0`](../vgg19-r50-s0/notes.md). It closes two threads at once: the
full grid had never been run, and the earlier result came from weights of the
same lineage but not verified identical to `IMAGENET1K_V1`.

## What it showed

| Layer | mean R² | pooled R² | spacing CV |
|---|---|---|---|
| `features.0` | 0.548 | 0.331 | 1.638 |
| `features.19` | 0.869 | 0.741 | 0.906 |
| **`classifier.3`** | **0.928** | 0.863 | 0.592 |
| `logits` | 0.878 | 0.811 | 0.886 |
| `prob` | 0.917 | 0.892 | 0.667 |

Three disagreements with the documented expectation:

1. **`prob` reaches 0.917, not 0.98.**
2. **R² does not peak at `prob`.** It peaks at `classifier.3` (fc7) and *dips*
   at `logits` before the softmax lifts it again. "Highest at `prob`" does not
   survive the full grid.
3. Against [the scrambled control](../vgg19-scramble-r250-s0/notes.md), the
   trained net is **worse** at `features.0` (0.548 vs 0.604) and `features.19`
   (0.869 vs 0.924). The learned contribution is confined to the classifier end,
   and there the gap is 0.149 — far short of the documented 0.98 − 0.60 = 0.38.

What this does **not** settle: against the r50 run, weight lineage *and*
repetition count changed together, so neither 0.976 → 0.917 nor 0.428 → 0.768 is
attributable to one alone. More reps should if anything raise R² by reducing
noise, which points at the weights — an inference, not a measurement.
`IMAGENET1K_V1` at `--reps 50` is the single cell that separates them.

Reproducibility: run
[30148332262](https://github.com/dekelron/RonInfra/actions/runs/30148332262)
measured the same grid at the same seed on different runner hardware and
produced this table to every digit shown. The scrambled control's spacing CVs
differ in the fourth significant figure, which is floating-point
non-associativity across microarchitectures, not a methodological difference.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-r250-s0 --notes vgg19, reps=250, seed=0; GitHub-hosted runner --figures out/
```

Code: `7067d624aaa7`. Weights: torchvision vgg19 IMAGENET1K_V1.
