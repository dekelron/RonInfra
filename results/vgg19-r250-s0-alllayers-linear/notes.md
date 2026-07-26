# vgg19-r250-s0-alllayers-linear

`vgg19`, 250 reps/cell, best mean R² 0.950 at `classifier.4`.

## What this run was for

The control for the one methodological caveat that touched every `logness`
number: the default contrast grid is log-spaced, which is **not neutral**
between the two laws `logness` compares — it gives the log fit evenly spread
leverage while bunching the linear fit's points near zero. This run samples the
same endpoints evenly instead, so it differs from
[`vgg19-r250-s0-alllayers-fixed`](../vgg19-r250-s0-alllayers-fixed/notes.md)
in nothing but where the contrast axis is sampled.

## What it showed

> Numbers below are on the `logness` definition adopted 2026-07-26. This run's
> `result.json` carries both it and the superseded `logness_r2diff`.

**The picture holds.** Mean |Δ logness| across the 45 layers is **0.090**,
against a profile spanning ~1.4 from conv stack to output, with **zero** sign
flips in 45. `prob` moves +0.466 → +0.540; the crossing ReLU is still
`classifier.4` (+0.477 → +0.594).

The sawtooth is reproduced very nearly layer for layer, not just in aggregate:
**43 of 44** consecutive steps move in the same direction on both grids. That is
the strongest form this control can take — the shape, not only its summary, is a
property of the network rather than of the sampling.

The residual shift averages −0.092 across the conv stack but ranges −0.178 to
+0.107, so it has no consistent direction, and it is well below the effects being
claimed.

> **Retracted.** The first write-up of this run claimed the shift was negative at
> *every one* of the 37 conv-stack taps (mean −0.041, range −0.056 to −0.010) and
> read that as a real bias toward "linear" from even sampling. It was an
> artifact: those were raw pre-2026-07-26 `logness` values compared across two
> grids whose ceilings differed (0.264 log-spaced vs 0.294 linear), so the
> comparison was between two different rulers. That statistic did not support a
> cross-grid comparison at all — the current one does, and under it the shift is
> not uniform. The headline (the profile survives the grid change) stands.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-r250-s0-alllayers-linear --notes Same grid endpoints sampled evenly instead of geometrically. Tests whether the logness depth profile is a property of VGG-19 or of the log-spaced contrast grid. --figures out/ --contrasts linear --layers all
```

Code: `aae17b04d27f`. Weights: torchvision vgg19 IMAGENET1K_V1.
