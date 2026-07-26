# vgg19-r250-s0-alllayers-linear

> **Metric change (2026-07-26).** `logness` was removed; the headline
> statistic is now `λ`, the exponent of `D = a + b·(c^λ − 1)/λ` — **0 is the
> log law, 1 linear in contrast**. Any `logness` value in the prose below is
> on the retired scale. This run's `result.json` carries `lambda`,
> `lambda_ci` and `lambda_r2`, re-fitted from the committed surfaces, and is
> the authority where the two disagree. See `wiki/Results.md`.

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

**The picture holds.** Mean **|Δλ|** across the 45 layers is **0.045**, against
a profile spanning ~2.7 from conv stack to output. `prob` moves +0.165 → +0.180;
the crossing ReLU is still `classifier.4` (+0.010 → +0.086). Individual shifts
range −0.049 to +0.142.

The sawtooth is reproduced layer for layer, not just in aggregate: **44 of 44**
consecutive steps move in the same direction on both grids. (The sawtooth is
this checkpoint's — `IMAGENET1K_V1` alternates conv/ReLU by −0.155/+0.166, the
converted Caffe run does not alternate at all. See `wiki/Results.md`.) That is the
strongest form this control can take, and it is now unanimous — the shape, not
only its summary, is a property of the network rather than of the sampling.

λ is grid-free by construction, so this is a comparison of two measurements
rather than of two rulers. That was not true of either `logness` definition, and
it is why this control had to be restated twice.

> **Retracted, left as a record.** The first write-up claimed the shift was
> negative at *every one* of the 37 conv-stack taps (mean −0.041, range −0.056
> to −0.010), read as a real bias toward "linear" from even sampling. It was an
> artifact of comparing raw `R²_log − R²_lin` across two grids whose ceilings
> differed (0.264 log-spaced vs 0.294 linear) — two different rulers. Restated
> on the residual-ratio form it became −0.092 mean, range −0.178 to +0.107, not
> uniform; on λ it is −0.049 to +0.142. The headline (the profile survives the
> grid change) has stood throughout.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-r250-s0-alllayers-linear --notes Same grid endpoints sampled evenly instead of geometrically. Tests whether the logness depth profile is a property of VGG-19 or of the log-spaced contrast grid. --figures out/ --contrasts linear --layers all
```

Code: `aae17b04d27f`. Weights: torchvision vgg19 IMAGENET1K_V1.
