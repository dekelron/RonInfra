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

**The picture holds.** Mean |Δ logness| across the 45 layers is **0.037**, one
sign flip in 45 (`features.33`, +0.002 → −0.027, a layer already sitting on
zero), and `prob` moves +0.151 → +0.164. The crossing ReLU is still
`classifier.4` (+0.153 → +0.162).

The sawtooth is reproduced layer for layer, not just in aggregate: **44 of 44**
consecutive steps move in the same direction on both grids. That is the strongest
form this control can take — the shape, not only its summary, is a property of
the network rather than of the sampling.

The shift is small but **not random**. It is negative at *every one* of the 37
conv-stack taps (mean −0.041, range −0.056 to −0.010) and near zero across the
classifier (mean −0.005, mixed sign). The linear grid nudges conv-stack verdicts
marginally toward "linear", which is the direction expected from giving the
linear fit even leverage instead of bunching its points near zero. An order of
magnitude below the effects being claimed, so nothing changes — but it is a real
bias, recorded rather than rounded to zero.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-r250-s0-alllayers-linear --notes Same grid endpoints sampled evenly instead of geometrically. Tests whether the logness depth profile is a property of VGG-19 or of the log-spaced contrast grid. --figures out/ --contrasts linear --layers all
```

Code: `aae17b04d27f`. Weights: torchvision vgg19 IMAGENET1K_V1.
