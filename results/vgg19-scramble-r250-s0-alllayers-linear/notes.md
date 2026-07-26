# vgg19-scramble-r250-s0-alllayers-linear

`vgg19`, 250 reps/cell, best mean R² 0.921 at `features.20`.

## What this run was for

The control for the one methodological caveat that touched every `logness`
number: the default contrast grid is log-spaced, which is **not neutral**
between the two laws `logness` compares — it gives the log fit evenly spread
leverage while bunching the linear fit's points near zero. This run samples the
same endpoints evenly instead, so it differs from
[`vgg19-scramble-r250-s0-alllayers-fixed`](../vgg19-scramble-r250-s0-alllayers-fixed/notes.md)
in nothing but where the contrast axis is sampled.

## What it showed

**The picture holds.** Mean |Δ logness| across the 45 layers is **0.024**, with
**no** sign flips anywhere, and `prob` moves +0.120 → +0.102. The finding this
run exists to protect — that scrambled `IMAGENET1K_V1` crosses zero at
`features.16` and then holds ≈ +0.1, unlike scrambled Caffe which never crosses
— is unchanged: the crossing is still at `features.16` (+0.081 → +0.115).

The shift is negative on average in both halves (conv stack mean −0.021, range
−0.044 to +0.034; classifier mean −0.017, every tap negative), i.e. marginally
toward "linear", the direction even sampling predicts.

Unlike the trained net, step directions agree only 34/44 across the two grids —
**this is not a failure to reproduce.** After `features.16` this profile is flat
near +0.12, so its layer-to-layer steps are noise about a constant rather than a
shape; there is no sawtooth here to reproduce. The trained run's 44/44 is where
that comparison carries information.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0-alllayers-linear --notes Same grid endpoints sampled evenly instead of geometrically. Tests whether the logness depth profile is a property of VGG-19 or of the log-spaced contrast grid. --figures out/ --contrasts linear --layers all --scramble
```

Code: `aae17b04d27f`. Weights: torchvision vgg19 IMAGENET1K_V1.
