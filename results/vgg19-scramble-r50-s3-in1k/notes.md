# vgg19-scramble-r50-s3-in1k

`vgg19`, 50 reps/cell, best mean R² 0.843 at `features.19`.

## What this run was for

Scramble-seed sweep: seed 3 at IMAGENET1K_V1/50 reps, to test whether the control value is stable across seeds

## What it showed

`prob` mean R² = **0.693**, spacing CV 1.24. One of four permutations
measured at identical settings (VGG-19, `IMAGENET1K_V1`, 50 reps); see
[`vgg19-scramble-r50-s0-in1k`](../vgg19-scramble-r50-s0-in1k/notes.md) for the
series and what it means.

Against the trained net at the same settings (0.913), this permutation leaves a
gap of **0.220**. Across the four seeds that gap ranges 0.050–0.220, so no
single run of this kind measures "the" learned contribution.

Caveat on the seed: at the code version this ran on, `--seed` drove both the
weight permutation *and* the orientation/phase draws, so strictly this varies
both. `--scramble-seed` now separates them.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 3 --save-run results/vgg19-scramble-r50-s3-in1k --notes Scramble-seed sweep: seed 3 at IMAGENET1K_V1/50 reps, to test whether the control value is stable across seeds --figures out/ --scramble
```

Code: `d446b36e387a`. Weights: torchvision vgg19 IMAGENET1K_V1.
