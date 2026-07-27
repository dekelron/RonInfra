# vgg19-r50-s0-alllayers-in1k

`vgg19`, 50 reps/cell, best mean R² 0.921 at `classifier.4`.

## What this run was for

45 taps at 50 reps, IMAGENET1K_V1. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric.

## What it showed

Against [`vgg19-r250-s0-alllayers-fixed`](../vgg19-r250-s0-alllayers-fixed/notes.md).

**1 of 45** taps on the floor — `features.0` at D(50)/D(250) = **2.221**,
98% noise. `features.1` and `features.2` are the only partial cases anywhere
in the four r50 runs, at 31% and 36%, and they are exactly where the two
orderings disagree most: λ **+1.67** against λ_mod **+1.01** at `features.1`.
The primary metric reports a wildly supralinear exponent there that is its
own sampling noise. Outside `features.0/1/2` the max noise fraction is
**3.1%**.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-in1k --notes 45 taps at 50 reps, IMAGENET1K_V1. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric. --figures out/ --layers all
```

Code: `564e392d056c`. Weights: torchvision vgg19 IMAGENET1K_V1.
