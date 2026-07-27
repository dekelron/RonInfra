# vgg19-scramble-r50-s0-alllayers-in1k

`vgg19`, 50 reps/cell, best mean R² 0.924 at `features.20`.

## What this run was for

45 taps at 50 reps, IMAGENET1K_V1. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric.

## What it showed

Control for [`vgg19-r50-s0-alllayers-in1k`](../vgg19-r50-s0-alllayers-in1k/notes.md).

**1 of 45** on the floor (`features.0`, **2.236**, 100% noise); max elsewhere
**1.8%**. Its λ ≈ +0.19 at R² 0.81 is the log-*looking* control that λ alone
cannot separate from the trained net — see wiki/Results.md. That reading is
unaffected by the floor: the taps carrying it are clean.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-scramble-r50-s0-alllayers-in1k --notes 45 taps at 50 reps, IMAGENET1K_V1. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric. --figures out/ --layers all --scramble
```

Code: `564e392d056c`. Weights: torchvision vgg19 IMAGENET1K_V1.
