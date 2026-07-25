# vgg19-scramble-r250-s0-alllayers-fixed

`vgg19`, 250 reps/cell, best mean R² 0.924 at `features.20`.

## What this run was for

All 43 taps on the fixed hook (pre-activation taps no longer overwritten by in-place ReLU). IMAGENET1K_V1 half of the checkpoint comparison.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0-alllayers-fixed --notes All 43 taps on the fixed hook (pre-activation taps no longer overwritten by in-place ReLU). IMAGENET1K_V1 half of the checkpoint comparison. --figures out/ --layers all --scramble
```

Code: `d2ebd925b086`. Weights: torchvision vgg19 IMAGENET1K_V1.
