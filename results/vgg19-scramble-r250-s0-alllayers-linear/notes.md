# vgg19-scramble-r250-s0-alllayers-linear

`vgg19`, 250 reps/cell, best mean R² 0.921 at `features.20`.

## What this run was for

Same grid endpoints sampled evenly instead of geometrically. Tests whether the logness depth profile is a property of VGG-19 or of the log-spaced contrast grid.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0-alllayers-linear --notes Same grid endpoints sampled evenly instead of geometrically. Tests whether the logness depth profile is a property of VGG-19 or of the log-spaced contrast grid. --figures out/ --contrasts linear --layers all --scramble
```

Code: `aae17b04d27f`. Weights: torchvision vgg19 IMAGENET1K_V1.
