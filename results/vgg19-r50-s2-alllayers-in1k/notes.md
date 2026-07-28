# vgg19-r50-s2-alllayers-in1k

`vgg19`, 50 reps/cell, best mean R² 0.924 at `classifier.4`.

## What this run was for

Seed sweep for the per-frequency lambda structure at prob: seed 2 of 3, IMAGENET1K_V1. This series carries the weakest of the four dips (+0.22) and its 7 cyc/img point is the least determined anywhere (interval 1.57 wide).

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 2 --save-run results/vgg19-r50-s2-alllayers-in1k --notes Seed sweep for the per-frequency lambda structure at prob: seed 2 of 3, IMAGENET1K_V1. This series carries the weakest of the four dips (+0.22) and its 7 cyc/img point is the least determined anywhere (interval 1.57 wide). --figures out/ --layers all
```

Code: `f6f5837c5fb0`. Weights: torchvision vgg19 IMAGENET1K_V1.
