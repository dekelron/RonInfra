# vgg19-scramble-r50-s2-in1k

`vgg19`, 50 reps/cell, best mean R² 0.920 at `features.19`.

## What this run was for

Scramble-seed sweep: seed 2 at IMAGENET1K_V1/50 reps, to test whether the control value is stable across seeds

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 2 --save-run results/vgg19-scramble-r50-s2-in1k --notes Scramble-seed sweep: seed 2 at IMAGENET1K_V1/50 reps, to test whether the control value is stable across seeds --figures out/ --scramble
```

Code: `d446b36e387a`. Weights: torchvision vgg19 IMAGENET1K_V1.
