# vgg19-scramble-r50-s0-in1k

`vgg19`, 50 reps/cell, best mean R² 0.924 at `features.19`.

## What this run was for

IMAGENET1K_V1 at 50 reps: the cell separating weight lineage from repetition count as the cause of the r50/r250 disagreement

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-scramble-r50-s0-in1k --notes IMAGENET1K_V1 at 50 reps: the cell separating weight lineage from repetition count as the cause of the r50/r250 disagreement --figures out/ --scramble
```

Code: `d446b36e387a`. Weights: torchvision vgg19 IMAGENET1K_V1.
