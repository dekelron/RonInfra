# vgg19-scramble-r250-s0

`vgg19`, 250 reps/cell, best mean R² 0.924 at `features.19`.

## What this run was for

vgg19, reps=250, seed=0, weights scrambled; GitHub-hosted runner

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0 --notes vgg19, reps=250, seed=0, weights scrambled; GitHub-hosted runner --figures out/ --scramble
```

Code: `7067d624aaa7`. Weights: torchvision vgg19 IMAGENET1K_V1.
