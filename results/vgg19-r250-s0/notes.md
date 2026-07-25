# vgg19-r250-s0

`vgg19`, 250 reps/cell, best mean R² 0.928 at `classifier.3`.

## What this run was for

vgg19, reps=250, seed=0; GitHub-hosted runner

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-r250-s0 --notes vgg19, reps=250, seed=0; GitHub-hosted runner --figures out/
```

Code: `7067d624aaa7`. Weights: torchvision vgg19 IMAGENET1K_V1.
