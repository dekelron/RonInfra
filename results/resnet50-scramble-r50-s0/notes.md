# resnet50-scramble-r50-s0

`resnet50`, 50 reps/cell, best mean R² 0.749 at `conv1`.

## What this run was for

Reps companion to resnet50-r250-s0. Load-bearing here rather than routine: the whole residual-stream prediction is that lambda ~ 1 deep in the net is the floor, and only D(50)/D(250) against sqrt(5) can tell that from a real locally-linear response.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model resnet50 --reps 50 --seed 0 --save-run results/resnet50-scramble-r50-s0 --notes Reps companion to resnet50-r250-s0. Load-bearing here rather than routine: the whole residual-stream prediction is that lambda ~ 1 deep in the net is the floor, and only D(50)/D(250) against sqrt(5) can tell that from a real locally-linear response. --figures out/ --layers all --scramble
```

Code: `7bdf878d43c4`. Weights: torchvision resnet50 IMAGENET1K_V1.
