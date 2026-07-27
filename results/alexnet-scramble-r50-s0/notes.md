# alexnet-scramble-r50-s0

`alexnet`, 50 reps/cell, best mean R² 0.944 at `features.9`.

## What this run was for

Reps companion to alexnet-r250-s0: D(50)/D(250) against sqrt(5)=2.236 says which of the 21 taps carry signal and which sit on the metric's noise floor.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model alexnet --reps 50 --seed 0 --save-run results/alexnet-scramble-r50-s0 --notes Reps companion to alexnet-r250-s0: D(50)/D(250) against sqrt(5)=2.236 says which of the 21 taps carry signal and which sit on the metric's noise floor. --figures out/ --layers all --scramble
```

Code: `f9ab3861976a`. Weights: torchvision alexnet IMAGENET1K_V1.
