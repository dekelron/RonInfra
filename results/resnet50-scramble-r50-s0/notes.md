# resnet50-scramble-r50-s0

`resnet50`, 50 reps/cell, best mean R² 0.749 at `conv1`.

## What this run was for

Reps companion to resnet50-r250-s0. Load-bearing here rather than routine: the whole residual-stream prediction is that lambda ~ 1 deep in the net is the floor, and only D(50)/D(250) against sqrt(5) can tell that from a real locally-linear response.

## What it showed

Confirms [the parent run](../resnet50-scramble-r250-s0/notes.md) is measuring
nothing at `prob`: λ moves −0.122 → **+0.028** across the rep change and the
95% interval opens to **[−3.00, +4.00]** — the entire search range, at λ-R²
0.460. That is the property λ was adopted for; pure noise says so instead of
returning a confident number.

The scramble decalibrates BatchNorm rather than degrading the weights. Do not
quote a trained-minus-scrambled gap for this architecture until the control
scrambles the running statistics alongside the weights.

## Reproduce

```
run.py --model resnet50 --reps 50 --seed 0 --save-run results/resnet50-scramble-r50-s0 --notes Reps companion to resnet50-r250-s0. Load-bearing here rather than routine: the whole residual-stream prediction is that lambda ~ 1 deep in the net is the floor, and only D(50)/D(250) against sqrt(5) can tell that from a real locally-linear response. --figures out/ --layers all --scramble
```

Code: `7bdf878d43c4`. Weights: torchvision resnet50 IMAGENET1K_V1.
