# resnet50-r50-s0

`resnet50`, 50 reps/cell, best mean R² 0.957 at `layer2.3.relu@2`.

## What this run was for

Reps companion to resnet50-r250-s0. Load-bearing here rather than routine: the whole residual-stream prediction is that lambda ~ 1 deep in the net is the floor, and only D(50)/D(250) against sqrt(5) can tell that from a real locally-linear response.

## What it showed

Reps companion to [`resnet50-r250-s0`](../resnet50-r250-s0/notes.md).

| tap | D(50)/D(250) | noise fraction |
|---|---|---|
| `conv1` | 2.218 | **98.5%** |
| `bn1` (BatchNorm) | 2.220 | **98.7%** |
| `relu` | 1.834 | 67.5% |
| everything else (157 taps) | ≤ 1.063 | ≤ **5.1%** |
| `logits`, `prob` | ≈ 1.000 | ≈ **0%** |

Three taps on or near the floor, and the first two are the affine prefix —
`conv1` followed by `bn1`, both reading essentially pure 1/√reps noise. This is
the third architecture to show it and the second to show a **BatchNorm layer**
on the floor, after `vgg19_bn`'s `features.1`.

Everything else carries signal to within 5.1%, which is what kills the
residual-stream hypothesis the parent run was testing: if the identity path
kept an affine component alive, deep taps would fall with repetition count.
None do. λ at `prob` is −0.223 at both rep counts.

## Reproduce

```
run.py --model resnet50 --reps 50 --seed 0 --save-run results/resnet50-r50-s0 --notes Reps companion to resnet50-r250-s0. Load-bearing here rather than routine: the whole residual-stream prediction is that lambda ~ 1 deep in the net is the floor, and only D(50)/D(250) against sqrt(5) can tell that from a real locally-linear response. --figures out/ --layers all
```

Code: `7bdf878d43c4`. Weights: torchvision resnet50 IMAGENET1K_V1.
