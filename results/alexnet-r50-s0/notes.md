# alexnet-r50-s0

`alexnet`, 50 reps/cell, best mean R² 0.959 at `prob`.

## What this run was for

Reps companion to alexnet-r250-s0: D(50)/D(250) against sqrt(5)=2.236 says which of the 21 taps carry signal and which sit on the metric's noise floor.

## What it showed

**Only `features.0` is on the floor — 1 of 21.** Against
[`alexnet-r250-s0`](../alexnet-r250-s0/notes.md):

| tap | D(50)/D(250) | noise fraction | reading |
|---|---|---|---|
| `features.0` | **2.240** | 100% | the floor (√5 = 2.236) |
| everything else | ≤ 1.022 | ≤ **1.8%** | signal |

Cleaner than VGG-19, where `features.1`/`.2` were partial cases at 31% and 36%.
AlexNet's conv1 has stride 4 and an 11×11 kernel, so the first ReLU is already
well clear of the floor.

λ at `prob` moves +0.053 → **+0.043** across the 5× rep change and mean R²
0.963 → 0.959, i.e. the headline is rep-invariant to within its own CI. The
declining conv profile is real, not an artifact of repetition count.

## Reproduce

```
run.py --model alexnet --reps 50 --seed 0 --save-run results/alexnet-r50-s0 --notes Reps companion to alexnet-r250-s0: D(50)/D(250) against sqrt(5)=2.236 says which of the 21 taps carry signal and which sit on the metric's noise floor. --figures out/ --layers all
```

Code: `f9ab3861976a`. Weights: torchvision alexnet IMAGENET1K_V1.
