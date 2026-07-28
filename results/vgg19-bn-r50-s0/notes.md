# vgg19-bn-r50-s0

`vgg19_bn`, 50 reps/cell, best mean R² 0.928 at `features.26`.

## What this run was for

Reps companion to vgg19-bn-r250-s0: separates a real locally-linear response from an empty tap, as the vgg19 r50/r250 pair did (only features.0 was on the floor there).

## What it showed

**Five of 61 taps are on the floor, and the second one is the interesting
case.** Against [`vgg19-bn-r250-s0`](../vgg19-bn-r250-s0/notes.md):

| tap | kind | D(50)/D(250) | noise fraction |
|---|---|---|---|
| `features.0` | Conv2d | 2.219 | **98.6%** |
| `features.1` | **BatchNorm2d** | **2.227** | **99.3%** |
| `features.2` | ReLU | 1.949 | 76.7% |
| `features.3` | Conv2d | 1.928 | 75.1% |
| `features.4` | BatchNorm2d | 1.909 | 73.6% |
| `features.5` | ReLU | 1.424 | 34.3% |
| `features.12`+ | — | ≤ 1.16 | ≤ **12.9%** |
| `features.26`, `classifier.4`, `logits`, `prob` | — | ≤ 1.010 | ≤ **0.8%** |

`features.1` is a **BatchNorm layer reading pure noise**, which is exactly what
the metric's floor predicts and had never been demonstrated on anything but a
first convolution: BN in eval mode is per-channel affine, so composed with
conv1 it is still affine in the input, its population D is identically zero,
and what a finite run measures is 1/√reps sampling noise. The floor is a
property of *affineness*, not of being the first layer.

The headline taps are clean to under 1%, so
[the parent run's](../vgg19-bn-r250-s0/notes.md) λ = −0.268 at `prob` and
−0.071 conv-stack median are real measurements. λ at `prob` moves −0.268 →
−0.328 across the 5× rep change, inside the CI.

## Reproduce

```
run.py --model vgg19_bn --reps 50 --seed 0 --save-run results/vgg19-bn-r50-s0 --notes Reps companion to vgg19-bn-r250-s0: separates a real locally-linear response from an empty tap, as the vgg19 r50/r250 pair did (only features.0 was on the floor there). --figures out/ --layers all
```

Code: `f9ab3861976a`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
