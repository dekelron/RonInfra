# alexnet-scramble-r50-s0

`alexnet`, 50 reps/cell, best mean R² 0.944 at `features.9`.

## What this run was for

Reps companion to alexnet-r250-s0: D(50)/D(250) against sqrt(5)=2.236 says which of the 21 taps carry signal and which sit on the metric's noise floor.

## What it showed

Reps companion to
[`alexnet-scramble-r250-s0`](../alexnet-scramble-r250-s0/notes.md). `prob` λ
+0.015 → **+0.005**, mean R² 0.865 → 0.857: the control is rep-invariant, so
its gap from the trained net (0.963 vs 0.865, i.e. **0.098**) is a property of
the weights and not of sampling.

That gap is far narrower than VGG-19's — Caffe gives 0.980 against 0.429, a gap
of 0.55. Scrambling hurts AlexNet much less than it hurts VGG-19. Not
interpreted here beyond stating it; a single permutation is a single sample.

## Reproduce

```
run.py --model alexnet --reps 50 --seed 0 --save-run results/alexnet-scramble-r50-s0 --notes Reps companion to alexnet-r250-s0: D(50)/D(250) against sqrt(5)=2.236 says which of the 21 taps carry signal and which sit on the metric's noise floor. --figures out/ --layers all --scramble
```

Code: `f9ab3861976a`. Weights: torchvision alexnet IMAGENET1K_V1.
