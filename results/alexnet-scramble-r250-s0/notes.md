# alexnet-scramble-r250-s0

`alexnet`, 250 reps/cell, best mean R² 0.945 at `features.9`.

## What this run was for

AlexNet depth profile, 21 taps. Depth control: does the crossover to log need 33 layers, or is it carried by the classifier ReLU + softmax?

## What it showed

**The control separates on R², not on λ — the documented trap, on a second
architecture.** Scrambled `prob` returns λ **+0.015**, numerically *closer to
the log law* than the trained net's +0.053. Only the fit quality tells them
apart:

| | trained | scrambled |
|---|---|---|
| `prob` λ | +0.053 | +0.015 |
| `prob` λ R² | **0.985** | **0.889** |
| `prob` mean R² | **0.963** | **0.865** |
| peak tap | `prob` | `features.9` |

This is exactly why the repo's rule is to quote λ only with its R². λ alone
would have called the scrambled net *more* logarithmic than the trained one.

**The peak moves off `prob`**, to `features.9` at 0.945 — the structural claim
breaks even though the number at `prob` stays high, which is the more
informative half of the control.

**`prob` carries no information beyond `logits`.** r(logits, prob) =
**1.000000** with ratio 9.98e-04 ≈ 1/1000: with 1000 classes and unconfident
logits the softmax never leaves its affine regime, so Δprob = Δlogits/1000.
Trained gives r = 0.977. That reproduces the VGG-19 finding exactly on a
different architecture, and is the sharpest evidence that the softmax's
contribution requires trained, confident logits rather than being automatic.

**0.865 sits with the `IMAGENET1K_V1` controls, not the Caffe ones** (0.693–0.863
vs 0.422–0.516), and above the 0.60 in [Method](../../wiki/Method.md). One
permutation is one sample — see the four-permutation sweep on VGG-19 before
reading anything into the exact value.

## Reproduce

```
run.py --model alexnet --reps 250 --seed 0 --save-run results/alexnet-scramble-r250-s0 --notes AlexNet depth profile, 21 taps. Depth control: does the crossover to log need 33 layers, or is it carried by the classifier ReLU + softmax? --figures out/ --layers all --scramble
```

Code: `f9ab3861976a`. Weights: torchvision alexnet IMAGENET1K_V1.
