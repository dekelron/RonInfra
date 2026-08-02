# vgg19-bn-r50-s0-alllayers-gates

`vgg19_bn`, 50 reps/cell, best mean R² 0.928 at `features.26`.

## What this run was for

Gate-flip instrument. vgg19_bn, all taps. The case that broke 'rectifications carry it': conv stack at lambda -0.071 with zero rectifiers added over plain VGG-19.

## What it showed

**The third leg: an affine normalisation moves λ by ~1.1 and leaves `G` alone.**

`vgg19_bn` has VGG-19's topology, ReLU count, task and stimulus; BatchNorm in
eval is a per-channel affine map that cannot add a gate. Its 53 `features.*`
taps average λ **+0.150** against Caffe's +1.055 — and average `G` **23.4%**
against Caffe's 31.3%, the same order of magnitude.

`r(log₁₀ G, λ)` = **−0.003**: the flip rate carries no information about λ here
at all. Counting only gates that switch during the sweep gives **+0.323**,
sign-consistent with the other two runs and positive — the wrong direction for
the gate-freezing reading.

This is the run that most cleanly separates the two candidate variables: same
rectifiers, same flip rates, λ shifted by ~0.9 from the Caffe checkpoint. What
differs is where units sit relative to threshold, not how often they cross it.

One seed, `--reps 50`, pretrained only. The scrambled control is deliberately
absent — it is invalid on a BN net (see `wiki/Results.md`) and tests nothing
here regardless. See `wiki/Results.md`.

## Reproduce

```
run.py --model vgg19_bn --reps 50 --seed 0 --save-run results/vgg19-bn-r50-s0-alllayers-gates --notes Gate-flip instrument. vgg19_bn, all taps. The case that broke 'rectifications carry it': conv stack at lambda -0.071 with zero rectifiers added over plain VGG-19. --figures out/ --layers all
```

Code: `cf72c73fb6b5`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
