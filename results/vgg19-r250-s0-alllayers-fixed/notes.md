# vgg19-r250-s0-alllayers-fixed

`vgg19`, 250 reps/cell, best mean R² 0.928 at `classifier.4`.

## What this run was for

Every leaf module tapped (43 + `logits`/`prob`), IMAGENET1K_V1 weights, trained. One quarter of the 2x2 that locates *where* along depth the response changes shape, and whether the two checkpoints differ there.

## What it showed

`logness` at the ends: -0.863 at `features.0` (conv1_1) and +0.466 at `prob`.
(On the definition adopted 2026-07-26; was -0.224 / +0.151 under the
superseded `logness_r2diff`, which `result.json` still records.)
The full profile is the point, not any single layer -- see
[Results](../../wiki/Results.md#where-the-log-response-appears-along-depth).

Agrees with the Caffe run at `features.0` to 0.001, then diverges with
depth to ~0.27 by `features.33`, then re-converges at the classifier
(0.006 at `logits`). Same input, different middle, same output.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-r250-s0-alllayers-fixed --notes All 43 taps on the fixed hook (pre-activation taps no longer overwritten by in-place ReLU). IMAGENET1K_V1 half of the checkpoint comparison. --figures out/ --layers all
```

Code: `d2ebd925b086`. Weights: torchvision vgg19 IMAGENET1K_V1.
