# vgg19-scramble-r250-s0-alllayers-fixed

> **Metric change (2026-07-26).** `logness` was removed; the headline
> statistic is now `λ`, the exponent of `D = a + b·(c^λ − 1)/λ` — **0 is the
> log law, 1 linear in contrast**. Any `logness` value in the prose below is
> on the retired scale. This run's `result.json` carries `lambda`,
> `lambda_ci` and `lambda_r2`, re-fitted from the committed surfaces, and is
> the authority where the two disagree. See `wiki/Results.md`.

`vgg19`, 250 reps/cell, best mean R² 0.924 at `features.20`.

## What this run was for

Every leaf module tapped (43 + `logits`/`prob`), IMAGENET1K_V1 weights, scrambled. One quarter of the 2x2 that locates *where* along depth the response changes shape, and whether the two checkpoints differ there.

## What it showed

`logness` at the ends: -0.860 at `features.0` (conv1_1) and +0.076 at `prob`.
(On the definition adopted 2026-07-26; was -0.225 / +0.120 under the
superseded `logness_r2diff`. The change matters most here: the old form read
this control as nearly as log-like as the trained net, which it is not.)
The full profile is the point, not any single layer -- see
[Results](../../wiki/Results.md#where-the-log-response-appears-along-depth).

Crosses zero at `features.16` and sits near +0.12 for the rest of the
network -- the scrambled net reaches the same log-side value as the
trained one, which is what collapses the learned contribution here.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0-alllayers-fixed --notes All 43 taps on the fixed hook (pre-activation taps no longer overwritten by in-place ReLU). IMAGENET1K_V1 half of the checkpoint comparison. --figures out/ --layers all --scramble
```

Code: `d2ebd925b086`. Weights: torchvision vgg19 IMAGENET1K_V1.
