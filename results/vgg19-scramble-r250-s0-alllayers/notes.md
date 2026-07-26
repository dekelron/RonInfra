# vgg19-scramble-r250-s0-alllayers

> **Metric change (2026-07-26).** `logness` was removed; the headline
> statistic is now `λ`, the exponent of `D = a + b·(c^λ − 1)/λ` — **0 is the
> log law, 1 linear in contrast**. Any `logness` value in the prose below is
> on the retired scale. This run's `result.json` carries `lambda`,
> `lambda_ci` and `lambda_r2`, re-fitted from the committed surfaces, and is
> the authority where the two disagree. See `wiki/Results.md`.

`vgg19`, 250 reps/cell, best mean R² 0.924 at `features.19`.

## What this run was for

First all-layers sweep. **Superseded** -- it ran before the in-place-ReLU tap fix, so every conv tap holds its ReLU's output and conv/ReLU pairs are bit-identical. Kept only as the cross-check that post-activation taps reproduce.

## What it showed

`logness` at the ends: -0.372 at `features.0` (conv1_1) and +0.120 at `prob`.
The full profile is the point, not any single layer -- see
[Results](../../wiki/Results.md#where-the-log-response-appears-along-depth).

**Do not read pre-activation numbers off this run.** Its replacement is
[`vgg19-scramble-r250-s0-alllayers-fixed`](../vgg19-scramble-r250-s0-alllayers-fixed/notes.md).

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0-alllayers --notes Full grid tapping every leaf module (43 taps), for the continuous depth profile of the linear-vs-log index --figures out/ --layers all --scramble
```

Code: `089c28dec23a`. Weights: torchvision vgg19 IMAGENET1K_V1.
