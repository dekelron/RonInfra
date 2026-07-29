# swin-t-r50-s0

`swin_t`, 50 reps/cell, best mean R² 0.955 at `features.2.reduction`.

## What this run was for

Architecture coverage batch: swin_t IMAGENET1K_V1, all layers, reps 50, seed 0; local CPU runner.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.60 | +0.81 | +0.33 | +0.13 | -0.20 | -0.46 | -0.05 | +0.09 |

Bands: low (1, 1.75) **+0.71**, mid (7–28) **-0.18**, high (56, 75) **+0.02**.
**Monotone in frequency, not a mid-band dip** (Spearman ρ = -0.76 against frequency). The band statistic reads +0.19, but that number presumes a dip; here λ simply declines across the range, so read the row as a monotone profile.

λ at `prob` = **+0.109** at λ-R² 0.911. Read the two together — λ locates a
response only insofar as the family describes it.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model swin_t --reps 50 --seed 0 --layers all --save-run results/swin-t-r50-s0 --notes Architecture coverage batch: swin_t IMAGENET1K_V1, all layers, reps 50, seed 0; local CPU runner.
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision swin_t IMAGENET1K_V1.
