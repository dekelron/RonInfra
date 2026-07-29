# convnext-small-r50-s0

`convnext_small`, 50 reps/cell, best mean R² 0.960 at `features.5.1.block.3`.

## What this run was for

Architecture coverage batch 3: convnext_small IMAGENET1K_V1, trained model, all layers, reps 50, seed 0; native 224 crop; matched scrambled control included.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.67 | +0.45 | +0.44 | +0.26 | -0.21 | -0.73 | -0.64 | -0.52 |

Bands: low (1, 1.75) **+0.56**, mid (7–28) **-0.23**, high (56, 75) **-0.58**.
**Monotone in frequency, not a mid-band dip** (Spearman ρ = -0.90 against frequency). The band statistic reads -0.35, but that number presumes a dip; here λ simply declines across the range, so read the row as a monotone profile.

λ at `prob` = **+0.025** at λ-R² 0.926. Read the two together — λ locates a
response only insofar as the family describes it.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model convnext_small --reps 50 --seed 0 --layers all --save-run results/convnext-small-r50-s0 --notes Architecture coverage batch 3: convnext_small IMAGENET1K_V1, trained model, all layers, reps 50, seed 0; native 224 crop; matched scrambled control included. --quiet
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision convnext_small IMAGENET1K_V1.
