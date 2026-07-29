# convnext-tiny-r50-s0

`convnext_tiny`, 50 reps/cell, best mean R² 0.946 at `features.4.1`.

## What this run was for

Architecture coverage batch: convnext_tiny IMAGENET1K_V1, all layers, reps 50, seed 0; local CPU runner.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.74 | +0.30 | -0.46 | -0.31 | -0.85 | -1.80 | -0.72 | -0.50 |

Bands: low (1, 1.75) **+0.52**, mid (7–28) **-0.99**, high (56, 75) **-0.61**.
**Monotone in frequency, not a mid-band dip** (Spearman ρ = -0.76 against frequency). The band statistic reads +0.38, but that number presumes a dip; here λ simply declines across the range, so read the row as a monotone profile.

λ at `prob` = **-0.481** at λ-R² 0.903. Read the two together — λ locates a
response only insofar as the family describes it.

**The interval is unbounded below at 28 cyc/img** (lower endpoint at the search bound), so λ there is a direction, not a value.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model convnext_tiny --reps 50 --seed 0 --layers all --save-run results/convnext-tiny-r50-s0 --notes Architecture coverage batch: convnext_tiny IMAGENET1K_V1, all layers, reps 50, seed 0; local CPU runner.
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision convnext_tiny IMAGENET1K_V1.
