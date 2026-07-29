# swin-v2-t-r50-s0

`swin_v2_t`, 50 reps/cell, best mean R² 0.983 at `features.5.3.mlp.0`.

## What this run was for

Architecture coverage batch 2: swin_v2_t IMAGENET1K_V1, trained shifted-window Transformer V2, all layers, reps 50, seed 0; matched scrambled control included.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.76 | +1.08 | +0.70 | +0.23 | -0.07 | -1.54 | -0.31 | +0.01 |

Bands: low (1, 1.75) **+0.92**, mid (7–28) **-0.46**, high (56, 75) **-0.15**.
**Monotone in frequency, not a mid-band dip** (Spearman ρ = -0.81 against frequency). The band statistic reads +0.31, but that number presumes a dip; here λ simply declines across the range, so read the row as a monotone profile.

λ at `prob` = **+0.121** at λ-R² 0.940. Read the two together — λ locates a
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
run.py --model swin_v2_t --reps 50 --seed 0 --layers all --save-run results/swin-v2-t-r50-s0 --notes Architecture coverage batch 2: swin_v2_t IMAGENET1K_V1, trained shifted-window Transformer V2, all layers, reps 50, seed 0; matched scrambled control included.
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision swin_v2_t IMAGENET1K_V1.
