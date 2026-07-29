# googlenet-r50-s0

`googlenet`, 50 reps/cell, best mean R² 0.974 at `inception4a.branch4.0`.

## What this run was for

Architecture coverage batch 3: googlenet IMAGENET1K_V1, trained Inception model, all layers, reps 50, seed 0; native 224 crop; no standard scramble because BatchNorm running statistics make it invalid.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| -0.03 | -0.30 | -0.15 | -0.31 | -0.58 | -0.55 | -0.57 | -0.52 |

Bands: low (1, 1.75) **-0.17**, mid (7–28) **-0.48**, high (56, 75) **-0.54**.
**Monotone in frequency, not a mid-band dip** (Spearman ρ = -0.76 against frequency). The band statistic reads -0.06, but that number presumes a dip; here λ simply declines across the range, so read the row as a monotone profile.

λ at `prob` = **-0.418** at λ-R² 0.973. Read the two together — λ locates a
response only insofar as the family describes it.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model googlenet --reps 50 --seed 0 --layers all --save-run results/googlenet-r50-s0 --notes Architecture coverage batch 3: googlenet IMAGENET1K_V1, trained Inception model, all layers, reps 50, seed 0; native 224 crop; no standard scramble because BatchNorm running statistics make it invalid. --quiet
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision googlenet IMAGENET1K_V1.
