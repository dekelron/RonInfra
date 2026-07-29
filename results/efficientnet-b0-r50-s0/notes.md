# efficientnet-b0-r50-s0

`efficientnet_b0`, 50 reps/cell, best mean R² 0.959 at `features.4.0.block.0.2`.

## What this run was for

Architecture coverage batch 2: efficientnet_b0 IMAGENET1K_V1, trained model, all layers, reps 50, seed 0. No standard scrambled companion because BatchNorm running statistics make that control invalid.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.20 | -0.14 | -0.51 | -1.46 | +0.41 | -0.04 | -0.05 | -0.16 |

Bands: low (1, 1.75) **+0.03**, mid (7–28) **-0.37**, high (56, 75) **-0.10**.
Mid-band dips by **0.26** relative to both ends (ρ = -0.05, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **-0.096** at λ-R² 0.932. Read the two together — λ locates a
response only insofar as the family describes it.

**The interval is unbounded below at 7 cyc/img** (lower endpoint at the search bound), so λ there is a direction, not a value.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model efficientnet_b0 --reps 50 --seed 0 --layers all --save-run results/efficientnet-b0-r50-s0 --notes Architecture coverage batch 2: efficientnet_b0 IMAGENET1K_V1, trained model, all layers, reps 50, seed 0. No standard scrambled companion because BatchNorm running statistics make that control invalid.
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision efficientnet_b0 IMAGENET1K_V1.
