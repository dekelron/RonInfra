# mobilenet-v3-large-r50-s0

`mobilenet_v3_large`, 50 reps/cell, best mean R² 0.967 at `features.6.block.1.0`.

## What this run was for

Architecture coverage batch 2: mobilenet_v3_large IMAGENET1K_V1, trained model, all layers, reps 50, seed 0. No standard scrambled companion because BatchNorm running statistics make that control invalid.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| -0.14 | -0.34 | -0.85 | -0.94 | -0.79 | -0.61 | -2.31 | -3.00 |

Bands: low (1, 1.75) **-0.24**, mid (7–28) **-0.78**, high (56, 75) **-2.31**.
Mid-band peaks by **1.53** relative to both ends (ρ = -0.68, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **-0.818** at λ-R² 0.881. Read the two together — λ locates a
response only insofar as the family describes it.

**λ is pinned at the search bound (-3) at 75 cyc/img — those cells are not measurements**, and the bands above are computed with them dropped. Including them would read -1.88 instead of -1.53: the difference is the bound, not a response.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model mobilenet_v3_large --reps 50 --seed 0 --layers all --save-run results/mobilenet-v3-large-r50-s0 --notes Architecture coverage batch 2: mobilenet_v3_large IMAGENET1K_V1, trained model, all layers, reps 50, seed 0. No standard scrambled companion because BatchNorm running statistics make that control invalid.
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision mobilenet_v3_large IMAGENET1K_V1.
