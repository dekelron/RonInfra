# resnext50-32x4d-r50-s0

`resnext50_32x4d`, 50 reps/cell, best mean R² 0.956 at `layer2.2.bn3`.

## What this run was for

Architecture coverage batch 3: resnext50_32x4d IMAGENET1K_V1, trained grouped-residual model, all layers, reps 50, seed 0; native 224 crop; no standard scramble because BatchNorm running statistics make it invalid.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.15 | -0.08 | -0.47 | -0.47 | +0.04 | +0.03 | -0.47 | -0.53 |

Bands: low (1, 1.75) **+0.04**, mid (7–28) **-0.13**, high (56, 75) **-0.50**.
Mid-band peaks by **0.36** relative to both ends (ρ = -0.57, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **-0.271** at λ-R² 0.958. Read the two together — λ locates a
response only insofar as the family describes it.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model resnext50_32x4d --reps 50 --seed 0 --layers all --save-run results/resnext50-32x4d-r50-s0 --notes Architecture coverage batch 3: resnext50_32x4d IMAGENET1K_V1, trained grouped-residual model, all layers, reps 50, seed 0; native 224 crop; no standard scramble because BatchNorm running statistics make it invalid. --quiet
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision resnext50_32x4d IMAGENET1K_V1.
