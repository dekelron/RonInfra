# shufflenet-v2-x1-0-r50-s0

`shufflenet_v2_x1_0`, 50 reps/cell, best mean R² 0.965 at `stage4.0.branch1.4`.

## What this run was for

Architecture coverage batch 3: shufflenet_v2_x1_0 IMAGENET1K_V1, trained model, all layers, reps 50, seed 0; native 224 crop; no standard scramble because BatchNorm running statistics make it invalid.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.28 | +0.15 | -0.13 | -0.15 | -0.25 | -0.45 | +0.38 | +0.42 |

Bands: low (1, 1.75) **+0.21**, mid (7–28) **-0.28**, high (56, 75) **+0.40**.
Mid-band dips by **0.50** relative to both ends (ρ = +0.17, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **+0.010** at λ-R² 0.977. Read the two together — λ locates a
response only insofar as the family describes it.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model shufflenet_v2_x1_0 --reps 50 --seed 0 --layers all --save-run results/shufflenet-v2-x1-0-r50-s0 --notes Architecture coverage batch 3: shufflenet_v2_x1_0 IMAGENET1K_V1, trained model, all layers, reps 50, seed 0; native 224 crop; no standard scramble because BatchNorm running statistics make it invalid. --quiet
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision shufflenet_v2_x1_0 IMAGENET1K_V1.
