# mobilenet-v2-r50-s0

`mobilenet_v2`, 50 reps/cell, best mean R² 0.959 at `features.17.conv.3`.

## What this run was for

Architecture coverage batch 3: mobilenet_v2 IMAGENET1K_V1, trained model, all layers, reps 50, seed 0; native 224 crop; no standard scramble because BatchNorm running statistics make it invalid.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.35 | +0.11 | -0.21 | -0.40 | -0.44 | +0.01 | +0.21 | +0.15 |

Bands: low (1, 1.75) **+0.23**, mid (7–28) **-0.28**, high (56, 75) **+0.18**.
Mid-band dips by **0.45** relative to both ends (ρ = -0.02, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **+0.060** at λ-R² 0.963. Read the two together — λ locates a
response only insofar as the family describes it.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model mobilenet_v2 --reps 50 --seed 0 --layers all --save-run results/mobilenet-v2-r50-s0 --notes Architecture coverage batch 3: mobilenet_v2 IMAGENET1K_V1, trained model, all layers, reps 50, seed 0; native 224 crop; no standard scramble because BatchNorm running statistics make it invalid. --quiet
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision mobilenet_v2 IMAGENET1K_V1.
