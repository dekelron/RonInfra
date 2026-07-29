# densenet121-r50-s0

`densenet121`, 50 reps/cell, best mean R² 0.964 at `features.denseblock4.denselayer6.relu2`.

## What this run was for

Architecture coverage batch 2: densenet121 IMAGENET1K_V1, trained model, all layers, reps 50, seed 0. No standard scrambled companion because BatchNorm running statistics make that control invalid.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.46 | +0.14 | -0.32 | -0.02 | -0.14 | -0.11 | -0.33 | -0.19 |

Bands: low (1, 1.75) **+0.30**, mid (7–28) **-0.09**, high (56, 75) **-0.26**.
Mid-band peaks by **0.17** relative to both ends (ρ = -0.69, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **-0.125** at λ-R² 0.978. Read the two together — λ locates a
response only insofar as the family describes it.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model densenet121 --reps 50 --seed 0 --layers all --save-run results/densenet121-r50-s0 --notes Architecture coverage batch 2: densenet121 IMAGENET1K_V1, trained model, all layers, reps 50, seed 0. No standard scrambled companion because BatchNorm running statistics make that control invalid.
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision densenet121 IMAGENET1K_V1.
