# maxvit-t-r50-s0

`maxvit_t`, 50 reps/cell, best mean R² 0.987 at `blocks.3.layers.0.layers.grid_attention.mlp_layer.2`.

## What this run was for

Architecture coverage batch 2: maxvit_t IMAGENET1K_V1, trained hybrid convolution-attention model, all layers, reps 50, seed 0. No standard scrambled companion because the convolutional path contains BatchNorm running statistics.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.51 | +0.31 | +0.03 | -0.54 | -0.42 | -0.81 | -0.32 | -0.23 |

Bands: low (1, 1.75) **+0.41**, mid (7–28) **-0.59**, high (56, 75) **-0.27**.
Mid-band dips by **0.32** relative to both ends (ρ = -0.60, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **-0.273** at λ-R² 0.968. Read the two together — λ locates a
response only insofar as the family describes it.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model maxvit_t --reps 50 --seed 0 --layers all --save-run results/maxvit-t-r50-s0 --notes Architecture coverage batch 2: maxvit_t IMAGENET1K_V1, trained hybrid convolution-attention model, all layers, reps 50, seed 0. No standard scrambled companion because the convolutional path contains BatchNorm running statistics.
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision maxvit_t IMAGENET1K_V1.
