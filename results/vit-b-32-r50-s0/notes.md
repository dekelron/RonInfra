# vit-b-32-r50-s0

`vit_b_32`, 50 reps/cell, best mean R² 0.943 at `encoder.layers.encoder_layer_2.mlp.0`.

## What this run was for

Architecture coverage batch: vit_b_32 IMAGENET1K_V1, all layers, reps 50, seed 0; local CPU runner.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| -0.45 | -0.53 | -1.18 | -0.56 | -0.77 | -0.78 | -0.12 | -0.25 |

Bands: low (1, 1.75) **-0.49**, mid (7–28) **-0.71**, high (56, 75) **-0.19**.
Mid-band dips by **0.21** relative to both ends (ρ = +0.29, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **-0.546** at λ-R² 0.927. Read the two together — λ locates a
response only insofar as the family describes it.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model vit_b_32 --reps 50 --seed 0 --layers all --save-run results/vit-b-32-r50-s0 --notes Architecture coverage batch: vit_b_32 IMAGENET1K_V1, all layers, reps 50, seed 0; local CPU runner.
```

Code: `baa2fa575e28`. Weights: torchvision vit_b_32 IMAGENET1K_V1.
