# focalnet-tiny-srf-r50-s0

`timm:focalnet_tiny_srf.ms_in1k`, 50 reps/cell, best mean R² 0.957 at `layers.2.blocks.0.modulation.focal_layers.1.1`.

## What this run was for

Unusual-architecture screen: FocalNet-Tiny-SRF, timm ms_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.71 | +1.17 | +0.94 | +0.62 | +0.13 | +0.09 | -0.10 | +0.22 |

`prob` λ = **+0.420** at λ-R² 0.941, 268 taps.
Band contrast -0.22, Spearman ρ vs frequency -0.79.
What it tests: focal modulation in place of attention.

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:focalnet_tiny_srf.ms_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/focalnet-tiny-srf-r50-s0 --notes Unusual-architecture screen: FocalNet-Tiny-SRF, timm ms_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.
```

Code: `unknown`. Weights: timm 1.0.22; focalnet_tiny_srf.ms_in1k; hf_hub_id=timm/focalnet_tiny_srf.ms_in1k; tag=ms_in1k.
