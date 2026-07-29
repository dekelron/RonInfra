# gmlp-s16-r50-s0

`timm:gmlp_s16_224.ra3_in1k`, 50 reps/cell, best mean R² 0.976 at `blocks.3.mlp_channels.fc1`.

## What this run was for

Unusual-architecture screen: gMLP-S16, timm ra3_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| -0.58 | -1.33 | +0.01 | -0.22 | -0.24 | -0.26 | -0.16 | -0.32 |

`prob` λ = **-0.250** at λ-R² 0.934, 246 taps.
Band contrast -0.71, Spearman ρ vs frequency +0.29.
What it tests: **MLP only** — no attention and no convolution beyond the patch embedding.

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:gmlp_s16_224.ra3_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/gmlp-s16-r50-s0 --notes Unusual-architecture screen: gMLP-S16, timm ra3_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.
```

Code: `unknown`. Weights: timm 1.0.22; gmlp_s16_224.ra3_in1k; hf_hub_id=timm/gmlp_s16_224.ra3_in1k; tag=ra3_in1k.
