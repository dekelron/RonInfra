# resmlp-12-r50-s0

`timm:resmlp_12_224.fb_in1k`, 50 reps/cell, best mean R² 0.902 at `blocks.8.mlp_channels.act`.

## What this run was for

Unusual-architecture screen: ResMLP-12, timm fb_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| -0.28 | -0.39 | -0.35 | -0.61 | -1.40 | -0.19 | -0.16 | -0.01 |

`prob` λ = **-0.315** at λ-R² 0.892, 114 taps.
Band contrast +0.40, Spearman ρ vs frequency +0.55.
What it tests: **MLP only**, with a learned per-channel Affine in place of LayerNorm.

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:resmlp_12_224.fb_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/resmlp-12-r50-s0 --notes Unusual-architecture screen: ResMLP-12, timm fb_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.
```

Code: `unknown`. Weights: timm 1.0.22; resmlp_12_224.fb_in1k; hf_hub_id=timm/resmlp_12_224.fb_in1k; tag=fb_in1k.
