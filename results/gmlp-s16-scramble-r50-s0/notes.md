# gmlp-s16-scramble-r50-s0

`timm:gmlp_s16_224.ra3_in1k`, 50 reps/cell, best mean R² 0.919 at `blocks.5.mlp_channels.gate.norm`.

## What this run was for

Unusual-architecture control: gMLP-S16 weights scrambled within layer, shared ImageNet normalization, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.08 | +0.06 | +0.04 | +0.02 | +0.01 | -0.05 | -0.06 | -0.04 |

`prob` λ = **+0.014** at λ-R² 0.888, 246 taps.
Band contrast -0.04, Spearman ρ vs frequency -0.93.


**Control validity — clean.** r(logits,prob) 0.992 at ratio 8.7e-4 — the softmax stays in its affine regime.

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:gmlp_s16_224.ra3_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/gmlp-s16-scramble-r50-s0 --notes Unusual-architecture control: gMLP-S16 weights scrambled within layer, shared ImageNet normalization, reps 50, seed 0. --scramble
```

Code: `unknown`. Weights: timm 1.0.22; gmlp_s16_224.ra3_in1k; hf_hub_id=timm/gmlp_s16_224.ra3_in1k; tag=ra3_in1k.
