# gmlp-s16-native-r50-s0

`timm:gmlp_s16_224.ra3_in1k`, 50 reps/cell, best mean R² 0.957 at `blocks.23.mlp_channels.gate.proj`.

## What this run was for

Preprocessing sensitivity: gMLP-S16 timm ra3_in1k, checkpoint-native mean/std, all layers, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| -0.33 | -0.62 | -0.44 | -0.36 | -0.75 | -0.17 | +0.06 | -0.12 |

`prob` λ = **-0.346** at λ-R² 0.936, 246 taps.
Band contrast -0.05, Spearman ρ vs frequency +0.60.
What it tests: the gMLP run again under the checkpoint's own normalisation, as a preprocessing sensitivity check.

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:gmlp_s16_224.ra3_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/gmlp-s16-native-r50-s0 --notes Preprocessing sensitivity: gMLP-S16 timm ra3_in1k, checkpoint-native mean/std, all layers, reps 50, seed 0. --preprocessing native
```

Code: `unknown`. Weights: timm 1.0.22; gmlp_s16_224.ra3_in1k; hf_hub_id=timm/gmlp_s16_224.ra3_in1k; tag=ra3_in1k.
