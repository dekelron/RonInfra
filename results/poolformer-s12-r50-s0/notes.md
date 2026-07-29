# poolformer-s12-r50-s0

`timm:poolformer_s12.sail_in1k`, 50 reps/cell, best mean R² 0.962 at `stages.2.blocks.0.mlp.fc1`.

## What this run was for

Unusual-architecture screen: PoolFormer-S12, timm sail_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.63 | +0.61 | +0.47 | -0.40 | -0.80 | -0.94 | -0.81 | +0.34 |

`prob` λ = **-0.034** at λ-R² 0.889, 173 taps.
Band contrast +0.48, Spearman ρ vs frequency -0.74.
What it tests: **pooling** in place of attention (the MetaFormer control).

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:poolformer_s12.sail_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/poolformer-s12-r50-s0 --notes Unusual-architecture screen: PoolFormer-S12, timm sail_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.
```

Code: `unknown`. Weights: timm 1.0.22; poolformer_s12.sail_in1k; hf_hub_id=timm/poolformer_s12.sail_in1k; tag=sail_in1k.
