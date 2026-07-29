# poolformer-s12-scramble-r50-s0

`timm:poolformer_s12.sail_in1k`, 50 reps/cell, best mean R² 0.896 at `stages.0.blocks.1.layer_scale1`.

## What this run was for

Unusual-architecture control: PoolFormer-S12 weights scrambled within layer, shared ImageNet normalization, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.31 | +0.34 | +0.58 | +0.57 | +0.47 | +0.32 | +0.45 | +0.46 |

`prob` λ = **+0.454** at λ-R² 0.924, 173 taps.
Band contrast -0.13, Spearman ρ vs frequency +0.17.


**Control validity — usable.** r(logits,prob) 0.968 at ratio 7.1e-4. Logit magnitudes are sane (~2), so the softmax is not saturating, but this is visibly less clean than a LayerNorm control (0.9998).

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:poolformer_s12.sail_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/poolformer-s12-scramble-r50-s0 --notes Unusual-architecture control: PoolFormer-S12 weights scrambled within layer, shared ImageNet normalization, reps 50, seed 0. --scramble
```

Code: `unknown`. Weights: timm 1.0.22; poolformer_s12.sail_in1k; hf_hub_id=timm/poolformer_s12.sail_in1k; tag=sail_in1k.
