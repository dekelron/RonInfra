# focalnet-tiny-srf-scramble-r50-s0

`timm:focalnet_tiny_srf.ms_in1k`, 50 reps/cell, best mean R² 0.881 at `prob`.

## What this run was for

Unusual-architecture control: FocalNet-Tiny-SRF weights scrambled within layer, shared ImageNet normalization, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.25 | +0.30 | +0.32 | +0.29 | +0.29 | +0.40 | +0.40 | +0.46 |

`prob` λ = **+0.312** at λ-R² 0.941, 268 taps.
Band contrast -0.05, Spearman ρ vs frequency +0.79.


**Control validity — usable.** r(logits,prob) 0.947 at ratio 1.4e-3. Same reading as PoolFormer: usable, not pristine.

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:focalnet_tiny_srf.ms_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/focalnet-tiny-srf-scramble-r50-s0 --notes Unusual-architecture control: FocalNet-Tiny-SRF weights scrambled within layer, shared ImageNet normalization, reps 50, seed 0. --scramble
```

Code: `unknown`. Weights: timm 1.0.22; focalnet_tiny_srf.ms_in1k; hf_hub_id=timm/focalnet_tiny_srf.ms_in1k; tag=ms_in1k.
