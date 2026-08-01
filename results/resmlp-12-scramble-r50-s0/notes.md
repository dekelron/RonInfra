# resmlp-12-scramble-r50-s0

`timm:resmlp_12_224.fb_in1k`, 50 reps/cell, best mean R² 0.750 at `blocks.0.norm2`.

## What this run was for

Unusual-architecture control: ResMLP-12 weights scrambled within layer, shared ImageNet normalization, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.84 | +0.77 | +0.82 | +0.81 | +0.74 | +0.56 | +0.63 | +0.54 |

`prob` λ = **+0.756** at λ-R² 0.924, 114 taps.
Band contrast -0.12, Spearman ρ vs frequency -0.90.


**Control validity — BROKEN.** r(logits,prob) **0.590** at ratio 1.3e-4, max |D_logits| **2409** against ~2 for the others, and **42 of 114 taps pinned at the λ = +4 bound** (trained companion: 0). ResMLP replaces LayerNorm with a learned per-channel Affine, which does not renormalise by the input, so nothing absorbs the permuted scales and the logits explode. **Do not table this against any other control.**

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:resmlp_12_224.fb_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/resmlp-12-scramble-r50-s0 --notes Unusual-architecture control: ResMLP-12 weights scrambled within layer, shared ImageNet normalization, reps 50, seed 0. --scramble
```

Code: `unknown`. Weights: timm 1.0.22; resmlp_12_224.fb_in1k; hf_hub_id=timm/resmlp_12_224.fb_in1k; tag=fb_in1k.

> **Its `run.json` overstates one field.** `model_details.standard_scramble_valid`
> reads `true` here, because `TimmModel` derived that flag from `batchnorm_modules
> == 0` alone — and this run is the standing proof that BN-free is only the
> *necessary* half of the renormalisation rule. The field is left as recorded
> (it is provenance), but the run it describes is the broken control, not a valid
> one. The back-end was changed on 2026-07-31 to report `batchnorm_free` plus a
> normalisation census instead, so no later run repeats the claim.
