# xcit-nano-12-p16-r50-s0

`timm:xcit_nano_12_p16_224.fb_in1k`, 50 reps/cell, best mean R² 0.973 at `blocks.1.drop_path3`.

## What this run was for

Unusual-architecture screen: XCiT-Nano-12 p16, timm fb_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| -0.04 | -0.45 | -2.07 | +1.71 | -1.47 | -3.00 | -0.48 | -0.42 |

`prob` λ = **-0.469** at λ-R² 0.745, 232 taps.
Band contrast -0.57, Spearman ρ vs frequency -0.14.
What it tests: cross-covariance attention (over channels, not tokens).

**1 taps pinned at a λ search bound** and excluded from the bands — those are not measurements.

Part of the 10-run unusual-architecture screen merged 2026-07-29 (`--reps 50`, seed 0,
`--layers all`, 224x224, grids identical to every other run here; timm back-end).
**One seed.** See [`wiki/Results.md`](../../wiki/Results.md#beyond-convolution-and-attention).

## Reproduce

```
run.py --model timm:xcit_nano_12_p16_224.fb_in1k --reps 50 --seed 0 --layers all --quiet --save-run ../results/xcit-nano-12-p16-r50-s0 --notes Unusual-architecture screen: XCiT-Nano-12 p16, timm fb_in1k, shared ImageNet normalization, all layers, reps 50, seed 0.
```

Code: `unknown`. Weights: timm 1.0.22; xcit_nano_12_p16_224.fb_in1k; hf_hub_id=timm/xcit_nano_12_p16_224.fb_in1k; tag=fb_in1k.
