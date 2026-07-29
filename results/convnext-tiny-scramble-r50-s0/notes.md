# convnext-tiny-scramble-r50-s0

`convnext_tiny`, 50 reps/cell, best mean R² 0.950 at `features.7.2.block.5`.

## What this run was for

Architecture coverage batch control: convnext_tiny IMAGENET1K_V1 weights scrambled within layer, all layers, reps 50, seed 0; local CPU runner.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.16 | +0.20 | +0.24 | +0.25 | +0.30 | +0.35 | +0.35 | +0.31 |

Bands: low (1, 1.75) **+0.18**, mid (7–28) **+0.30**, high (56, 75) **+0.33**.
**Monotone in frequency, not a mid-band dip** (Spearman ρ = +0.90 against frequency). The band statistic reads -0.12, but that number presumes a dip; here λ simply declines across the range, so read the row as a monotone profile.

λ at `prob` = **+0.274** at λ-R² 0.960. Read the two together — λ locates a
response only insofar as the family describes it.

**Control validity.** r(`logits`, `prob`) = 0.999804 at ratio 9.743e-04, i.e. the softmax is in its affine regime — this is a usable control, unlike the BatchNorm ones (r 0.162/0.673 at ratio 1e-10). This net has no running statistics for `--scramble` to desynchronise.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model convnext_tiny --reps 50 --seed 0 --layers all --scramble --save-run results/convnext-tiny-scramble-r50-s0 --notes Architecture coverage batch control: convnext_tiny IMAGENET1K_V1 weights scrambled within layer, all layers, reps 50, seed 0; local CPU runner.
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision convnext_tiny IMAGENET1K_V1.
