# convnext-small-scramble-r50-s0

`convnext_small`, 50 reps/cell, best mean R² 0.866 at `features.7.2.stochastic_depth`.

## What this run was for

Architecture coverage batch 3 control: convnext_small IMAGENET1K_V1 weights scrambled within layer, all layers, reps 50, stimulus seed 0, scramble seed 0; LayerNorm has no running statistics.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.58 | +0.57 | +0.55 | +0.59 | +0.61 | +0.57 | +0.54 | +0.54 |

Bands: low (1, 1.75) **+0.57**, mid (7–28) **+0.59**, high (56, 75) **+0.54**.
Mid-band peaks by **0.05** relative to both ends (ρ = -0.55, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **+0.567** at λ-R² 0.975. Read the two together — λ locates a
response only insofar as the family describes it.

**Control validity.** r(`logits`, `prob`) = 0.999807 at ratio 1.006e-03, i.e. the softmax is in its affine regime — this is a usable control, unlike the BatchNorm ones (r 0.162/0.673 at ratio 1e-10). This net has no running statistics for `--scramble` to desynchronise.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model convnext_small --reps 50 --seed 0 --layers all --scramble --save-run results/convnext-small-scramble-r50-s0 --notes Architecture coverage batch 3 control: convnext_small IMAGENET1K_V1 weights scrambled within layer, all layers, reps 50, stimulus seed 0, scramble seed 0; LayerNorm has no running statistics. --quiet
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision convnext_small IMAGENET1K_V1.
