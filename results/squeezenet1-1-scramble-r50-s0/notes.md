# squeezenet1-1-scramble-r50-s0

`squeezenet1_1`, 50 reps/cell, best mean R² 0.750 at `features.0`.

## What this run was for

Architecture coverage batch 3 control: squeezenet1_1 IMAGENET1K_V1 weights scrambled within layer, all layers, reps 50, stimulus seed 0, scramble seed 0; valid because SqueezeNet has no BatchNorm running statistics.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +1.26 | +1.19 | +1.07 | +1.00 | +1.00 | +1.05 | +1.07 | +1.04 |

Bands: low (1, 1.75) **+1.23**, mid (7–28) **+1.02**, high (56, 75) **+1.05**.
Mid-band dips by **0.04** relative to both ends (ρ = -0.60, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **+1.057** at λ-R² 1.000. Read the two together — λ locates a
response only insofar as the family describes it.

**Control validity.** r(`logits`, `prob`) = 0.999899 at ratio 6.855e-04, i.e. the softmax is in its affine regime — this is a usable control, unlike the BatchNorm ones (r 0.162/0.673 at ratio 1e-10). This net has no running statistics for `--scramble` to desynchronise.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model squeezenet1_1 --reps 50 --seed 0 --layers all --scramble --scramble-seed 0 --save-run results/squeezenet1-1-scramble-r50-s0 --notes Architecture coverage batch 3 control: squeezenet1_1 IMAGENET1K_V1 weights scrambled within layer, all layers, reps 50, stimulus seed 0, scramble seed 0; valid because SqueezeNet has no BatchNorm running statistics. --quiet
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision squeezenet1_1 IMAGENET1K_V1.
