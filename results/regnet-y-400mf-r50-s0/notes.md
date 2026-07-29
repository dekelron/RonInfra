# regnet-y-400mf-r50-s0

`regnet_y_400mf`, 50 reps/cell, best mean R² 0.975 at `trunk_output.block4.block4-3.f.b.1`.

## What this run was for

Architecture coverage batch 2: regnet_y_400mf IMAGENET1K_V1, trained model, all layers, reps 50, seed 0. No standard scrambled companion because BatchNorm running statistics make that control invalid.

## What it showed

Per-frequency λ at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.37 | +0.02 | -0.38 | -3.00 | +0.16 | +0.14 | +0.29 | +0.33 |

Bands: low (1, 1.75) **+0.20**, mid (7–28) **+0.15**, high (56, 75) **+0.31**.
Mid-band dips by **0.05** relative to both ends (ρ = +0.18, so this is a genuine band shape rather than a monotone trend).

λ at `prob` = **+0.148** at λ-R² 0.948. Read the two together — λ locates a
response only insofar as the family describes it.

**λ is pinned at the search bound (-3) at 7 cyc/img — those cells are not measurements**, and the bands above are computed with them dropped. Including them would read +1.10 instead of +0.05: the difference is the bound, not a response.

Part of the 23-run architecture screen merged on 2026-07-29 (`--reps 50`, seed 0,
all layers, grids identical to every other run here). The screen's conclusion is
in [`wiki/Results.md`](../../wiki/Results.md#the-mid-band-dip-does-not-generalise-across-architectures):
the mid-band dip established on six seed-swept series does **not** hold across a
wider set. **One seed** — the seeded series needed 3-4 runs each before their
shapes could be trusted, and this run has not had that.

## Reproduce

```
run.py --model regnet_y_400mf --reps 50 --seed 0 --layers all --save-run results/regnet-y-400mf-r50-s0 --notes Architecture coverage batch 2: regnet_y_400mf IMAGENET1K_V1, trained model, all layers, reps 50, seed 0. No standard scrambled companion because BatchNorm running statistics make that control invalid.
```

Code: `baa2fa575e28` (dirty tree). Weights: torchvision regnet_y_400mf IMAGENET1K_V1.
