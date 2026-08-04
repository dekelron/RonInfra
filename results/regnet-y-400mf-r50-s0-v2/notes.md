# regnet-y-400mf-r50-s0-v2

`regnet_y_400mf:IMAGENET1K_V2`, 50 reps/cell, best mean R² 0.969 at `trunk_output.block3.block3-0.f.se.activation`.

## What this run was for

Lineage pair 5 of 5 on the torchvision V1-vs-V2 axis; pairs with results/regnet-y-400mf-r50-s0 -- the run whose band contrast was +1.10 with a bound-pinned cell and +0.05 without, so the pinned-cell filter is mandatory when reading this one.

## What it showed

The recipe change alone moves λ well past the sampling noise. Against [`regnet-y-400mf-r50-s0`](../regnet-y-400mf-r50-s0/notes.md): `logits` λ **+0.354** (R² 0.948) → **-0.050** (R² 0.861), `prob` **+0.148** (R² 0.948) → **-0.296** (R² 0.891). The three-seed sweep on `resnet50` puts the sampling sd at 0.043 (`logits`) and 0.007 (`prob`), so these are large. No conversion is involved anywhere — same architecture, same framework, same file format, only the training run differs. See `wiki/Results.md`, "One architecture, several training runs".

## Reproduce

```
run.py --model regnet_y_400mf:IMAGENET1K_V2 --reps 50 --seed 0 --save-run results/regnet-y-400mf-r50-s0-v2 --notes Lineage pair 5 of 5 on the torchvision V1-vs-V2 axis; pairs with results/regnet-y-400mf-r50-s0 -- the run whose band contrast was +1.10 with a bound-pinned cell and +0.05 without, so the pinned-cell filter is mandatory when reading this one. --figures out/ --layers all
```

Code: `a100e1e034c9`. Weights: torchvision regnet_y_400mf IMAGENET1K_V2.
