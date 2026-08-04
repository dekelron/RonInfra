# mobilenet-v3-large-r50-s0-v2

`mobilenet_v3_large:IMAGENET1K_V2`, 50 reps/cell, best mean R² 0.965 at `features.7.block.0.0`.

## What this run was for

Lineage pair 3 of 5 on the torchvision V1-vs-V2 axis; pairs with results/mobilenet-v3-large-r50-s0. Note the V1 companion is one of the runs with a lambda pinned at a search bound, so read this pair per-tap with the pinned cells dropped.

## What it showed

The recipe change alone moves λ well past the sampling noise. Against [`mobilenet-v3-large-r50-s0`](../mobilenet-v3-large-r50-s0/notes.md): `logits` λ **-0.273** (R² 0.897) → **-0.203** (R² 0.870), `prob` **-0.818** (R² 0.881) → **-0.585** (R² 0.783). The three-seed sweep on `resnet50` puts the sampling sd at 0.043 (`logits`) and 0.007 (`prob`), so these are large. No conversion is involved anywhere — same architecture, same framework, same file format, only the training run differs. See `wiki/Results.md`, "One architecture, several training runs".

## Reproduce

```
run.py --model mobilenet_v3_large:IMAGENET1K_V2 --reps 50 --seed 0 --save-run results/mobilenet-v3-large-r50-s0-v2 --notes Lineage pair 3 of 5 on the torchvision V1-vs-V2 axis; pairs with results/mobilenet-v3-large-r50-s0. Note the V1 companion is one of the runs with a lambda pinned at a search bound, so read this pair per-tap with the pinned cells dropped. --figures out/ --layers all
```

Code: `a100e1e034c9`. Weights: torchvision mobilenet_v3_large IMAGENET1K_V2.
