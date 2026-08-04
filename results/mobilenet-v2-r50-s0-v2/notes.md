# mobilenet-v2-r50-s0-v2

`mobilenet_v2:IMAGENET1K_V2`, 50 reps/cell, best mean R² 0.971 at `features.10.conv.1.1`.

## What this run was for

Lineage pair 4 of 5 on the torchvision V1-vs-V2 axis; pairs with results/mobilenet-v2-r50-s0.

## What it showed

The recipe change alone moves λ well past the sampling noise. Against [`mobilenet-v2-r50-s0`](../mobilenet-v2-r50-s0/notes.md): `logits` λ **+0.057** (R² 0.956) → **+0.131** (R² 0.945), `prob` **+0.060** (R² 0.963) → **+0.163** (R² 0.920). The three-seed sweep on `resnet50` puts the sampling sd at 0.043 (`logits`) and 0.007 (`prob`), so these are large. No conversion is involved anywhere — same architecture, same framework, same file format, only the training run differs. See `wiki/Results.md`, "One architecture, several training runs".

## Reproduce

```
run.py --model mobilenet_v2:IMAGENET1K_V2 --reps 50 --seed 0 --save-run results/mobilenet-v2-r50-s0-v2 --notes Lineage pair 4 of 5 on the torchvision V1-vs-V2 axis; pairs with results/mobilenet-v2-r50-s0. --figures out/ --layers all
```

Code: `a100e1e034c9`. Weights: torchvision mobilenet_v2 IMAGENET1K_V2.
