# vit-b-16-scramble-r50-s0

`vit_b_16`, 50 reps/cell, best mean R² 0.848 at `encoder.layers.encoder_layer_0.ln_1`.

## What this run was for

Reps companion to vit_b_16-r250-s0. ViT's affine prefix is conv_proj plus the first LayerNorm, so the floor should extend past the patch embedding exactly as BatchNorm extended it in vgg19_bn.

## What it showed

Reps companion to
[`vit-b-16-scramble-r250-s0`](../vit-b-16-scramble-r250-s0/notes.md), and
unusually for a scrambled run it is stable: `prob` λ +0.711 → **+0.714** at
λ-R² 0.981 both times, mean R² 0.797 → 0.796.

That stability is the contrast worth keeping. The scrambled BatchNorm controls
(`vgg19_bn`, `resnet50`) move by 1.5 and 0.15 respectively across the same rep
change with intervals spanning most or all of the search range; this one is
reproducible to three decimals. A scrambled control *can* be a well-determined
measurement — it is the running-statistics decalibration, not scrambling as
such, that makes the other two unreadable.

## Reproduce

```
run.py --model vit_b_16 --reps 50 --seed 0 --save-run results/vit-b-16-scramble-r50-s0 --notes Reps companion to vit_b_16-r250-s0. ViT's affine prefix is conv_proj plus the first LayerNorm, so the floor should extend past the patch embedding exactly as BatchNorm extended it in vgg19_bn. --figures out/ --layers all --scramble
```

Code: `7bdf878d43c4`. Weights: torchvision vit_b_16 IMAGENET1K_V1.
