# vit-b-16-scramble-r50-s0

`vit_b_16`, 50 reps/cell, best mean R² 0.848 at `encoder.layers.encoder_layer_0.ln_1`.

## What this run was for

Reps companion to vit_b_16-r250-s0. ViT's affine prefix is conv_proj plus the first LayerNorm, so the floor should extend past the patch embedding exactly as BatchNorm extended it in vgg19_bn.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vit_b_16 --reps 50 --seed 0 --save-run results/vit-b-16-scramble-r50-s0 --notes Reps companion to vit_b_16-r250-s0. ViT's affine prefix is conv_proj plus the first LayerNorm, so the floor should extend past the patch embedding exactly as BatchNorm extended it in vgg19_bn. --figures out/ --layers all --scramble
```

Code: `7bdf878d43c4`. Weights: torchvision vit_b_16 IMAGENET1K_V1.
