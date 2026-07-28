# vit-b-16-r50-s0

`vit_b_16`, 50 reps/cell, best mean R² 0.951 at `encoder.layers.encoder_layer_2.mlp.0`.

## What this run was for

Reps companion to vit_b_16-r250-s0. ViT's affine prefix is conv_proj plus the first LayerNorm, so the floor should extend past the patch embedding exactly as BatchNorm extended it in vgg19_bn.

## What it showed

Reps companion to [`vit-b-16-r250-s0`](../vit-b-16-r250-s0/notes.md).

**One tap on the floor out of 65**: `conv_proj`, ratio 2.219 (√5 = 2.236),
**98.6%** noise. Max noise fraction elsewhere is 19.7%, and `logits`/`prob` are
at 0.1–0.2%.

The floor stops at the patch embedding — it does **not** extend through the
first LayerNorm, where the `vgg19_bn` and `resnet50` runs both had it extend
through their first BatchNorm. LayerNorm normalises by the input's own
statistics, so unlike BN in eval it is not an affine function of the input and
has no zero-population floor. The metric's floor tracks affineness precisely.

λ at `prob` −0.162 → −0.173 across the 5× rep change, well inside the CI, so
the saturating output response is a real measurement.

## Reproduce

```
run.py --model vit_b_16 --reps 50 --seed 0 --save-run results/vit-b-16-r50-s0 --notes Reps companion to vit_b_16-r250-s0. ViT's affine prefix is conv_proj plus the first LayerNorm, so the floor should extend past the patch embedding exactly as BatchNorm extended it in vgg19_bn. --figures out/ --layers all
```

Code: `7bdf878d43c4`. Weights: torchvision vit_b_16 IMAGENET1K_V1.
