# timm-vit-base-patch16-224-orig-in21k-ft-in1k-r50-s0

`timm:vit_base_patch16_224.orig_in21k_ft_in1k`, 50 reps/cell, best mean R² 0.951 at `blocks.1.norm1`.

## What this run was for

ViT-B/16 lineage pair, half 1: the ORIGINAL Dosovitskiy et al. ViT weights, ImageNet-21k pretrained then fine-tuned to 1k. Same architecture, same data, same task as the augreg half -- augmentation and regularization are the only variable, which is exactly what the AugReg paper varied.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model timm:vit_base_patch16_224.orig_in21k_ft_in1k --reps 50 --seed 0 --save-run results/timm-vit-base-patch16-224-orig-in21k-ft-in1k-r50-s0 --notes ViT-B/16 lineage pair, half 1: the ORIGINAL Dosovitskiy et al. ViT weights, ImageNet-21k pretrained then fine-tuned to 1k. Same architecture, same data, same task as the augreg half -- augmentation and regularization are the only variable, which is exactly what the AugReg paper varied. --figures out/ --layers all --size 224
```

Code: `a100e1e034c9`. Weights: timm 1.0.28; vit_base_patch16_224.orig_in21k_ft_in1k; hf_hub_id=timm/vit_base_patch16_224.orig_in21k_ft_in1k; tag=orig_in21k_ft_in1k.
