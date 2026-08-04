# timm-vit-base-patch16-224-augreg-in21k-ft-in1k-r50-s0

`timm:vit_base_patch16_224.augreg_in21k_ft_in1k`, 50 reps/cell, best mean R² 0.960 at `blocks.6.mlp.fc1`.

## What this run was for

ViT-B/16 lineage pair, half 2: the AugReg weights (Steiner et al. 2021) -- same 21k-to-1k pipeline with dropout, stochastic depth, Mixup and RandAugment added. With the orig half this isolates augmentation/regularization at fixed architecture, data and objective; results/vit-b-16-r50-s0 is a third lineage of the same architecture (torchvision, 1k from scratch).

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model timm:vit_base_patch16_224.augreg_in21k_ft_in1k --reps 50 --seed 0 --save-run results/timm-vit-base-patch16-224-augreg-in21k-ft-in1k-r50-s0 --notes ViT-B/16 lineage pair, half 2: the AugReg weights (Steiner et al. 2021) -- same 21k-to-1k pipeline with dropout, stochastic depth, Mixup and RandAugment added. With the orig half this isolates augmentation/regularization at fixed architecture, data and objective; results/vit-b-16-r50-s0 is a third lineage of the same architecture (torchvision, 1k from scratch). --figures out/ --layers all --size 224
```

Code: `a100e1e034c9`. Weights: timm 1.0.28; vit_base_patch16_224.augreg_in21k_ft_in1k; hf_hub_id=timm/vit_base_patch16_224.augreg_in21k_ft_in1k; tag=augreg_in21k_ft_in1k.
