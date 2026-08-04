# timm-vit-base-patch16-224-augreg-in21k-ft-in1k-r50-s0

`timm:vit_base_patch16_224.augreg_in21k_ft_in1k`, 50 reps/cell, best mean R² 0.960 at `blocks.6.mlp.fc1`.

## What this run was for

ViT-B/16 lineage pair, half 2: the AugReg weights (Steiner et al. 2021) -- same 21k-to-1k pipeline with dropout, stochastic depth, Mixup and RandAugment added. With the orig half this isolates augmentation/regularization at fixed architecture, data and objective; results/vit-b-16-r50-s0 is a third lineage of the same architecture (torchvision, 1k from scratch).

## What it showed

AugReg against the original ViT weights moves `prob` λ -0.096 → **+0.232** (R² 0.874 → 0.924), a shift of **+0.328** — **46×** ViT-B/16's three-seed sampling sd of 0.0072 — and **mean |Δλ| 0.380 over 189 shared taps**, the largest depth-profile shift of any pair measured here. Profile correlation is also the lowest at **+0.387**: AugReg reshapes the profile rather than shifting it, with the last blocks moving most (`blocks.11.attn.proj` −0.659 → +1.262). Architecture, data and objective are all held fixed, so this is augmentation and regularization alone. Caveat: both tags normalise natively at 0.5/0.5 and both ran under the shared ImageNet constants — identical on the two sides, but off-native for both.

## Reproduce

```
run.py --model timm:vit_base_patch16_224.augreg_in21k_ft_in1k --reps 50 --seed 0 --save-run results/timm-vit-base-patch16-224-augreg-in21k-ft-in1k-r50-s0 --notes ViT-B/16 lineage pair, half 2: the AugReg weights (Steiner et al. 2021) -- same 21k-to-1k pipeline with dropout, stochastic depth, Mixup and RandAugment added. With the orig half this isolates augmentation/regularization at fixed architecture, data and objective; results/vit-b-16-r50-s0 is a third lineage of the same architecture (torchvision, 1k from scratch). --figures out/ --layers all --size 224
```

Code: `a100e1e034c9`. Weights: timm 1.0.28; vit_base_patch16_224.augreg_in21k_ft_in1k; hf_hub_id=timm/vit_base_patch16_224.augreg_in21k_ft_in1k; tag=augreg_in21k_ft_in1k.
