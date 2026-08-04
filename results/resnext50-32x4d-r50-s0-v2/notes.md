# resnext50-32x4d-r50-s0-v2

`resnext50_32x4d:IMAGENET1K_V2`, 50 reps/cell, best mean R² 0.962 at `layer2.1.relu`.

## What this run was for

Lineage pair 2 of 5 on the torchvision V1-vs-V2 axis; pairs with results/resnext50-32x4d-r50-s0. Same recipe delta as resnet50 (90-epoch step-LR against the 600-epoch TrivialAugment/mixup/cutmix/EMA recipe), different architecture, so it tests whether the lineage effect is architecture-specific.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model resnext50_32x4d:IMAGENET1K_V2 --reps 50 --seed 0 --save-run results/resnext50-32x4d-r50-s0-v2 --notes Lineage pair 2 of 5 on the torchvision V1-vs-V2 axis; pairs with results/resnext50-32x4d-r50-s0. Same recipe delta as resnet50 (90-epoch step-LR against the 600-epoch TrivialAugment/mixup/cutmix/EMA recipe), different architecture, so it tests whether the lineage effect is architecture-specific. --figures out/ --layers all
```

Code: `a100e1e034c9`. Weights: torchvision resnext50_32x4d IMAGENET1K_V2.
