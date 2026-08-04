# resnet50-r50-s0-v2

`resnet50:IMAGENET1K_V2`, 50 reps/cell, best mean R² 0.937 at `layer2.1.relu`.

## What this run was for

Checkpoint lineage, cleanest available isolation: same architecture, same framework, no conversion, only the training recipe differs (V1 = 90 epochs, step LR, flip+RRC, 76.13% top-1; V2 = 600 epochs, cosine, TrivialAugment-wide, mixup/cutmix, label smoothing, EMA, 176px train crop, 80.86%). Pairs with results/resnet50-r50-s0 and against its own 3-seed noise floor (prob lambda -0.223/-0.230/-0.215).

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model resnet50:IMAGENET1K_V2 --reps 50 --seed 0 --save-run results/resnet50-r50-s0-v2 --notes Checkpoint lineage, cleanest available isolation: same architecture, same framework, no conversion, only the training recipe differs (V1 = 90 epochs, step LR, flip+RRC, 76.13% top-1; V2 = 600 epochs, cosine, TrivialAugment-wide, mixup/cutmix, label smoothing, EMA, 176px train crop, 80.86%). Pairs with results/resnet50-r50-s0 and against its own 3-seed noise floor (prob lambda -0.223/-0.230/-0.215). --figures out/ --layers all
```

Code: `a100e1e034c9`. Weights: torchvision resnet50 IMAGENET1K_V2.
