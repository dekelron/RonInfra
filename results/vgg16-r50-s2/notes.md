# vgg16-r50-s2

`vgg16`, 50 reps/cell, best mean R² 0.934 at `prob`.

## What this run was for

Seed replication for the VGG-16 lineage pair, torchvision half, seed 2 of 3.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 2 --save-run results/vgg16-r50-s2 --notes Seed replication for the VGG-16 lineage pair, torchvision half, seed 2 of 3. --figures out/ --layers all
```

Code: `26f9c7c7e74a`. Weights: torchvision vgg16 IMAGENET1K_V1.
