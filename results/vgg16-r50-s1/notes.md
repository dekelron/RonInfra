# vgg16-r50-s1

`vgg16`, 50 reps/cell, best mean R² 0.934 at `prob`.

## What this run was for

Seed replication for the VGG-16 lineage pair, torchvision half, seed 1 of 3. Pairs with the Caffe seed-1 run so the lineage delta can be read against VGG-16's own seed-to-seed spread rather than VGG-19's.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 1 --save-run results/vgg16-r50-s1 --notes Seed replication for the VGG-16 lineage pair, torchvision half, seed 1 of 3. Pairs with the Caffe seed-1 run so the lineage delta can be read against VGG-16's own seed-to-seed spread rather than VGG-19's. --figures out/ --layers all
```

Code: `26f9c7c7e74a`. Weights: torchvision vgg16 IMAGENET1K_V1.
