# mobilenet-v2-r50-s0-v2

`mobilenet_v2:IMAGENET1K_V2`, 50 reps/cell, best mean R² 0.971 at `features.10.conv.1.1`.

## What this run was for

Lineage pair 4 of 5 on the torchvision V1-vs-V2 axis; pairs with results/mobilenet-v2-r50-s0.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model mobilenet_v2:IMAGENET1K_V2 --reps 50 --seed 0 --save-run results/mobilenet-v2-r50-s0-v2 --notes Lineage pair 4 of 5 on the torchvision V1-vs-V2 axis; pairs with results/mobilenet-v2-r50-s0. --figures out/ --layers all
```

Code: `a100e1e034c9`. Weights: torchvision mobilenet_v2 IMAGENET1K_V2.
