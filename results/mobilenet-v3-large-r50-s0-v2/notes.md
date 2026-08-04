# mobilenet-v3-large-r50-s0-v2

`mobilenet_v3_large:IMAGENET1K_V2`, 50 reps/cell, best mean R² 0.965 at `features.7.block.0.0`.

## What this run was for

Lineage pair 3 of 5 on the torchvision V1-vs-V2 axis; pairs with results/mobilenet-v3-large-r50-s0. Note the V1 companion is one of the runs with a lambda pinned at a search bound, so read this pair per-tap with the pinned cells dropped.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model mobilenet_v3_large:IMAGENET1K_V2 --reps 50 --seed 0 --save-run results/mobilenet-v3-large-r50-s0-v2 --notes Lineage pair 3 of 5 on the torchvision V1-vs-V2 axis; pairs with results/mobilenet-v3-large-r50-s0. Note the V1 companion is one of the runs with a lambda pinned at a search bound, so read this pair per-tap with the pinned cells dropped. --figures out/ --layers all
```

Code: `a100e1e034c9`. Weights: torchvision mobilenet_v3_large IMAGENET1K_V2.
