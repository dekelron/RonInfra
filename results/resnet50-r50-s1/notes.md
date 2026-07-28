# resnet50-r50-s1

`resnet50`, 50 reps/cell, best mean R² 0.956 at `layer2.3.relu@2`.

## What this run was for

Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. resnet50 has no valid scrambled control (BatchNorm), so seed spread is the only null available for it.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model resnet50 --reps 50 --seed 1 --save-run results/resnet50-r50-s1 --notes Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. resnet50 has no valid scrambled control (BatchNorm), so seed spread is the only null available for it. --figures out/ --layers all
```

Code: `9101de3820ba`. Weights: torchvision resnet50 IMAGENET1K_V1.
