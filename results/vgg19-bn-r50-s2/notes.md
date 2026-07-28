# vgg19-bn-r50-s2

`vgg19_bn`, 50 reps/cell, best mean R² 0.931 at `features.26`.

## What this run was for

Seed sweep for the per-frequency lambda dip at prob: seed 2 of 3. vgg19_bn carries the largest dip (+0.54) and has no valid scrambled control, so seed spread is the only null available for it.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19_bn --reps 50 --seed 2 --save-run results/vgg19-bn-r50-s2 --notes Seed sweep for the per-frequency lambda dip at prob: seed 2 of 3. vgg19_bn carries the largest dip (+0.54) and has no valid scrambled control, so seed spread is the only null available for it. --figures out/ --layers all
```

Code: `9101de3820ba`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
