# vgg19-bn-scramble-r50-s0

`vgg19_bn`, 50 reps/cell, best mean R² 0.806 at `features.45`.

## What this run was for

Reps companion to vgg19-bn-r250-s0: separates a real locally-linear response from an empty tap, as the vgg19 r50/r250 pair did (only features.0 was on the floor there).

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19_bn --reps 50 --seed 0 --save-run results/vgg19-bn-scramble-r50-s0 --notes Reps companion to vgg19-bn-r250-s0: separates a real locally-linear response from an empty tap, as the vgg19 r50/r250 pair did (only features.0 was on the floor there). --figures out/ --layers all --scramble
```

Code: `f9ab3861976a`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
