# vgg19-bn-scramble-r250-s0

`vgg19_bn`, 250 reps/cell, best mean R² 0.814 at `features.45`.

## What this run was for

VGG-19 + BatchNorm depth profile, 61 taps. Identical topology and ReLU count to vgg19; BN is affine in eval so it adds no gates, only moves the operating point. Tests whether lambda through the conv stack is set by the operating point rather than the rectifier count.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19_bn --reps 250 --seed 0 --save-run results/vgg19-bn-scramble-r250-s0 --notes VGG-19 + BatchNorm depth profile, 61 taps. Identical topology and ReLU count to vgg19; BN is affine in eval so it adds no gates, only moves the operating point. Tests whether lambda through the conv stack is set by the operating point rather than the rectifier count. --figures out/ --layers all --scramble
```

Code: `f9ab3861976a`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
