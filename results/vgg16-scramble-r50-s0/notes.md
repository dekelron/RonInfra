# vgg16-scramble-r50-s0

`vgg16`, 50 reps/cell, best mean R² 0.850 at `features.15`.

## What this run was for

VGG-16 on torchvision IMAGENET1K_V1: the canonical half of the VGG-16 lineage pair. Tests whether the Caffe-vs-torchvision disagreement found on VGG-19 is a property of the two training recipes or of VGG-19 specifically. BN-free, so the weight scramble is a valid control here.

## What it showed

Weight-scrambled control for [`vgg16-r50-s0`](../vgg16-r50-s0/notes.md): `prob` λ **+0.729** (R² 0.633) against the trained run's +0.093 (R² 0.960). VGG-16 has no BatchNorm, so unlike the ResNet/RegNet/MobileNet lineage pairs this control is valid — the permutation degrades rather than decalibrates.

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 0 --save-run results/vgg16-scramble-r50-s0 --notes VGG-16 on torchvision IMAGENET1K_V1: the canonical half of the VGG-16 lineage pair. Tests whether the Caffe-vs-torchvision disagreement found on VGG-19 is a property of the two training recipes or of VGG-19 specifically. BN-free, so the weight scramble is a valid control here. --figures out/ --layers all --scramble
```

Code: `a100e1e034c9`. Weights: torchvision vgg16 IMAGENET1K_V1.
