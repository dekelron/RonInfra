# vgg16-r50-s0

`vgg16`, 50 reps/cell, best mean R² 0.936 at `prob`.

## What this run was for

VGG-16 on torchvision IMAGENET1K_V1: the canonical half of the VGG-16 lineage pair. Tests whether the Caffe-vs-torchvision disagreement found on VGG-19 is a property of the two training recipes or of VGG-19 specifically. BN-free, so the weight scramble is a valid control here.

## What it showed

VGG-16 on torchvision `IMAGENET1K_V1`: `prob` λ **+0.093** (R² 0.960). The pair **reproduces the VGG-19 lineage result on a second architecture**: conv-stack median λ +0.664 (torchvision) against **+1.025** (Caffe), mean |Δλ| **0.353** over 39 shared taps, against VGG-19's +0.690 / +1.045 and 0.328. Caffe holds λ ≈ 1 — flatly linear in contrast — through the conv stack of both nets, so that is a property of the Oxford training recipe rather than of VGG-19. VGG-16 is BN-free, so its scrambled control is valid: mean log-R² 0.936/0.950 trained against 0.492/0.731 scrambled.

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 0 --save-run results/vgg16-r50-s0 --notes VGG-16 on torchvision IMAGENET1K_V1: the canonical half of the VGG-16 lineage pair. Tests whether the Caffe-vs-torchvision disagreement found on VGG-19 is a property of the two training recipes or of VGG-19 specifically. BN-free, so the weight scramble is a valid control here. --figures out/ --layers all
```

Code: `a100e1e034c9`. Weights: torchvision vgg16 IMAGENET1K_V1.
