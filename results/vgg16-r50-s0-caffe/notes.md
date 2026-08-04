# vgg16-r50-s0-caffe

`vgg16`, 50 reps/cell, best mean R² 0.959 at `classifier.4`.

## What this run was for

VGG-16 on the original Oxford/Caffe weights (Keras HDF5 port), converted on the runner with the preprocessing fold verified. The Caffe half of the VGG-16 lineage pair -- the direct analogue of the VGG-19 result, on a different architecture through the same conversion path.

## What it showed

VGG-16 on the original Oxford/Caffe weights: `prob` λ **+0.135** (R² 0.969). The pair **reproduces the VGG-19 lineage result on a second architecture**: conv-stack median λ +0.664 (torchvision) against **+1.025** (Caffe), mean |Δλ| **0.353** over 39 shared taps, against VGG-19's +0.690 / +1.045 and 0.328. Caffe holds λ ≈ 1 — flatly linear in contrast — through the conv stack of both nets, so that is a property of the Oxford training recipe rather than of VGG-19. VGG-16 is BN-free, so its scrambled control is valid: mean log-R² 0.936/0.950 trained against 0.492/0.731 scrambled.

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 0 --save-run results/vgg16-r50-s0-caffe --notes VGG-16 on the original Oxford/Caffe weights (Keras HDF5 port), converted on the runner with the preprocessing fold verified. The Caffe half of the VGG-16 lineage pair -- the direct analogue of the VGG-19 result, on a different architecture through the same conversion path. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth --layers all
```

Code: `a100e1e034c9`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth.
