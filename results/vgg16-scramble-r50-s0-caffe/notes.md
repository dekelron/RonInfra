# vgg16-scramble-r50-s0-caffe

`vgg16`, 50 reps/cell, best mean R² 0.860 at `features.20`.

## What this run was for

VGG-16 on the original Oxford/Caffe weights (Keras HDF5 port), converted on the runner with the preprocessing fold verified. The Caffe half of the VGG-16 lineage pair -- the direct analogue of the VGG-19 result, on a different architecture through the same conversion path.

## What it showed

Weight-scrambled control for [`vgg16-r50-s0-caffe`](../vgg16-r50-s0-caffe/notes.md): `prob` λ **+0.861** (R² 0.974) against the trained run's +0.135 (R² 0.969). VGG-16 has no BatchNorm, so unlike the ResNet/RegNet/MobileNet lineage pairs this control is valid — the permutation degrades rather than decalibrates.

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 0 --save-run results/vgg16-scramble-r50-s0-caffe --notes VGG-16 on the original Oxford/Caffe weights (Keras HDF5 port), converted on the runner with the preprocessing fold verified. The Caffe half of the VGG-16 lineage pair -- the direct analogue of the VGG-19 result, on a different architecture through the same conversion path. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth --layers all --scramble
```

Code: `a100e1e034c9`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth.
