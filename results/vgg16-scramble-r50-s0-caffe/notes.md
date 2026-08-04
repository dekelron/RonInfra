# vgg16-scramble-r50-s0-caffe

`vgg16`, 50 reps/cell, best mean R² 0.860 at `features.20`.

## What this run was for

VGG-16 on the original Oxford/Caffe weights (Keras HDF5 port), converted on the runner with the preprocessing fold verified. The Caffe half of the VGG-16 lineage pair -- the direct analogue of the VGG-19 result, on a different architecture through the same conversion path.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 0 --save-run results/vgg16-scramble-r50-s0-caffe --notes VGG-16 on the original Oxford/Caffe weights (Keras HDF5 port), converted on the runner with the preprocessing fold verified. The Caffe half of the VGG-16 lineage pair -- the direct analogue of the VGG-19 result, on a different architecture through the same conversion path. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth --layers all --scramble
```

Code: `a100e1e034c9`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth.
