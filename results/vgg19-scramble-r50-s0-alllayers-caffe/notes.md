# vgg19-scramble-r50-s0-alllayers-caffe

`vgg19`, 50 reps/cell, best mean R² 0.771 at `features.22`.

## What this run was for

45 taps at 50 reps, converted Caffe. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric.

## What it showed

Control for [`vgg19-r50-s0-alllayers-caffe`](../vgg19-r50-s0-alllayers-caffe/notes.md).

Same verdict on the noise question: **1 of 45** taps is on the floor
(`features.0`, D(50)/D(250) = **2.234**, 100% noise); the largest noise
fraction anywhere else is **2.9%**. So the scrambled net's supralinear
classifier (λ ≈ +2.76, R² 0.971) is a real measurement too, not an artifact.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-scramble-r50-s0-alllayers-caffe --notes 45 taps at 50 reps, converted Caffe. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all --scramble
```

Code: `564e392d056c`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
