# vgg19-r50-s0-alllayers-caffe

`vgg19`, 50 reps/cell, best mean R² 0.976 at `prob`.

## What this run was for

45 taps at 50 reps, converted Caffe. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-caffe --notes 45 taps at 50 reps, converted Caffe. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all
```

Code: `564e392d056c`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
