# vgg19-r50-s0-alllayers-caffe-gates

`vgg19`, 50 reps/cell, best mean R² 0.976 at `prob`.

## What this run was for

Gate-flip instrument, first measurement. Caffe VGG-19, all 45 taps. Predicts G ~ 0 through the flat lambda ~ 1 conv stack.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-caffe-gates --notes Gate-flip instrument, first measurement. Caffe VGG-19, all 45 taps. Predicts G ~ 0 through the flat lambda ~ 1 conv stack. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all
```

Code: `cf72c73fb6b5`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
