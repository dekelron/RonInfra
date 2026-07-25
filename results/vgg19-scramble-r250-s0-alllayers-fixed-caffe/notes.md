# vgg19-scramble-r250-s0-alllayers-fixed-caffe

`vgg19`, 250 reps/cell, best mean R² 0.770 at `features.22`.

## What this run was for

Caffe half of the checkpoint comparison: all 43 taps, fixed hook, same settings as the IMAGENET1K_V1 pair so the two are measured identically

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0-alllayers-fixed-caffe --notes Caffe half of the checkpoint comparison: all 43 taps, fixed hook, same settings as the IMAGENET1K_V1 pair so the two are measured identically --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all --scramble
```

Code: `368ba36a1e84`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
