# vgg19-scramble-r50-s0-p1-alllayers-caffe

`vgg19`, 50 reps/cell, best mean R² 0.748 at `features.0`.

## What this run was for

Caffe scrambled control, permutation seed 1 with the orientation/phase draws held at seed 0. Isolates permutation variance, which the earlier IN1K sweep could not.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-scramble-r50-s0-p1-alllayers-caffe --notes Caffe scrambled control, permutation seed 1 with the orientation/phase draws held at seed 0. Isolates permutation variance, which the earlier IN1K sweep could not. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all --scramble --scramble-seed 1
```

Code: `25b0e1726840`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
