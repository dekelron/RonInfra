# vgg19-r50-s0-alllayers-units-caffe

`vgg19`, 50 reps/cell, best mean R² 0.976 at `prob`.

## What this run was for

Per-unit surfaces at the head taps. D is an L1 norm, so lambda cannot see a response whose carriers rotate with contrast; this asks whether that is what happens at the fc7 ReLU where lambda drops 1.110 -> 0.231. Also the first read of scale_matched (|z0| against the perturbation scale) at classifier.3, the rectifier input.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-units-caffe --notes Per-unit surfaces at the head taps. D is an L1 norm, so lambda cannot see a response whose carriers rotate with contrast; this asks whether that is what happens at the fc7 ReLU where lambda drops 1.110 -> 0.231. Also the first read of scale_matched (|z0| against the perturbation scale) at classifier.3, the rectifier input. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all --unit-taps classifier.0,classifier.1,classifier.3,classifier.4,prob
```

Code: `35e5237476b1`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
