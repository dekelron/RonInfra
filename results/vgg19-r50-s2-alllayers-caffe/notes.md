# vgg19-r50-s2-alllayers-caffe

`vgg19`, 50 reps/cell, best mean R² 0.977 at `prob`.

## What this run was for

Seed sweep for the per-frequency lambda structure at prob: seed 2 of 3, paper's checkpoint. Caffe is one of the two inverted cases (mid band more linear); seed 0 alone could not say whether that shape is reproducible.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 2 --save-run results/vgg19-r50-s2-alllayers-caffe --notes Seed sweep for the per-frequency lambda structure at prob: seed 2 of 3, paper's checkpoint. Caffe is one of the two inverted cases (mid band more linear); seed 0 alone could not say whether that shape is reproducible. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all
```

Code: `f6f5837c5fb0`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
