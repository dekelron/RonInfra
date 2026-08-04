# vgg16-r50-s2-caffe

`vgg16`, 50 reps/cell, best mean R² 0.959 at `classifier.4`.

## What this run was for

Seed replication for the VGG-16 lineage pair, Caffe half, seed 2 of 3.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 2 --save-run results/vgg16-r50-s2-caffe --notes Seed replication for the VGG-16 lineage pair, Caffe half, seed 2 of 3. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth --layers all
```

Code: `26f9c7c7e74a`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth.
