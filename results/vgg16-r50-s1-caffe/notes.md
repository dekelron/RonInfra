# vgg16-r50-s1-caffe

`vgg16`, 50 reps/cell, best mean R² 0.959 at `classifier.4`.

## What this run was for

Seed replication for the VGG-16 lineage pair, Caffe half, seed 1 of 3. The VGG-16 result (conv-stack median lambda +0.664 torchvision -> +1.025 Caffe, mean |dlambda| 0.353) currently rests on one seed and borrows its noise scale from the VGG-19 sweep; this gives VGG-16 its own.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 1 --save-run results/vgg16-r50-s1-caffe --notes Seed replication for the VGG-16 lineage pair, Caffe half, seed 1 of 3. The VGG-16 result (conv-stack median lambda +0.664 torchvision -> +1.025 Caffe, mean |dlambda| 0.353) currently rests on one seed and borrows its noise scale from the VGG-19 sweep; this gives VGG-16 its own. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth --layers all
```

Code: `26f9c7c7e74a`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth.
