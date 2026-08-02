# vgg19-r50-s0-alllayers-in1k-gates

`vgg19`, 50 reps/cell, best mean R² 0.921 at `classifier.4`.

## What this run was for

Gate-flip instrument. IMAGENET1K_V1 VGG-19, all 45 taps. Same topology as the caffe run, lambda drifting 0.69 -> 0.16; predicts G rising at the taps where lambda falls.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-in1k-gates --notes Gate-flip instrument. IMAGENET1K_V1 VGG-19, all 45 taps. Same topology as the caffe run, lambda drifting 0.69 -> 0.16; predicts G rising at the taps where lambda falls. --figures out/ --layers all
```

Code: `cf72c73fb6b5`. Weights: torchvision vgg19 IMAGENET1K_V1.
