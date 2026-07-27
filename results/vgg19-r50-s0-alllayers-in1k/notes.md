# vgg19-r50-s0-alllayers-in1k

`vgg19`, 50 reps/cell, best mean R² 0.921 at `classifier.4`.

## What this run was for

45 taps at 50 reps, IMAGENET1K_V1. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-in1k --notes 45 taps at 50 reps, IMAGENET1K_V1. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric. --figures out/ --layers all
```

Code: `564e392d056c`. Weights: torchvision vgg19 IMAGENET1K_V1.
