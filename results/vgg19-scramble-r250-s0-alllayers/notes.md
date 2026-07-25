# vgg19-scramble-r250-s0-alllayers

`vgg19`, 250 reps/cell, best mean R² 0.924 at `features.19`.

## What this run was for

Full grid tapping every leaf module (43 taps), for the continuous depth profile of the linear-vs-log index

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0-alllayers --notes Full grid tapping every leaf module (43 taps), for the continuous depth profile of the linear-vs-log index --figures out/ --layers all --scramble
```

Code: `089c28dec23a`. Weights: torchvision vgg19 IMAGENET1K_V1.
