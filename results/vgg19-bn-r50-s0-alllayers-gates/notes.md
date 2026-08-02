# vgg19-bn-r50-s0-alllayers-gates

`vgg19_bn`, 50 reps/cell, best mean R² 0.928 at `features.26`.

## What this run was for

Gate-flip instrument. vgg19_bn, all taps. The case that broke 'rectifications carry it': conv stack at lambda -0.071 with zero rectifiers added over plain VGG-19.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vgg19_bn --reps 50 --seed 0 --save-run results/vgg19-bn-r50-s0-alllayers-gates --notes Gate-flip instrument. vgg19_bn, all taps. The case that broke 'rectifications carry it': conv stack at lambda -0.071 with zero rectifiers added over plain VGG-19. --figures out/ --layers all
```

Code: `cf72c73fb6b5`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
