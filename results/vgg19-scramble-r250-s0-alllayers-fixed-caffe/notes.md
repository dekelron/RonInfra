# vgg19-scramble-r250-s0-alllayers-fixed-caffe

`vgg19`, 250 reps/cell, best mean R² 0.770 at `features.22`.

## What this run was for

Every leaf module tapped (43 + `logits`/`prob`), converted Caffe weights, scrambled. One quarter of the 2x2 that locates *where* along depth the response changes shape, and whether the two checkpoints differ there.

## What it showed

`logness` at the ends: -0.225 at `features.0` (conv1_1) and -0.381 at `prob`.
The full profile is the point, not any single layer -- see
[Results](../../wiki/Results.md#where-the-log-response-appears-along-depth).

Flat at about -0.38 throughout and **never crosses zero**: this control
does not become log-like anywhere. That is why the trained-minus-
scrambled gap is large on Caffe and near zero on `IMAGENET1K_V1`,
where the control does reach the same place as the trained net.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0-alllayers-fixed-caffe --notes Caffe half of the checkpoint comparison: all 43 taps, fixed hook, same settings as the IMAGENET1K_V1 pair so the two are measured identically --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all --scramble
```

Code: `368ba36a1e84`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
