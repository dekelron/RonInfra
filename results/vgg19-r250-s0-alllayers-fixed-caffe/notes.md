# vgg19-r250-s0-alllayers-fixed-caffe

`vgg19`, 250 reps/cell, best mean R² 0.980 at `prob`.

## What this run was for

Every leaf module tapped (43 + `logits`/`prob`), converted Caffe weights, trained. One quarter of the 2x2 that locates *where* along depth the response changes shape, and whether the two checkpoints differ there.

## What it showed

`logness` at the ends: -0.224 at `features.0` (conv1_1) and +0.227 at `prob`.
The full profile is the point, not any single layer -- see
[Results](../../wiki/Results.md#where-the-log-response-appears-along-depth).

Stays linear-in-contrast for the whole network and crosses only at
`classifier.4`, the ReLU after fc7: -0.241 -> +0.133, a jump of 0.374
at one rectification.

Its `weights_sha256` is `2c7887c87148`, matching the sandbox
conversion behind [`vgg19-r50-s0`](../vgg19-r50-s0/notes.md) while the
file hash differs -- the conversion reproduces across machines.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-r250-s0-alllayers-fixed-caffe --notes Caffe half of the checkpoint comparison: all 43 taps, fixed hook, same settings as the IMAGENET1K_V1 pair so the two are measured identically --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all
```

Code: `368ba36a1e84`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
