# vgg19-r50-s1-alllayers-caffe

`vgg19`, 50 reps/cell, best mean R² 0.979 at `prob`.

## What this run was for

Seed sweep for the per-frequency lambda structure at prob: seed 1 of 3, paper's checkpoint. Caffe is one of the two inverted cases (mid band more linear); seed 0 alone could not say whether that shape is reproducible.

## What it showed

**The mid-band structure reproduces on a new image sample.** Per-frequency λ
at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.13 | -0.10 | -0.01 | +0.23 | +0.13 | +0.25 | +0.03 | -0.22 |

Bands: low (1, 1.75) **+0.01**, mid (7–28) **+0.20**,
high (56, 75) **-0.09** — so this run peaks in the mid band,
**0.30** above both ends. λ at `prob` = +0.079 at λ-R² 0.990,
mean R² 0.979.

Across the whole VGG-19 (Caffe) series (4 runs, 3 seeds) the band contrast is
**-0.335 ± 0.025**, every run the same sign, and the effect is **14×** its own
seed-to-seed sd. This run is one member of that; the series conclusion is in
[`wiki/Results.md`](../../wiki/Results.md#λ-varies-more-across-frequency-than-it-does-across-architecture).

Why it was run: at seed 0 the 95% profile-F intervals resolved this band
structure for four of the six series and not for the other two, and a single
seed cannot distinguish "no structure" from "a loose interval". The sweep
settles it — **all six reproduce**, and the interval is 2.9–7.1× more
conservative than the measured across-seed sd. The mid band here is
the most linear part of the range, not the least.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 1 --save-run results/vgg19-r50-s1-alllayers-caffe --notes Seed sweep for the per-frequency lambda structure at prob: seed 1 of 3, paper's checkpoint. Caffe is one of the two inverted cases (mid band more linear); seed 0 alone could not say whether that shape is reproducible. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all
```

Code: `f6f5837c5fb0`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
