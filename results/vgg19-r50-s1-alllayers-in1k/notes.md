# vgg19-r50-s1-alllayers-in1k

`vgg19`, 50 reps/cell, best mean R² 0.923 at `classifier.4`.

## What this run was for

Seed sweep for the per-frequency lambda structure at prob: seed 1 of 3, IMAGENET1K_V1. This series carries the weakest of the four dips (+0.22) and its 7 cyc/img point is the least determined anywhere (interval 1.57 wide).

## What it showed

**The mid-band structure reproduces on a new image sample.** Per-frequency λ
at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.58 | +0.20 | -0.02 | +0.36 | -0.30 | -0.26 | +0.14 | +0.16 |

Bands: low (1, 1.75) **+0.39**, mid (7–28) **-0.06**,
high (56, 75) **+0.15** — so this run dips in the mid band,
**0.21** below both ends. λ at `prob` = +0.147 at λ-R² 0.953,
mean R² 0.921.

Across the whole VGG-19 (IN1K) series (4 runs, 3 seeds) the band contrast is
**+0.198 ± 0.022**, every run the same sign, and the effect is **9×** its own
seed-to-seed sd. This run is one member of that; the series conclusion is in
[`wiki/Results.md`](../../wiki/Results.md#λ-varies-more-across-frequency-than-it-does-across-architecture).

Why it was run: at seed 0 the 95% profile-F intervals resolved this band
structure for four of the six series and not for the other two, and a single
seed cannot distinguish "no structure" from "a loose interval". The sweep
settles it — **all six reproduce**, and the interval is 2.9–7.1× more
conservative than the measured across-seed sd. The mid band here is
saturating most, and rising toward linear at both ends.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 1 --save-run results/vgg19-r50-s1-alllayers-in1k --notes Seed sweep for the per-frequency lambda structure at prob: seed 1 of 3, IMAGENET1K_V1. This series carries the weakest of the four dips (+0.22) and its 7 cyc/img point is the least determined anywhere (interval 1.57 wide). --figures out/ --layers all
```

Code: `f6f5837c5fb0`. Weights: torchvision vgg19 IMAGENET1K_V1.
