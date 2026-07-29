# vgg19-bn-r50-s2

`vgg19_bn`, 50 reps/cell, best mean R² 0.931 at `features.26`.

## What this run was for

Seed sweep for the per-frequency lambda dip at prob: seed 2 of 3. vgg19_bn carries the largest dip (+0.54) and has no valid scrambled control, so seed spread is the only null available for it.

## What it showed

**The mid-band structure reproduces on a new image sample.** Per-frequency λ
at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.51 | +0.23 | -0.20 | -0.50 | -1.31 | -0.68 | -0.28 | -0.35 |

Bands: low (1, 1.75) **+0.37**, mid (7–28) **-0.83**,
high (56, 75) **-0.31** — so this run dips in the mid band,
**0.52** below both ends. λ at `prob` = -0.315 at λ-R² 0.950,
mean R² 0.837.

Across the whole VGG-19+BN series (4 runs, 3 seeds) the band contrast is
**+0.533 ± 0.036**, every run the same sign, and the effect is **15×** its own
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
run.py --model vgg19_bn --reps 50 --seed 2 --save-run results/vgg19-bn-r50-s2 --notes Seed sweep for the per-frequency lambda dip at prob: seed 2 of 3. vgg19_bn carries the largest dip (+0.54) and has no valid scrambled control, so seed spread is the only null available for it. --figures out/ --layers all
```

Code: `9101de3820ba`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
