# resnet50-r50-s1

`resnet50`, 50 reps/cell, best mean R² 0.956 at `layer2.3.relu@2`.

## What this run was for

Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. resnet50 has no valid scrambled control (BatchNorm), so seed spread is the only null available for it.

## What it showed

**The mid-band structure reproduces on a new image sample.** Per-frequency λ
at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.22 | -0.19 | -0.39 | -0.47 | -0.27 | -0.40 | +0.04 | -0.01 |

Bands: low (1, 1.75) **+0.01**, mid (7–28) **-0.38**,
high (56, 75) **+0.02** — so this run dips in the mid band,
**0.39** below both ends. λ at `prob` = -0.230 at λ-R² 0.969,
mean R² 0.928.

Across the whole ResNet-50 series (4 runs, 3 seeds) the band contrast is
**+0.404 ± 0.014**, every run the same sign, and the effect is **30×** its own
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
run.py --model resnet50 --reps 50 --seed 1 --save-run results/resnet50-r50-s1 --notes Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. resnet50 has no valid scrambled control (BatchNorm), so seed spread is the only null available for it. --figures out/ --layers all
```

Code: `9101de3820ba`. Weights: torchvision resnet50 IMAGENET1K_V1.
