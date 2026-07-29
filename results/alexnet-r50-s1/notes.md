# alexnet-r50-s1

`alexnet`, 50 reps/cell, best mean R² 0.962 at `prob`.

## What this run was for

Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. Tests whether the mid-band (7-28 cyc/img) saturation survives a new image sample, which the r50/r250 pair at seed 0 could not.

## What it showed

**The mid-band structure reproduces on a new image sample.** Per-frequency λ
at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| +0.32 | +0.15 | -0.03 | -0.11 | -0.17 | -0.40 | +0.09 | +0.21 |

Bands: low (1, 1.75) **+0.23**, mid (7–28) **-0.22**,
high (56, 75) **+0.15** — so this run dips in the mid band,
**0.38** below both ends. λ at `prob` = +0.029 at λ-R² 0.983,
mean R² 0.962.

Across the whole AlexNet series (4 runs, 3 seeds) the band contrast is
**+0.409 ± 0.021**, every run the same sign, and the effect is **19×** its own
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
run.py --model alexnet --reps 50 --seed 1 --save-run results/alexnet-r50-s1 --notes Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. Tests whether the mid-band (7-28 cyc/img) saturation survives a new image sample, which the r50/r250 pair at seed 0 could not. --figures out/ --layers all
```

Code: `9101de3820ba`. Weights: torchvision alexnet IMAGENET1K_V1.
