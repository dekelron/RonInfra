# vit-b-16-r50-s1

`vit_b_16`, 50 reps/cell, best mean R² 0.955 at `encoder.layers.encoder_layer_2.mlp.0`.

## What this run was for

Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. vit_b_16 is the null case -- its band structure is NOT interval-resolved at seed 0, so this tests whether the flatness holds or was one sample.

## What it showed

**The mid-band structure reproduces on a new image sample.** Per-frequency λ
at `prob`:

| 1 | 1.75 | 3.5 | 7 | 14 | 28 | 56 | 75 |
|---|---|---|---|---|---|---|---|
| -0.17 | -0.51 | -0.27 | -0.08 | +0.04 | -0.15 | -0.29 | -0.11 |

Bands: low (1, 1.75) **-0.34**, mid (7–28) **-0.06**,
high (56, 75) **-0.20** — so this run peaks in the mid band,
**0.28** above both ends. λ at `prob` = -0.159 at λ-R² 0.928,
mean R² 0.900.

Across the whole ViT-B/16 series (4 runs, 3 seeds) the band contrast is
**-0.284 ± 0.022**, every run the same sign, and the effect is **13×** its own
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
run.py --model vit_b_16 --reps 50 --seed 1 --save-run results/vit-b-16-r50-s1 --notes Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. vit_b_16 is the null case -- its band structure is NOT interval-resolved at seed 0, so this tests whether the flatness holds or was one sample. --figures out/ --layers all
```

Code: `9101de3820ba`. Weights: torchvision vit_b_16 IMAGENET1K_V1.
