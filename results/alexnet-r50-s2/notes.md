# alexnet-r50-s2

`alexnet`, 50 reps/cell, best mean R² 0.961 at `prob`.

## What this run was for

Seed sweep for the per-frequency lambda dip at prob: seed 2 of 3. Tests whether the mid-band (7-28 cyc/img) saturation survives a new image sample, which the r50/r250 pair at seed 0 could not.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model alexnet --reps 50 --seed 2 --save-run results/alexnet-r50-s2 --notes Seed sweep for the per-frequency lambda dip at prob: seed 2 of 3. Tests whether the mid-band (7-28 cyc/img) saturation survives a new image sample, which the r50/r250 pair at seed 0 could not. --figures out/ --layers all
```

Code: `9101de3820ba`. Weights: torchvision alexnet IMAGENET1K_V1.
