# vit-b-16-r50-s1

`vit_b_16`, 50 reps/cell, best mean R² 0.955 at `encoder.layers.encoder_layer_2.mlp.0`.

## What this run was for

Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. vit_b_16 is the null case -- its band structure is NOT interval-resolved at seed 0, so this tests whether the flatness holds or was one sample.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vit_b_16 --reps 50 --seed 1 --save-run results/vit-b-16-r50-s1 --notes Seed sweep for the per-frequency lambda dip at prob: seed 1 of 3. vit_b_16 is the null case -- its band structure is NOT interval-resolved at seed 0, so this tests whether the flatness holds or was one sample. --figures out/ --layers all
```

Code: `9101de3820ba`. Weights: torchvision vit_b_16 IMAGENET1K_V1.
