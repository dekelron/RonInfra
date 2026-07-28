# vgg19-bn-scramble-r50-s0

`vgg19_bn`, 50 reps/cell, best mean R² 0.806 at `features.45`.

## What this run was for

Reps companion to vgg19-bn-r250-s0: separates a real locally-linear response from an empty tap, as the vgg19 r50/r250 pair did (only features.0 was on the floor there).

## What it showed

Reps companion to
[`vgg19-bn-scramble-r250-s0`](../vgg19-bn-scramble-r250-s0/notes.md), and it
confirms that run's reading that the λ there means nothing.

`prob` λ goes **−2.794 → −1.258** across the rep change — a move of 1.5, where
every other run in the repo moves by less than 0.1 — and the 95% interval opens
to **[−3.00, +1.32]**, i.e. very nearly the entire search range. That is the
property the λ statistic was adopted for: *pure noise returns the whole range
rather than a confident number*. λ-R² is 0.487.

So the scrambled BN control has no measurable contrast exponent at `prob`. See
the parent run for why — the scramble decalibrates BatchNorm rather than
degrading it, saturating the softmax.

## Reproduce

```
run.py --model vgg19_bn --reps 50 --seed 0 --save-run results/vgg19-bn-scramble-r50-s0 --notes Reps companion to vgg19-bn-r250-s0: separates a real locally-linear response from an empty tap, as the vgg19 r50/r250 pair did (only features.0 was on the floor there). --figures out/ --layers all --scramble
```

Code: `f9ab3861976a`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
