# data-r50-s0

`data`, 50 reps/cell, best mean R² 0.749 at `data`.

## What this run was for

The companion to [`data-r250-s0`](../data-r250-s0/notes.md): identical in every
respect but the repetition count, so the **1/√reps** scaling of the metric's
noise floor is checkable from committed surfaces rather than asserted.

## What it showed

Median ratio of the two surfaces, cell by cell:

```
D(50) / D(250)  =  2.237        (√5 = 2.236)
```

A real response is invariant to the repetition count; a noise floor scales as
1/√reps. This is the whole test, and it costs one extra run at a fifth of the
price.

The shape is unchanged, as it must be — λ **+0.94** [+0.79, +1.04] against
+0.925 at 250 reps, power-family R² 0.983 against 0.985. Only the *magnitude*
moves: D at c = 1 is 4.06e-02 here against 1.82e-02 at 250 reps.

Applied to the real runs, this is what says `features.0` is a floor and
`features.19` is not: across the r50/r250 pair on `IMAGENET1K_V1`,
`features.0` moves by up to 2.19× while `features.19`, `classifier.3`,
`logits` and `prob` all sit flat at 1.00.

## Reproduce

```
python -m log_response.run --model data --reps 50 --save-run results/data-r50-s0
```

Code: `3ecbdfe9cdb9`. Weights: none (raw pixels).
