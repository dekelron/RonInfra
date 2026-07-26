# data-r250-s0

`data`, 250 reps/cell, best mean R² 0.754 at `data`.

## What this run was for

Raw image pixels — the first of the four representations in the paper's
Figure 3b (`data`, `conv1_1`, `fc8`, `prob`), which had no back-end here until
now. It doubles as the reference **noise floor** for the metric.

## What it showed

Phase is drawn `U[0, 2π)`, so `E[grating] = gray` *exactly*. D is the distance
of the class-**mean** representation from gray, so at any layer that is an
affine function of the input the population value of D is identically **zero**
and a finite run measures only sampling noise. Raw pixels are the purest such
layer, so this run is what that noise looks like:

| | value |
|---|---|
| λ | **+0.925** [+0.84, +1.02] |
| power-family R² | **0.985** |
| mean R² vs log c | **0.754** |
| D at c = 1 | 1.82e-02 |
| frequency spread | 1.24× at lowest c, 1.78× at c = 1 |

Two things follow, and both matter for reading a depth profile.

**The floor is exactly linear in contrast.** `D = c · mean_i|ḡ_i|` with `ḡ`
independent of `c`, so λ ≈ 1 at a high power-family R² is what an *empty* tap
looks like — not only what a linearly-responding one looks like. For reference,
R² of a perfectly linear response against log c on this 14-point grid is
**0.736**, a property of the grid alone.

**It falls as 1/√reps.** Against [`data-r50-s0`](../data-r50-s0/notes.md), which
changes nothing but the repetition count, the median ratio of the two surfaces
is **2.237** against √5 = **2.236**. That is the cheap per-layer test for
whether a tap carries any signal at all: a real response holds D when reps
change, a floor does not.

**`features.0` in VGG-19 is this run.** It is the only tap upstream of every
nonlinearity, and across all four 45-tap runs it reproduces these numbers:

| run | λ | power R² | log R² |
|---|---|---|---|
| this run (no model at all) | +0.925 | 0.985 | 0.754 |
| trained, Caffe | +0.922 | 0.985 | 0.756 |
| trained, `IMAGENET1K_V1` | +0.923 | 0.986 | 0.756 |
| scrambled, Caffe | +0.926 | 0.985 | 0.754 |
| scrambled, `IMAGENET1K_V1` | +0.926 | 0.985 | 0.754 |

Trained and scrambled, two different checkpoints, and a model-free control all
agree to three decimals — because none of them is measuring the network. See
[Results](../../wiki/Results.md#the-metric-has-a-noise-floor-and-features0-is-sitting-on-it).

Deeper taps are **not** covered by this argument: they sit downstream of a
ReLU, so `E[a(x)] ≠ a(gray)` and they can carry real signal even when they are
convolutions. `features.19` holds its D across a 5× rep change, so it does.

## Reproduce

```
python -m log_response.run --model data --save-run results/data-r250-s0
```

Code: `3ecbdfe9cdb9`. Weights: none (raw pixels) — weight-free by construction,
so `pretrained_verified` is `null`, not `false`.
