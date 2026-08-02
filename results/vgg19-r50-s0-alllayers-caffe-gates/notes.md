# vgg19-r50-s0-alllayers-caffe-gates

`vgg19`, 50 reps/cell, best mean R² 0.976 at `prob`.

## What this run was for

Gate-flip instrument, first measurement. Caffe VGG-19, all 45 taps. Predicts G ~ 0 through the flat lambda ~ 1 conv stack.

## What it showed

**The gate-freezing reading is falsified, and this run is what kills it.**

The prediction under test: a ReLU net is piecewise linear, so while no rectifier
changes state `D = c·|J·g|` exactly and λ = 1 — hence this checkpoint's flat
λ ≈ 1.06 conv stack should be a stack in which nothing switches.

It is not. Across the 37 `features.*` taps `G`, the fraction of units whose sign
the grating flips, averages **31.3%** at full contrast (range 6.6–47.4%) and
**15.0%** even at `c = 1/128`. 33 of the 37 sit within 0.15 of λ = 1, and among
those the median `G` is **35.0%**. Over the whole 45-tap set: **36 taps
near-linear-but-flipping, 0 near-linear-and-quiet** — the cell the prediction
requires is empty.

`r(log₁₀ G, λ)` = **+0.107** across the stack: no relation at all.

What survives is positive homogeneity. `ReLU(c·g) = c·ReLU(g)`, so a unit whose
gray pre-activation sits *at* threshold flips on nearly every draw and still
contributes a response exactly linear in `c`. Frozen gates are sufficient for
λ = 1, not necessary.

Instrument check, passed: a conv tap and the ReLU immediately after it report
**bit-identical** `G` (5/5 pairs), as `sign(z)` and `ReLU(z) > 0` must. `prob`
reads `G` = 0.000% with `open` = 100% trivially — probabilities are positive, so
there is no gate there to count.

One seed, `--reps 50`, pretrained only. See `wiki/Results.md`.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-caffe-gates --notes Gate-flip instrument, first measurement. Caffe VGG-19, all 45 taps. Predicts G ~ 0 through the flat lambda ~ 1 conv stack. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all
```

Code: `cf72c73fb6b5`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
