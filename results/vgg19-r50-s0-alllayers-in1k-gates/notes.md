# vgg19-r50-s0-alllayers-in1k-gates

`vgg19`, 50 reps/cell, best mean R² 0.921 at `classifier.4`.

## What this run was for

Gate-flip instrument. IMAGENET1K_V1 VGG-19, all 45 taps. Same topology as the caffe run, lambda drifting 0.69 -> 0.16; predicts G rising at the taps where lambda falls.

## What it showed

**The companion that shows `G` does not explain the checkpoint difference.**

Matched tap-for-tap against `vgg19-r50-s0-alllayers-caffe-gates` — same
architecture, same 37 `features.*` names, only the weight lineage differs:

```
λ      +1.055 → +0.759   (Δ −0.296)
G      31.3%  → 28.1%    (Δ −3.2 points)
r(Δλ, ΔG) = −0.209        r(Δλ, Δopen) = +0.004
```

So λ moves by a third of a unit while the gate-flip rate barely moves, and the
per-tap changes in the two are uncorrelated. Within this run `r(log₁₀ G, λ)` =
**+0.209** — again nothing.

The fairer test counts only gates that switch *during* the sweep,
`G(c_max) − G(c_min)`, since a unit already flipped at the lowest contrast never
switches within the sampled range. It correlates with λ at **+0.414** here:
resolved, and *positive* — the taps where more gates switch are the **more**
linear ones, the opposite of what the reading predicts.

One seed, `--reps 50`, pretrained only. See `wiki/Results.md`.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-in1k-gates --notes Gate-flip instrument. IMAGENET1K_V1 VGG-19, all 45 taps. Same topology as the caffe run, lambda drifting 0.69 -> 0.16; predicts G rising at the taps where lambda falls. --figures out/ --layers all
```

Code: `cf72c73fb6b5`. Weights: torchvision vgg19 IMAGENET1K_V1.
