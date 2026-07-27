# vgg19-r50-s0-alllayers-caffe

`vgg19`, 50 reps/cell, best mean R² 0.976 at `prob`.

## What this run was for

45 taps at 50 reps, converted Caffe. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric.

## What it showed

**The conv stack is real.** Against
[`vgg19-r250-s0-alllayers-fixed-caffe`](../vgg19-r250-s0-alllayers-fixed-caffe/notes.md),
only **1 of 45** taps falls with repetition count:

| tap | D(50)/D(250) | noise fraction | reading |
|---|---|---|---|
| `features.0` | **2.222** | 98% | the floor (√5 = 2.236) |
| `features.1` | 1.050 | 3% | signal |
| `features.19` | 1.006 | 0% | signal |
| `classifier.4` | 1.002 | 0% | signal |
| `prob` | 1.001 | 0% | signal |

Outside `features.0/1/2` the largest noise fraction anywhere in the network is
**3.4%**. So the flat λ ≈ 1 conv stack — 26 taps sitting within 0.15 of the
noise-floor λ, which was the reason for this run — is **measuring a genuinely
locally-linear response**, not empty. The competing explanation is dead.

**The two orderings agree wherever the tap is clean**, which is the other half
of the check and needs no second run:

```
|λ − λ_mod|   features.0  0.05     (98% noise)
              features.1  0.18     (3%)
              features.19 0.00     (0%)
              prob        0.01     (0%)
```

Across all 180 tap-runs in the four r50 runs: median |λ − λ_mod| is **0.039**
where the noise fraction is under 5% (n=171) and **0.277** where it is over
(n=9) — a 7× separation, and all 9 are `features.0/1/2`.

The profile itself reproduces the r250 run: flat at λ ≈ 1 through the conv
stack, one ReLU at `classifier.4` taking it to +0.23, `prob` at **+0.06**.
Mean R² 0.976 at `prob` against 0.980 at 250 reps.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-caffe --notes 45 taps at 50 reps, converted Caffe. Companion to the r250 45-tap run: a tap whose D holds across the 5x rep change carries signal, one that falls by sqrt(5) is on the metric's noise floor. Also the first model run carrying both orderings of the metric. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all
```

Code: `564e392d056c`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
