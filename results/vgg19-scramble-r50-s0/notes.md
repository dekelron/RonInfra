# vgg19-scramble-r50-s0

`vgg19`, 50 reps/cell, weights permuted within each layer (seed 0).

## What this run was for

Control: within-layer weight scrambling, to separate the trained contribution
from architecture + softmax + metric.

## What it showed

The effect collapses, confirming it is a property of *training* rather than of
architecture. Against the paired trained run:

| Layer | Trained | Scrambled |
|---|---|---|
| `features.0` | 0.669 | 0.634 |
| `features.19` | 0.718 | 0.669 |
| `classifier.3` | 0.972 | 0.427 |
| `logits` | 0.941 | 0.428 |
| `prob` | 0.976 | **0.428** |

The early conv layers barely move (0.669→0.634), which makes sense: at conv1
the response is dominated by the filter bank's raw contrast sensitivity, and
permuting weights preserves the marginal distribution. Everything from `fc7`
onward collapses — that is where the trained structure lives.

The scrambled response is also **~4000× smaller in absolute magnitude** and
degenerates into a spike at the top contrast rather than a graded curve, so the
surviving 0.428 is fitting a nearly-flat surface, not a weaker log law.

> **A third value now exists.** The full `--reps 250` grid on `IMAGENET1K_V1`
> puts this control at 0.768 — see [Results](../../wiki/Results.md). So the
> control has been measured at 0.428 (here), 0.768 (full grid) and 0.60
> (documented), and the full grid also has the scrambled net *beating* the
> trained one at the early and middle taps. Three disagreeing values on the same
> control is the finding; do not quote any of them as settled.

## Disagreement with the documented expectation

`wiki/Method.md` documents this control at **0.60**; it measured **0.428**. The
direction of the claim holds, but the residual is smaller than documented, which
makes the learned contribution look *larger* than stated. Unresolved. Candidates,
in the order worth testing:

1. **Seed.** One scramble seed (0) was run. The permutation is a single draw and
   nothing pins its variance.
2. **Repetition count.** 50 reps/cell rather than 250; the scrambled surface is
   near-flat, so its fit is the more noise-sensitive of the two.
3. **Weight provenance.** Converted Caffe weights rather than `IMAGENET1K_V1`
   itself — this should not matter for a scrambled net, but it is not excluded.

Do not treat 0.428 as the control value until at least (1) is settled.

## Reproduce

```
python -m log_response.run --model vgg19 --weights <converted> --reps 50 --scramble --save runs/vgg19_scr
```

Code: `4c204e5f9b89`. Weights: Oxford VGG-19 ImageNet weights, converted to
torchvision layout.
