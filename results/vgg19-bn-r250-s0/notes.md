# vgg19-bn-r250-s0

`vgg19_bn`, 250 reps/cell, best mean R² 0.931 at `features.26`.

## What this run was for

VGG-19 + BatchNorm depth profile, 61 taps. Identical topology and ReLU count to vgg19; BN is affine in eval so it adds no gates, only moves the operating point. Tests whether lambda through the conv stack is set by the operating point rather than the rectifier count.

## What it showed

**BatchNorm moves the entire conv stack to the log law, without adding a single
rectification.** This is the run's finding and it bears directly on the live
hypothesis in [CLAUDE.md](../../CLAUDE.md).

| conv-stack median λ | R² |
|---|---|
| converted Caffe VGG-19 | **+1.06** (flatly linear in contrast) | 0.999 |
| `IMAGENET1K_V1` VGG-19 | +0.69 | — |
| **`vgg19_bn`, this run** | **−0.071** | 0.971 |

Same topology, same ReLU count, same task, same 224² stimulus. BatchNorm in
eval mode is a **per-channel affine map** — it cannot add gates. Yet the conv
stack goes from linear-in-contrast to *at the log law* (λ = 0 within noise)
across 41 taps.

So "the crossover to log is carried by rectifications" is not sufficient as
stated. What BN changes is the **operating point** each unit sits at relative
to its ReLU, and that alone accounts for a shift of ~1.1 in λ — larger than the
entire depth profile of the Caffe checkpoint. The perturbation/gate-flip
reading survives, but the controlling variable is where the units sit, not how
many rectifiers they pass through.

**At the output, λ goes past log into saturating:** `prob` λ = **−0.268**
[−0.53, −0.14] at R² 0.961, against Caffe's +0.059 and IN1K's +0.165.
`classifier.4` reaches **−0.436** and the softmax pushes it back *toward* zero,
the same direction it acts on IN1K.

**No sawtooth.** By transition type, trained, past the floor:

| transition | mean Δλ | negative |
|---|---|---|
| `Conv2d → BatchNorm2d` | +0.007 | 9/16 |
| `BatchNorm2d → ReLU` | +0.071 | 11/16 |
| `ReLU → Conv2d` | −0.153 | 4/11 |

Third architecture, third pattern. The `IMAGENET1K_V1` sawtooth is a property
of that checkpoint and should not be quoted as a property of VGG-19.

**Caveats.**

- **The floor extends deeper here, and it is BatchNorm's doing.** Five taps sit
  on or near it — `features.0` (conv, 98.6% noise), **`features.1` (BN, 99.3%)**,
  and `features.2/3/4` at 74–77%. BN in eval is affine, so `features.1` composed
  with `features.0` is still affine in the input and its population D is
  identically zero. Everything from `features.12` on is under 13% noise and the
  headline taps are under 1% — see
  [`vgg19-bn-r50-s0`](../vgg19-bn-r50-s0/notes.md).
- `prob` is at **95.4%** of its 2/1000 ceiling. Prefer `classifier.4`.
- `prob` mean R² is **0.850** and the peak is `features.26` (0.931), so this
  checkpoint does *not* reproduce the paper's "peak at `prob`" structure.

## Reproduce

```
run.py --model vgg19_bn --reps 250 --seed 0 --save-run results/vgg19-bn-r250-s0 --notes VGG-19 + BatchNorm depth profile, 61 taps. Identical topology and ReLU count to vgg19; BN is affine in eval so it adds no gates, only moves the operating point. Tests whether lambda through the conv stack is set by the operating point rather than the rectifier count. --figures out/ --layers all
```

Code: `f9ab3861976a`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
