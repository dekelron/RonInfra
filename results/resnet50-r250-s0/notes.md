# resnet50-r250-s0

`resnet50`, 250 reps/cell, best mean R² 0.957 at `layer2.3.relu@2`.

## What this run was for

ResNet-50 depth profile, 160 taps incl. 32 reuse slots. First run possible after the hook reuse fix. Prediction under test: the identity path keeps an affine component alive deep into the net, so lambda ~ 1 should persist AND those taps should fall as 1/sqrt(reps) -- the floor signature VGG-19's conv stack did not show.

## What it showed

**The peak tap in the whole network is one the old code threw away.** Best mean
R² over 160 taps is **0.957 at `layer2.3.relu@2`** — a reuse slot, i.e. the
*second* firing of a ReLU module that `self._acts[name]` used to overwrite. This
run is the first ResNet profile that could be trusted at all; see the hook fix
in commit `f9ab386`.

Across all 32 reuse slots, median |λ(base) − λ(@n)| = **0.167**, max **0.554**
(`layer2.1.relu@3`: +1.06 at the base against +0.51 at the third firing). So the
discarded activations were not near-duplicates of the survivors — the old
profile was reporting one of three genuinely different responses under a name
that read as the first.

**`prob` λ = −0.223** [−0.37, −0.08] at R² 0.968, mean R² 0.924 — past the log
law into saturating, like `vgg19_bn` (−0.268) and ViT-B/16 (−0.162) and unlike
either plain VGG-19 (+0.06 / +0.17) or AlexNet (+0.05). Rep-invariant: the r50
companion returns −0.223 to three decimals.

**The residual-stream prediction this run was launched to test is false.** The
hypothesis was that the identity path keeps an affine component alive deep into
the network, so λ ≈ 1 would persist and those taps would read the metric's
noise floor. Measured:

| | median λ | within 0.15 of λ = 1 |
|---|---|---|
| `layer1`/`layer2` (67 taps) | +0.729 | — |
| `layer3`/`layer4` (85 taps) | **−0.262** | **0%** |

Not one deep tap sits near λ = 1, and outside the first three modules the
largest noise fraction anywhere is **5.1%**
([`resnet50-r50-s0`](../resnet50-r50-s0/notes.md)). The profile declines
smoothly with depth instead. Skip connections do not hold the response linear.

**The floor is the affine prefix again**, and again it is two modules deep:
`conv1` (98.5% noise) and `bn1` (98.7%) — BatchNorm in eval is affine, so
conv1∘bn1 is still affine in the input and its population D is identically
zero. Exactly the `vgg19_bn` finding on a second architecture.

## Reproduce

```
run.py --model resnet50 --reps 250 --seed 0 --save-run results/resnet50-r250-s0 --notes ResNet-50 depth profile, 160 taps incl. 32 reuse slots. First run possible after the hook reuse fix. Prediction under test: the identity path keeps an affine component alive deep into the net, so lambda ~ 1 should persist AND those taps should fall as 1/sqrt(reps) -- the floor signature VGG-19's conv stack did not show. --figures out/ --layers all
```

Code: `7bdf878d43c4`. Weights: torchvision resnet50 IMAGENET1K_V1.
