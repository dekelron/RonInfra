# resnet50-scramble-r250-s0

`resnet50`, 250 reps/cell, best mean R² 0.756 at `bn1`.

## What this run was for

ResNet-50 depth profile, 160 taps incl. 32 reuse slots. First run possible after the hook reuse fix. Prediction under test: the identity path keeps an affine component alive deep into the net, so lambda ~ 1 should persist AND those taps should fall as 1/sqrt(reps) -- the floor signature VGG-19's conv stack did not show.

## What it showed

**Not a usable control — same BatchNorm failure as `vgg19_bn`, as predicted.**

ResNet-50 carries BatchNorm, so `--scramble` permutes γ across channels while
`running_mean`/`running_var` stay put, decalibrating the network rather than
degrading it. The signature is the one documented in
[`vgg19-bn-scramble-r250-s0`](../vgg19-bn-scramble-r250-s0/notes.md):

| | r(logits, prob) | median D_prob/D_logits |
|---|---|---|
| scrambled VGG-19 / AlexNet / ViT-B/16 | ≈ 1.000000 | 1/1000 |
| scrambled `vgg19_bn` | 0.162 | 1.1e-10 |
| **scrambled `resnet50`** | **0.673** | **1.7e-10** |

Ten orders of magnitude below the affine-regime ratio, i.e. the softmax is
saturated to one-hot. `prob` λ −0.122 with a CI of [−0.50, +1.10] at λ-R²
**0.692**, and the [r50 companion](../resnet50-scramble-r50-s0/notes.md) opens
to the entire search range. **The λ is uninformative and mean R² 0.658 is not
comparable** to VGG-19's 0.429 or AlexNet's 0.865.

Two architectures now, both with running statistics, both broken the same way —
and ViT-B/16, whose LayerNorm has no running statistics, scrambles cleanly
(r = 0.999975, ratio 9.96e-04). That is as sharp a localisation of the cause as
the available runs can give: it is the *running statistics*, not normalisation
as such.

## Reproduce

```
run.py --model resnet50 --reps 250 --seed 0 --save-run results/resnet50-scramble-r250-s0 --notes ResNet-50 depth profile, 160 taps incl. 32 reuse slots. First run possible after the hook reuse fix. Prediction under test: the identity path keeps an affine component alive deep into the net, so lambda ~ 1 should persist AND those taps should fall as 1/sqrt(reps) -- the floor signature VGG-19's conv stack did not show. --figures out/ --layers all --scramble
```

Code: `7bdf878d43c4`. Weights: torchvision resnet50 IMAGENET1K_V1.
