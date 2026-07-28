# vgg19-bn-scramble-r250-s0

`vgg19_bn`, 250 reps/cell, best mean R² 0.814 at `features.45`.

## What this run was for

VGG-19 + BatchNorm depth profile, 61 taps. Identical topology and ReLU count to vgg19; BN is affine in eval so it adds no gates, only moves the operating point. Tests whether lambda through the conv stack is set by the operating point rather than the rectifier count.

## What it showed

**This control is not comparable to the VGG-19 ones, and the reason is
methodological.** Reported here rather than quoted as a control value.

`--scramble` permutes the elements of every parameter named `*weight*`. On a
plain net that permutes conv filters. On a **BatchNorm** net it also permutes γ
across channels while leaving `running_mean`/`running_var` in place, so each
channel is normalised by one channel's statistics and rescaled by another's.
That does not degrade the network the way filter scrambling does — it
decalibrates it, and the activations blow up.

The signature is unmistakable:

| | scrambled VGG-19 (either checkpoint) | scrambled AlexNet | **scrambled `vgg19_bn`** |
|---|---|---|---|
| r(logits, prob) | 1.000000 | 1.000000 | **0.162** |
| median D_prob/D_logits | 1/1000 | 1/1000 | **1.1e-10** |

Every other scrambled run in this repo sits in the softmax's affine regime,
where Δprob = Δlogits/1000 exactly. Here the logits are so large that the
softmax is saturated to one-hot and `prob` barely moves at all — ten orders of
magnitude below the logit response.

Consequences for reading it:

- `prob` mean R² **0.214** and λ −2.794 at λ-R² **0.613**. The power family does
  not describe this, so **the λ is uninformative** — per the repo's rule, λ is
  read against its R² or not at all.
- The conv taps do return a clean λ ≈ **+1.08** at R² 0.994–0.998, i.e. flatly
  linear in contrast, which is the usual scrambled-conv reading.
- **Do not put 0.214 in a trained-minus-scrambled table** next to VGG-19's 0.429
  or AlexNet's 0.865. It is measuring a differently-broken network.

The honest conclusion is about the tool, not the net: **the within-layer
scrambling control is not architecture-neutral.** A control for a normalised
architecture needs to leave the normalisation statistics consistent with the
weights it scrambles — or scramble the running statistics with them.

## Reproduce

```
run.py --model vgg19_bn --reps 250 --seed 0 --save-run results/vgg19-bn-scramble-r250-s0 --notes VGG-19 + BatchNorm depth profile, 61 taps. Identical topology and ReLU count to vgg19; BN is affine in eval so it adds no gates, only moves the operating point. Tests whether lambda through the conv stack is set by the operating point rather than the rectifier count. --figures out/ --layers all --scramble
```

Code: `f9ab3861976a`. Weights: torchvision vgg19_bn IMAGENET1K_V1.
