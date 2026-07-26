# vgg19-scramble-r250-s0-alllayers-linear

`vgg19`, 250 reps/cell, best mean R² 0.921 at `features.20`.

## What this run was for

The control for the one methodological caveat that touched every `logness`
number: the default contrast grid is log-spaced, which is **not neutral**
between the two laws `logness` compares — it gives the log fit evenly spread
leverage while bunching the linear fit's points near zero. This run samples the
same endpoints evenly instead, so it differs from
[`vgg19-scramble-r250-s0-alllayers-fixed`](../vgg19-scramble-r250-s0-alllayers-fixed/notes.md)
in nothing but where the contrast axis is sampled.

## What it showed

> Numbers below are on the `logness` definition adopted 2026-07-26. This run's
> `result.json` carries both it and the superseded `logness_r2diff`.

**The picture holds where it can.** Mean |Δ logness| across the 45 layers is
**0.103**, and `prob` moves +0.076 → +0.066. The finding this run exists to
protect — that scrambled `IMAGENET1K_V1` crosses zero at `features.16` and then
stays weakly positive, unlike scrambled Caffe which never crosses — is
unchanged: the crossing is still at `features.16` (+0.216 → +0.307).

**Most per-layer statistics on this run carry no information, by construction.**
After `features.16` the profile averages **+0.091**, i.e. it sits on top of zero.
So its 23 sign flips and its 40/44 step agreement are what a coin flip about zero
looks like, not a failure to reproduce — there is no shape here to reproduce.
That +0.091 is itself the finding: on the current metric, zero means *neither*
law describes the response, and a scrambled net lands there. The trained run's
43/44 with zero sign flips is where the grid comparison carries information.

> Under the retired `logness_r2diff` this profile read ≈ +0.12 and was described
> as "flat near +0.12, reaching the same place as its trained counterpart". That
> reading was an artifact of normalising by total variance, which could not
> distinguish "follows the log law" from "follows neither law". See the
> retraction note in `wiki/Results.md`.

## Reproduce

```
run.py --model vgg19 --reps 250 --seed 0 --save-run results/vgg19-scramble-r250-s0-alllayers-linear --notes Same grid endpoints sampled evenly instead of geometrically. Tests whether the logness depth profile is a property of VGG-19 or of the log-spaced contrast grid. --figures out/ --contrasts linear --layers all --scramble
```

Code: `aae17b04d27f`. Weights: torchvision vgg19 IMAGENET1K_V1.
