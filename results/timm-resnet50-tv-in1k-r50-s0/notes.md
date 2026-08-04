# timm-resnet50-tv-in1k-r50-s0

`timm:resnet50.tv_in1k`, 50 reps/cell, best mean R² 0.957 at `layer2.3.act2`.

## What this run was for

Cross-back-end control for the lineage series: .tv_in1k IS torchvision IMAGENET1K_V1, re-hosted by timm. If its lambda matches results/resnet50-r50-s0 then back-end and preprocessing are excluded and the rest of the series measures recipe alone. Size pinned to 224 so every tag in the series is sampled identically (a3 trains at 160).

## What it showed

**The control worked, exactly.** These are torchvision's `IMAGENET1K_V1` weights re-hosted by timm, and the measurement is identical to [`resnet50-r50-s0`](../resnet50-r50-s0/notes.md) to three decimals: `logits` λ +0.044, `prob` −0.223, **mean |Δλ| = 0.000 over all 110 shared taps**, profile correlation r = +1.000. Different library, different module naming, different preprocessing code, same numbers — so back-end and preprocessing are excluded as explanations for any difference between the other lineage tags, and the spread across them is the weights.

## Reproduce

```
run.py --model timm:resnet50.tv_in1k --reps 50 --seed 0 --save-run results/timm-resnet50-tv-in1k-r50-s0 --notes Cross-back-end control for the lineage series: .tv_in1k IS torchvision IMAGENET1K_V1, re-hosted by timm. If its lambda matches results/resnet50-r50-s0 then back-end and preprocessing are excluded and the rest of the series measures recipe alone. Size pinned to 224 so every tag in the series is sampled identically (a3 trains at 160). --figures out/ --layers all --size 224
```

Code: `a100e1e034c9`. Weights: timm 1.0.28; resnet50.tv_in1k; hf_hub_id=timm/resnet50.tv_in1k; tag=tv_in1k.
