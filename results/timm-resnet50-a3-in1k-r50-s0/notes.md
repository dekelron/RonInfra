# timm-resnet50-a3-in1k-r50-s0

`timm:resnet50.a3_in1k`, 50 reps/cell, best mean R² 0.945 at `layer2.1.act1`.

## What this run was for

ResNet Strikes Back procedure A3: 100 epochs, LAMB, BCE, 160px train crop, 78.1% top-1 -- same recipe family as A1 at a fifth the schedule. With A1 and .tv_in1k this makes the recipe axis graded rather than binary. Evaluated at 224, which is A3's own test resolution.

## What it showed

ResNet-50 trained by ResNet Strikes Back A3 (100 epochs, LAMB, BCE): `logits` λ **-0.261** (R² 0.934), `prob` **-0.381** (R² 0.948), against torchvision `IMAGENET1K_V1`'s +0.044 / −0.223 on the identical stimulus. One of five distinct lineages of the same 25.6 M parameters; λ spans 0.476 at `logits` across them, 11× the three-seed sampling sd, while the duplicate lineage (`.tv_in1k`) returns exactly 0.000. Every non-V1 lineage is more negative than V1 at both taps, and the ordering does not track top-1 accuracy. See `wiki/Results.md`.

## Reproduce

```
run.py --model timm:resnet50.a3_in1k --reps 50 --seed 0 --save-run results/timm-resnet50-a3-in1k-r50-s0 --notes ResNet Strikes Back procedure A3: 100 epochs, LAMB, BCE, 160px train crop, 78.1% top-1 -- same recipe family as A1 at a fifth the schedule. With A1 and .tv_in1k this makes the recipe axis graded rather than binary. Evaluated at 224, which is A3's own test resolution. --figures out/ --layers all --size 224
```

Code: `a100e1e034c9`. Weights: timm 1.0.28; resnet50.a3_in1k; hf_hub_id=timm/resnet50.a3_in1k; tag=a3_in1k.
