# timm-resnet50-a1-in1k-r50-s0

`timm:resnet50.a1_in1k`, 50 reps/cell, best mean R² 0.977 at `layer2.1.act3`.

## What this run was for

ResNet Strikes Back procedure A1: 600 epochs, LAMB, binary cross-entropy, RandAugment + mixup 0.2 + cutmix 1.0 + repeated augmentation, 80.4% top-1. Same 25.6 M parameters as resnet50.tv_in1k (90 epochs, SGD, cross-entropy, flip+RRC, 76.1%). The far end of the ResNet-50 recipe series.

## What it showed

ResNet-50 trained by ResNet Strikes Back A1 (600 epochs, LAMB, BCE): `logits` λ **-0.166** (R² 0.900), `prob` **-0.347** (R² 0.894), against torchvision `IMAGENET1K_V1`'s +0.044 / −0.223 on the identical stimulus. One of five distinct lineages of the same 25.6 M parameters; λ spans 0.476 at `logits` across them, 11× the three-seed sampling sd, while the duplicate lineage (`.tv_in1k`) returns exactly 0.000. Every non-V1 lineage is more negative than V1 at both taps, and the ordering does not track top-1 accuracy. See `wiki/Results.md`.

## Reproduce

```
run.py --model timm:resnet50.a1_in1k --reps 50 --seed 0 --save-run results/timm-resnet50-a1-in1k-r50-s0 --notes ResNet Strikes Back procedure A1: 600 epochs, LAMB, binary cross-entropy, RandAugment + mixup 0.2 + cutmix 1.0 + repeated augmentation, 80.4% top-1. Same 25.6 M parameters as resnet50.tv_in1k (90 epochs, SGD, cross-entropy, flip+RRC, 76.1%). The far end of the ResNet-50 recipe series. --figures out/ --layers all --size 224
```

Code: `a100e1e034c9`. Weights: timm 1.0.28; resnet50.a1_in1k; hf_hub_id=timm/resnet50.a1_in1k; tag=a1_in1k.
