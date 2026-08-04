# timm-resnet50-gluon-in1k-r50-s0

`timm:resnet50.gluon_in1k`, 50 reps/cell, best mean R² 0.958 at `layer2.0.act3`.

## What this run was for

A third ResNet-50 lineage: the MXNet Gluon release (Bag-of-Tricks recipe -- cosine LR, label smoothing, mixup, 120 epochs), ported to timm. Trained in a different framework from both torchvision tags, so it tests whether recipe effects and framework effects separate on the same architecture.

## What it showed

ResNet-50 trained by the MXNet Gluon / Bag-of-Tricks release: `logits` λ **-0.393** (R² 0.941), `prob` **-0.313** (R² 0.946), against torchvision `IMAGENET1K_V1`'s +0.044 / −0.223 on the identical stimulus. One of five distinct lineages of the same 25.6 M parameters; λ spans 0.476 at `logits` across them, 11× the three-seed sampling sd, while the duplicate lineage (`.tv_in1k`) returns exactly 0.000. Every non-V1 lineage is more negative than V1 at both taps, and the ordering does not track top-1 accuracy. See `wiki/Results.md`.

## Reproduce

```
run.py --model timm:resnet50.gluon_in1k --reps 50 --seed 0 --save-run results/timm-resnet50-gluon-in1k-r50-s0 --notes A third ResNet-50 lineage: the MXNet Gluon release (Bag-of-Tricks recipe -- cosine LR, label smoothing, mixup, 120 epochs), ported to timm. Trained in a different framework from both torchvision tags, so it tests whether recipe effects and framework effects separate on the same architecture. --figures out/ --layers all --size 224
```

Code: `a100e1e034c9`. Weights: timm 1.0.28; resnet50.gluon_in1k; hf_hub_id=timm/resnet50.gluon_in1k; tag=gluon_in1k.
