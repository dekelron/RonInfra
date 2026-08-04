# timm-resnet50-gluon-in1k-r50-s0

`timm:resnet50.gluon_in1k`, 50 reps/cell, best mean R² 0.958 at `layer2.0.act3`.

## What this run was for

A third ResNet-50 lineage: the MXNet Gluon release (Bag-of-Tricks recipe -- cosine LR, label smoothing, mixup, 120 epochs), ported to timm. Trained in a different framework from both torchvision tags, so it tests whether recipe effects and framework effects separate on the same architecture.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model timm:resnet50.gluon_in1k --reps 50 --seed 0 --save-run results/timm-resnet50-gluon-in1k-r50-s0 --notes A third ResNet-50 lineage: the MXNet Gluon release (Bag-of-Tricks recipe -- cosine LR, label smoothing, mixup, 120 epochs), ported to timm. Trained in a different framework from both torchvision tags, so it tests whether recipe effects and framework effects separate on the same architecture. --figures out/ --layers all --size 224
```

Code: `a100e1e034c9`. Weights: timm 1.0.28; resnet50.gluon_in1k; hf_hub_id=timm/resnet50.gluon_in1k; tag=gluon_in1k.
