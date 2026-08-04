# vgg16-r50-s2

`vgg16`, 50 reps/cell, best mean R² 0.934 at `prob`.

## What this run was for

Seed replication for the VGG-16 lineage pair, torchvision half, seed 2 of 3.

## What it showed

Seed 2 of the VGG-16 three-seed sweep on torchvision `IMAGENET1K_V1`: conv-stack median λ **+0.676**, `prob` λ +0.107 (R² 0.958). With seeds 0 and 2 this gives VGG-16 **its own** noise floor instead of borrowing VGG-19's — conv-stack median sd is 0.014 (torchvision) and **0.003** (Caffe) across the three, so Caffe's λ = 1 conv stack is pinned to three decimals under independent image draws. The lineage effect is **22×** that floor: mean |Δλ| over 37 shared taps is 0.021 / 0.012 within lineage against **0.369** across, and the profile correlations do not overlap (0.992–0.999 vs 0.597–0.683). The plateau-then-cliff shape replicates too. See `wiki/Results.md`.

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 2 --save-run results/vgg16-r50-s2 --notes Seed replication for the VGG-16 lineage pair, torchvision half, seed 2 of 3. --figures out/ --layers all
```

Code: `26f9c7c7e74a`. Weights: torchvision vgg16 IMAGENET1K_V1.
