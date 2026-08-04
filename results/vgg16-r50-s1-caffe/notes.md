# vgg16-r50-s1-caffe

`vgg16`, 50 reps/cell, best mean R² 0.959 at `classifier.4`.

## What this run was for

Seed replication for the VGG-16 lineage pair, Caffe half, seed 1 of 3. The VGG-16 result (conv-stack median lambda +0.664 torchvision -> +1.025 Caffe, mean |dlambda| 0.353) currently rests on one seed and borrows its noise scale from the VGG-19 sweep; this gives VGG-16 its own.

## What it showed

Seed 1 of the VGG-16 three-seed sweep on Oxford/Caffe: conv-stack median λ **+1.023**, `prob` λ +0.120 (R² 0.970). With seeds 0 and 2 this gives VGG-16 **its own** noise floor instead of borrowing VGG-19's — conv-stack median sd is 0.014 (torchvision) and **0.003** (Caffe) across the three, so Caffe's λ = 1 conv stack is pinned to three decimals under independent image draws. The lineage effect is **22×** that floor: mean |Δλ| over 37 shared taps is 0.021 / 0.012 within lineage against **0.369** across, and the profile correlations do not overlap (0.992–0.999 vs 0.597–0.683). The plateau-then-cliff shape replicates too. See `wiki/Results.md`.

## Reproduce

```
run.py --model vgg16 --reps 50 --seed 1 --save-run results/vgg16-r50-s1-caffe --notes Seed replication for the VGG-16 lineage pair, Caffe half, seed 1 of 3. The VGG-16 result (conv-stack median lambda +0.664 torchvision -> +1.025 Caffe, mean |dlambda| 0.353) currently rests on one seed and borrows its noise scale from the VGG-19 sweep; this gives VGG-16 its own. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth --layers all
```

Code: `26f9c7c7e74a`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg16_caffe.pth.
