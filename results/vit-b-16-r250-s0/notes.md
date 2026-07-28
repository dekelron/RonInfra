# vit-b-16-r250-s0

`vit_b_16`, 250 reps/cell, best mean R² 0.953 at `encoder.layers.encoder_layer_2.mlp.0`.

## What this run was for

ViT-B/16 depth profile, 65 taps (12 of 75 registered modules never fire -- MultiheadAttention out_proj, now warned about). GELU not ReLU, LayerNorm, attention softmax: no hard gates anywhere. Sharpest test of the gate-flip reading. Watch the high-frequency cells: 56 and 75 cyc/img are above the 14x14 patch grid's Nyquist.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model vit_b_16 --reps 250 --seed 0 --save-run results/vit-b-16-r250-s0 --notes ViT-B/16 depth profile, 65 taps (12 of 75 registered modules never fire -- MultiheadAttention out_proj, now warned about). GELU not ReLU, LayerNorm, attention softmax: no hard gates anywhere. Sharpest test of the gate-flip reading. Watch the high-frequency cells: 56 and 75 cyc/img are above the 14x14 patch grid's Nyquist. --figures out/ --layers all
```

Code: `7bdf878d43c4`. Weights: torchvision vit_b_16 IMAGENET1K_V1.
