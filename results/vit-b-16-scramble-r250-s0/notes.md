# vit-b-16-scramble-r250-s0

`vit_b_16`, 250 reps/cell, best mean R² 0.842 at `encoder.layers.encoder_layer_0.ln_1`.

## What this run was for

ViT-B/16 depth profile, 65 taps (12 of 75 registered modules never fire -- MultiheadAttention out_proj, now warned about). GELU not ReLU, LayerNorm, attention softmax: no hard gates anywhere. Sharpest test of the gate-flip reading. Watch the high-frequency cells: 56 and 75 cyc/img are above the 14x14 patch grid's Nyquist.

## What it showed

**A clean control, and the one that localises the BatchNorm problem.**

Unlike scrambled `vgg19_bn` and `resnet50`, this run behaves exactly as every
plain-network control does: r(logits, prob) = **0.999975** with median ratio
**9.96e-04** ≈ 1/1000, i.e. squarely in the softmax's affine regime. ViT's
LayerNorm has **no running statistics**, so permuting the weight tensors cannot
leave normalisation constants inconsistent with the weights they normalise.

Two architectures with running statistics break; two without them, plus this
one, do not. That is the sharpest available evidence that the scrambling
control's failure is specifically about *running statistics*, not about
normalisation in general.

The measurement itself:

| | trained | scrambled |
|---|---|---|
| `prob` λ | −0.162 | **+0.711** |
| `prob` λ R² | 0.933 | 0.981 |
| `prob` mean R² | **0.905** | 0.797 |
| peak tap | `encoder_layer_2.mlp.0` (0.953) | `encoder_layer_0.ln_1` (0.842) |

Note the direction: scrambling moves λ from saturating **up toward linear in
contrast** (+0.711 at a high R² 0.981), the same way it did on the Caffe VGG-19
checkpoint (+1.76 to +3.00) and the opposite of `IMAGENET1K_V1`. The peak also
moves to the first block. λ here is well determined — this is one of the
scrambled runs where the power family genuinely fits, so the value can be read.

## Reproduce

```
run.py --model vit_b_16 --reps 250 --seed 0 --save-run results/vit-b-16-scramble-r250-s0 --notes ViT-B/16 depth profile, 65 taps (12 of 75 registered modules never fire -- MultiheadAttention out_proj, now warned about). GELU not ReLU, LayerNorm, attention softmax: no hard gates anywhere. Sharpest test of the gate-flip reading. Watch the high-frequency cells: 56 and 75 cyc/img are above the 14x14 patch grid's Nyquist. --figures out/ --layers all --scramble
```

Code: `7bdf878d43c4`. Weights: torchvision vit_b_16 IMAGENET1K_V1.
