# vit-b-16-r250-s0

`vit_b_16`, 250 reps/cell, best mean R² 0.953 at `encoder.layers.encoder_layer_2.mlp.0`.

## What this run was for

ViT-B/16 depth profile, 65 taps (12 of 75 registered modules never fire -- MultiheadAttention out_proj, now warned about). GELU not ReLU, LayerNorm, attention softmax: no hard gates anywhere. Sharpest test of the gate-flip reading. Watch the high-frequency cells: 56 and 75 cyc/img are above the 14x14 patch grid's Nyquist.

## What it showed

**The compression does not need a rectifier.** ViT-B/16 has **no ReLU
anywhere** — GELU (smooth, no hard gate), LayerNorm, attention softmax — and λ
goes not merely to the log law but well past it:

| tap | λ | λ R² | mean R² |
|---|---|---|---|
| `conv_proj` (patch embedding) | +0.926 | 0.987 | 0.757 |
| `encoder_layer_11.mlp.3` | **−0.617** | 0.917 | 0.697 |
| `encoder.ln` | −0.484 | 0.946 | 0.794 |
| `logits` (= `heads.head`) | −0.373 | 0.940 | 0.868 |
| **`prob`** | **−0.162** | 0.933 | **0.905** |

The gate-flip reading held that λ < 1 is the signature of ReLU gates switching
with contrast. There are no gates here to switch, and λ still travels from
+0.93 at the patch embedding to −0.62 mid-encoder. Together with
[`vgg19-bn-r250-s0`](../vgg19-bn-r250-s0/notes.md) — where BatchNorm moved the
conv stack by ~1.1 in λ while adding no rectifications — the surviving reading
is that what matters is the **operating point** relative to whatever
nonlinearity is present, not the count or the hardness of the rectifiers.

Rep-invariant: the [r50 companion](../vit-b-16-r50-s0/notes.md) gives −0.173 at
`prob`.

**`conv_proj` is on the floor and nothing else is.** λ +0.926 at R² 0.987
reproduces the model-free [`data-r250-s0`](../data-r250-s0/notes.md)'s +0.925
to three decimals, as the first layer of every architecture measured so far has.

**But the floor does *not* extend past it, and that is the informative part.**
The prediction when this run was launched was that `conv_proj` plus the first
LayerNorm would both read the floor, by analogy with `vgg19_bn`'s
conv1 + BatchNorm. Wrong, for a reason worth recording: **BatchNorm in eval is
affine** (fixed running statistics) while **LayerNorm is not** — it normalises
by the input's own mean and variance, so it is a nonlinear function of the
input and its population D is not zero. Only `conv_proj` is affine, so only
`conv_proj` reads the floor.

**Caveats.** `prob` mean R² 0.905 is the peak among the terminal taps but not
of the network — `encoder_layer_2.mlp.0` reaches 0.953, so ViT does not
reproduce the paper's "peak at `prob`" structure. And the two highest
frequencies (56 and 75 cyc/img) sit above the 14×14 patch grid's Nyquist; they
are included in the fits as on every other run, but a per-frequency read should
expect aliasing there.

## Reproduce

```
run.py --model vit_b_16 --reps 250 --seed 0 --save-run results/vit-b-16-r250-s0 --notes ViT-B/16 depth profile, 65 taps (12 of 75 registered modules never fire -- MultiheadAttention out_proj, now warned about). GELU not ReLU, LayerNorm, attention softmax: no hard gates anywhere. Sharpest test of the gate-flip reading. Watch the high-frequency cells: 56 and 75 cyc/img are above the 14x14 patch grid's Nyquist. --figures out/ --layers all
```

Code: `7bdf878d43c4`. Weights: torchvision vit_b_16 IMAGENET1K_V1.
