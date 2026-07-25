# vgg19-r50-s0

`vgg19`, 50 reps/cell, best mean R² 0.976 at `prob`.

## What this run was for

Primary measurement: does the log-contrast response reproduce on a trained
ImageNet CNN in this environment?

> **Superseded in part.** The full `--reps 250` grid on torchvision's canonical
> `IMAGENET1K_V1` reports `prob` at 0.917 with the peak at `classifier.3`, not
> at `prob` — see [Results](../../wiki/Results.md). Two variables differ between
> that run and this one (weight lineage and repetition count), so neither
> supersedes the other cleanly. Read the "peaks at `prob`" claim below as
> specific to this run, not established.

## What it showed

Yes. Mean R² of `D` vs `log10(c)`, per layer:

| Layer | Mean R² | Pooled R² | Spacing CV |
|---|---|---|---|
| `features.0` (conv1_1) | 0.669 | 0.488 | 1.297 |
| `features.19` (conv4_1) | 0.718 | 0.620 | 1.186 |
| `classifier.3` (fc7) | 0.972 | 0.919 | 0.406 |
| `logits` | 0.941 | 0.817 | 0.591 |
| `prob` | **0.976** | 0.885 | 0.422 |

Three things line up with `wiki/Method.md`:

- `prob` reaches 0.976 against the documented 0.98.
- R² climbs monotonically with depth, so the log-like compression develops late
  rather than being a property of the first conv.
- The contrast-response family is band-pass across spatial frequency at low
  contrast and converges toward frequency-flat at high contrast.

`logits` (0.941) sits below `prob` (0.976), so the softmax contributes to the
compression but does not create it — `fc7` is already at 0.972 pre-softmax.

## Caveats

50 reps/cell, not the documented 250, so these carry more sampling noise than
the headline figures. One seed. Weights are the original Caffe VGG-19 converted
to torchvision layout (see `run.json` for the digest and the conversion), same
lineage as `IMAGENET1K_V1` but not verified bit-identical.

Pairs with `../vgg19-scramble-r50-s0`, the within-layer scrambling control.

## Reproduce

```
python -m log_response.run --model vgg19 --weights <converted> --reps 50 --save runs/vgg19
```

Code: `4c204e5f9b89`. Weights: Oxford VGG-19 ImageNet weights, converted to
torchvision layout. Re-fit and re-plot without a model or weights:

```
python -m log_response.run --load results/vgg19-r50-s0 --figures out/
```
