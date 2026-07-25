# vgg19-r50-s0-in1k

`vgg19`, 50 reps/cell, best mean R² 0.921 at `classifier.3`.

## What this run was for

IMAGENET1K_V1 at 50 reps: the cell separating weight lineage from repetition count as the cause of the r50/r250 disagreement

## What it showed

**Weight lineage is the cause; repetition count is not.** Holding reps at 50 and
changing only the checkpoint moves `prob` by 0.063. Holding the checkpoint at
`IMAGENET1K_V1` and going 50 → 250 reps moves it by 0.004.

| weights | reps | `prob` trained | `prob` scrambled |
|---|---|---|---|
| converted Caffe ([`vgg19-r50-s0`](../vgg19-r50-s0/notes.md)) | 50 | 0.976 | 0.428 |
| `IMAGENET1K_V1` (this run) | 50 | **0.913** | **0.760** |
| `IMAGENET1K_V1` ([`vgg19-r250-s0`](../vgg19-r250-s0/notes.md)) | 250 | 0.917 | 0.768 |

| Layer | mean R² | pooled R² | spacing CV |
|---|---|---|---|
| `features.0` | 0.566 | 0.338 | 1.598 |
| `features.19` | 0.867 | 0.743 | 0.924 |
| **`classifier.3`** | **0.921** | 0.858 | 0.673 |
| `logits` | 0.877 | 0.815 | 0.971 |
| `prob` | 0.913 | 0.889 | 0.736 |

Every feature of the r250 result reappears here at a fifth of the sampling: the
peak at `classifier.3` rather than `prob`, the dip at `logits`, and the scrambled
control exceeding the trained net at `features.19` (0.924 vs 0.867). None of it
depends on repetition count.

**What this relocates.** The 0.976 that appeared to reproduce the documented 0.98
came from the converted Oxford/Caffe checkpoint, not from torchvision's. So
either the two checkpoints genuinely differ in log-contrast response, or the
conversion carries an artifact that classification accuracy does not expose — it
was validated at 89.9 % "Samoyed", which would not catch a gain or channel-scaling
error that leaves argmax intact while shifting `D`. That is now the open
question, and it is a sharper one than the reps/lineage confound it replaces.

One seed, one architecture. The claim here is about the two checkpoints, not
about VGG-19 in general.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-in1k --notes IMAGENET1K_V1 at 50 reps: the cell separating weight lineage from repetition count as the cause of the r50/r250 disagreement --figures out/
```

Code: `d446b36e387a`. Weights: torchvision vgg19 IMAGENET1K_V1.
