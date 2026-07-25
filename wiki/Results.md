# Results

## VGG-19, full grid on the canonical checkpoint

The documented grid — 14 contrasts × 8 frequencies, **`--reps 250`** — on
torchvision's **`IMAGENET1K_V1`** (`vgg19-dcbb9e9d.pth`, downloaded on the
runner, not converted). Committed as
[`results/vgg19-r250-s0`](../results/vgg19-r250-s0/notes.md) and
[`results/vgg19-scramble-r250-s0`](../results/vgg19-scramble-r250-s0/notes.md),
so every number below re-fits from the repo:

```bash
python -m log_response.run --load results/vgg19-r250-s0 --panels out/panels.png
```

Measured twice, on different runner hardware at the same seed — runs
[30148332262](https://github.com/dekelron/RonInfra/actions/runs/30148332262) and
[30150601076](https://github.com/dekelron/RonInfra/actions/runs/30150601076) —
and the two agree to every digit shown here. The committed directories are from
the latter.

| Layer | Trained | Weights scrambled |
|---|---|---|
| `features.0` (conv1_1) | 0.548 | *0.604* |
| `features.19` (conv4_1) | 0.869 | *0.924* |
| **`classifier.3`** (fc7) | **0.928** | 0.760 |
| `logits` | 0.878 | 0.768 |
| `prob` (softmax) | 0.917 | 0.768 |

This is the run [Method](Method.md) asks for, and it disagrees with it on three
counts:

1. **`prob` reaches 0.917, not 0.98.**
2. **R² does not peak at `prob`.** It peaks one layer earlier, at `classifier.3`
   (fc7, 0.928), and *dips* at `logits` (0.878) before the softmax lifts it
   again. "Highest at `prob`" does not survive the full grid.
3. **The scrambled control beats the trained net at the early and middle taps**
   (0.604 vs 0.548, 0.924 vs 0.869; italicised above). The learned contribution
   is confined to the classifier end, where the ordering does hold — and even
   there the gap is 0.149, far short of the documented 0.98 − 0.60 = 0.38.

Read the scrambled column with its spacing CV: 4.07 / 3.54 / 3.54 at the three
late taps, against 0.59–0.89 for the trained net. A high R² there coexists with
grossly non-uniform spacing — the scrambled response is a spike at the top
contrast that a straight line happens to fit, not an even log ladder. R² alone
is the wrong summary for that column.

### Which variable moved it: the checkpoint

Against the `--reps 50` run below, two things changed at once — weight lineage
and repetition count. [`vgg19-r50-s0-in1k`](../results/vgg19-r50-s0-in1k/notes.md)
holds reps at 50 and changes only the checkpoint, which separates them:

| weights | reps | `prob` trained | `prob` scrambled |
|---|---|---|---|
| converted Caffe | 50 | 0.976 | 0.428 |
| `IMAGENET1K_V1` | 50 | 0.913 | 0.760 |
| `IMAGENET1K_V1` | 250 | 0.917 | 0.768 |

Changing the checkpoint at fixed reps moves `prob` by **0.063** and the control
by **0.332**. Changing reps 5× at a fixed checkpoint moves them by **0.004** and
**0.008**. **The weight lineage is the cause; repetition count is not.** The
whole r250 pattern — peak at `classifier.3`, dip at `logits`, control exceeding
trained at `features.19` — reproduces at 50 reps.

So the 0.976 that looked like it reproduced the documented 0.98 belongs to the
converted Oxford/Caffe checkpoint, not to torchvision's. Either the checkpoints
genuinely differ in this property, or the conversion carries an artifact that
argmax accuracy cannot see — it was validated at 89.9 % "Samoyed", which a gain
or channel-scaling error would survive while still shifting `D`. That is the
open question now.

## VGG-19, verified run

14 contrasts × 8 frequencies, `--reps 50`, 224×224, CPU. Mean R² of
`D = a·log10(c) + b`, fit per frequency and averaged over the 8.

| Layer | Trained | Weights scrambled |
|---|---|---|
| `features.0` (conv1_1) | 0.669 | 0.634 |
| `features.19` (conv4_1) | 0.718 | 0.669 |
| `classifier.3` (fc7) | 0.972 | 0.427 |
| `logits` | 0.941 | 0.428 |
| **`prob`** (softmax) | **0.976** | **0.428** |

Confirmed: R² climbs monotonically with depth and peaks at `prob` (0.976 against
the 0.98 claimed in [Method](Method.md)), so the log-like compression develops
late rather than being present at the first conv. The contrast-response family is
band-pass across spatial frequency at low contrast and converges toward
frequency-flat at high contrast. Scrambling collapses the effect — and the
scrambled response is also ~4000× smaller in absolute magnitude, degenerating
into a spike at the top contrast.

**Deviation from the documented expectation:** the scrambled control came out at
**0.428**, not the 0.60 in [Method](Method.md). The direction of the claim holds —
the effect is a property of training, not of architecture plus softmax — but the
residual is smaller than documented, which makes the learned contribution look
*larger* than stated. Unresolved; candidates are the scramble seed and the
reduced repetition count.

### What would firm this up

- ~~`--reps 250` (the documented grid)~~ — **done**, on the canonical checkpoint;
  see above. It did not confirm these numbers, it moved them.
- Repeat the scramble across seeds before treating *either* 0.428 or 0.768 as
  the real control value. Two runs disagreeing by 0.34 on the same control is
  itself the finding.
- Run `IMAGENET1K_V1` at `--reps 50` — the one cell that separates "weights
  lineage" from "repetition count" as the cause of the disagreement above.
- The weights were the original Caffe VGG-19 converted to torchvision layout (see
  [Running](Running.md#this-sandbox-weights-are-the-blocker)), same lineage as
  `IMAGENET1K_V1` but not verified bit-identical to it. Conversion was checked
  end-to-end: 89.9 % "Samoyed" on the standard PyTorch test image, all top-5 dog
  breeds.

## Synthetic front-end

Offline pipeline check, `--reps 12`. Not a model of any network — it only
verifies the measurement and fit code read the intended quantities.

| Stage | Mean R² |
|---|---|
| `energy` (pre-compression) | 0.495 |
| `output` (compressive) | 0.975 |

The pre-compression stage is *not* log-linear and the compressive stage is,
which is the intended contrast. Its frequency curves stay parallel rather than
converging at high contrast, so the frequency-flat high-contrast regime is a
property of the trained CNN and is not reproduced by this stand-in.
