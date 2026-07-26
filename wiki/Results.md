# Results

> **`logness` was redefined on 2026-07-26 and every number below is on the new
> scale.** It is now
> `(RSS_lin − RSS_log)/(RSS_lin + RSS_log)`, which reaches −1 and +1 exactly and
> reads the same on any contrast grid. The old definition — plain
> `R²_log − R²_lin` — could not: a *perfect* log response scored only **+0.264**
> on the default grid while the docstring claimed +1, and that ceiling moved
> with the grid (0.294 when linearly spaced). Every pre-2026-07-26 figure quoted
> here was therefore ~3.8× smaller than its own stated scale implied. Nothing
> was re-run: `result.npz` holds the surfaces, so all 16 committed runs re-fit
> into the new statistic, and each `result.json` now carries both (`logness` and
> the superseded `logness_r2diff`). Which findings changed is stated below.

## Where the log response appears along depth

All 45 leaf modules, `--reps 250`, both checkpoints, trained and scrambled —
four runs measured on identical code. Read with `logness` (−1 linear in
contrast, +1 linear in log contrast, 0 neither law), not R²: R² is floored high
by any rising response and cannot tell the two laws apart.

**The checkpoints agree at the input.** `features.0` (conv1_1) is −0.863 on
`IMAGENET1K_V1` and −0.860 on the converted Caffe weights, a difference of
**0.003** on a scale where ±1 is saturation. This was the pre-registered test
for a conversion artifact — a preprocessing gain error would have shown up
here, at the only layer touching the input — and it is negative. Both sit hard
against the linear-in-contrast end, which is what a linear filter must do.

**Divergence grows with depth and mostly closes again.** It reaches **1.110** at
`features.35` — the two checkpoints at opposite ends of the scale — then falls
to **0.049** at `logits`. It does not close completely: `prob` still differs by
**0.364** (+0.466 canonical, +0.830 Caffe). Same input, different middle, nearly
the same output.

**One operation does the work, and it is a rectification.** On Caffe the
response stays pinned near −1 for the entire conv stack and crosses only at
`classifier.4`, the ReLU after fc7: −0.927 → **+0.702**, a jump of **1.629** at
a single ReLU — most of the available range, in one operation. `IMAGENET1K_V1`
does the same thing earlier and more gradually (`features.34` −0.679 →
`features.35` +0.148, and −0.211 → +0.477 at that same `classifier.4`). The
three-tap view looked like a smooth trend with depth; it is a sawtooth, and the
rectifications carry it.

**The two controls differ in kind, not degree.** Scrambled Caffe spans −0.987 to
−0.502 across all 45 taps and never crosses zero — it does not become log-like
anywhere. Scrambled `IMAGENET1K_V1` crosses at `features.16` (+0.216) and then
averages **+0.091** — positive, but close enough to zero to mean *neither law
describes it*, which is exactly what the new scale's zero is for. The
trained-minus-scrambled gap at `prob` is **+0.390** on `IMAGENET1K_V1` and
**+1.333** on Caffe.

**This is where the metric change mattered most.** Under the old definition the
scrambled control read +0.120 at `prob` against the trained net's +0.151 — a
near-tie that made the control look almost as log-like as the trained network,
and it exceeded the trained net at 32 of 45 taps. That was an artifact of
normalising by total variance: the scrambled response is poorly fit by *either*
law, and the old form did not notice. Dividing by the residual budget does, so
a badly-fit response is now pushed toward 0 rather than scoring high. The
qualitative findings — the input agreement, the crossing at `classifier.4`, the
sawtooth, the two controls behaving differently — all survived the change; the
apparent "the control is as log-like as the trained net" did not.

Regenerate the profile from the committed surfaces:

```bash
python -m log_response.run --load results/vgg19-r250-s0-alllayers-fixed --panels out/p.png
```

### The log-spaced grid is not what produces this — measured, not argued

The one methodological caveat under every `logness` number above: the default
contrast grid is log-spaced, which is **not neutral** between the two laws being
compared. It hands the log fit evenly spread leverage while bunching the linear
fit's points near zero, so a log-shaped verdict could in principle be an artifact
of where the axis was sampled.

The control is the same 45 taps at `--reps 250` with `--contrasts linear` —
identical endpoints, sampled evenly instead of geometrically, nothing else
changed. Committed as
[`vgg19-r250-s0-alllayers-linear`](../results/vgg19-r250-s0-alllayers-linear/notes.md)
and
[`vgg19-scramble-r250-s0-alllayers-linear`](../results/vgg19-scramble-r250-s0-alllayers-linear/notes.md).

**Every claim above survives.** Across the 45 layers, mean |Δ `logness`| is
**0.090** trained and **0.103** scrambled, against a profile that spans ~1.4
from conv stack to output. **Zero** sign flips in 45 on the trained net. `prob`
+0.466 → +0.540; `classifier.4`, the crossing ReLU, +0.477 → +0.594; the
scrambled control still crosses at `features.16` (+0.216 → +0.307). The sawtooth
is reproduced very nearly layer for layer: **43 of 44** consecutive steps move in
the same direction on both grids.

The shift is small, and it leans slightly toward "log" on the linear grid rather
than away from it. Note that the metric no longer *has* a grid-dependent
ceiling — that was the old definition's problem, and it is the reason this
comparison is now meaningful at all rather than a comparison of two rulers.

The scrambled control's step directions agree 40/44 and it shows 23 sign flips —
neither figure carries information. Its profile sits at +0.091 on average after
`features.16`, i.e. on top of zero, so its steps are noise about a constant and
its sign is a coin flip. The trained net's 43/44 with zero sign flips is the
meaningful figure.

> **Retraction (2026-07-26).** An earlier version of this section claimed the
> grid shift was "systematically negative through the conv stack", negative at
> *every one* of the 37 conv taps. That was an artifact of comparing raw
> pre-2026-07-26 `logness` across two grids with different ceilings — a
> comparison the old statistic did not support. On the current metric the
> conv-stack shift averages −0.092 but ranges −0.178 to **+0.107**, so it is
> **not** uniformly negative. The headline — that the depth profile survives the
> change of contrast grid — is unaffected and now rests on a statistic that is
> actually comparable across grids.

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

### The scrambled control is not a single number

Four permutations at identical settings (`IMAGENET1K_V1`, 50 reps, seeds 0–3)
give `prob` mean R² of **0.760, 0.863, 0.704, 0.693** — spread 0.169, sd 0.078.

The consequence is not that the control is noisy, it is that **a one-seed
control cannot measure what it is for**. The trained net scores 0.913 at these
settings, so the learned contribution comes out anywhere from **0.050 to 0.220**
depending only on which permutation was drawn. The four values straddle the 0.60
in [Method](Method.md) rather than confirming or refuting it, and the earlier
single values — 0.428 on the Caffe checkpoint, 0.768 at 250 reps — should be
read as draws from that distribution, not as measurements of a constant.

The spacing CV moves 1.24–4.15 and not in step with R²: seed 1 has the highest
R² *and* a low CV (genuinely straighter), seed 3 the lowest R² and also a low
CV. Different permutations differ in kind, not just in estimate.

Caveat: at the code version these ran on, `--seed` drove both the permutation
and the orientation/phase draws. Sampling is unlikely to explain much — at a
fixed seed, 50 → 250 reps moved this value 0.760 → 0.768 — but the sweep did
vary both. `--scramble-seed` now separates them for the next one.

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
converted Oxford/Caffe checkpoint, not to torchvision's. That left two
explanations: the checkpoints genuinely differ in this property, or the
conversion carries an artifact.

**The artifact explanation is now ruled out**, at least for the input-side gain
that was the specific worry. The suspect step was folding caffe preprocessing
into conv1, because a gain error there rescales a grating's effective contrast
and slides the contrast-response curve along its own axis — exactly mimicking a
checkpoint difference, and exactly what argmax accuracy cannot see.
`python -m log_response.convert_weights --verify` compares the folded conv1
against a directly-computed caffe path: **relative error 2.9e-8, best-fit gain
1.000000001**. The fold is an exact linear identity, and conv1 is the only layer
that touches the input, so no input-gain or channel-scaling error survives
anywhere in the converted net. `test_preprocessing_fold_is_exact` pins the
arithmetic offline.

They do genuinely differ, and the depth profile above says where: identical at
conv1_1, diverging through the conv stack, re-converging at the classifier. A
real difference between two legitimately different checkpoints.

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
  breeds — and, more to the point, the preprocessing fold is verified exact to
  2.9e-8 (see above), which argmax accuracy alone could not establish.
  Regenerate the checkpoint with
  `python -m log_response.convert_weights --out W.pth --verify`.

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
