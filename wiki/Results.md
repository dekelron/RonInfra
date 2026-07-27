# Results

> **`logness` was removed on 2026-07-26. Every number below is λ.** The old
> statistic raced two straight lines — `D = a + b·log c` against `D = a + b·c` —
> and reported which lost less. That only means something if one of them is
> right, and neither is: the trained net is **convex in `log c` at 95%** of
> layer-frequency cells and the scrambled control is **not even monotone at
> 41%** of them. Because the race summed *squared* residuals, a single contrast
> point out of 14 carried **20–55%** of the verdict.
>
> λ is the exponent of the nested family `D = a + b·(c^λ − 1)/λ`
> (Box-Tidwell/Box-Cox): **λ = 0 is the log law, λ = 1 linear in contrast**, 0.5
> a square root, negative saturating. One measured parameter with a profile-F
> interval, instead of a verdict between two guesses. It fits **0.92–0.998**
> everywhere the straight lines did not, it is grid-free, and on pure noise it
> returns the *entire* search range rather than a confident zero.
>
> Nothing was re-run: `result.npz` holds the surfaces, so all 16 committed runs
> re-fit, and `result.json` now carries `lambda`, `lambda_ci`, `lambda_r2`.
> Which findings changed is stated below.

## λ at a glance

Every 45-tap run, re-fitted from its committed surfaces. λ = 0 is the log law,
λ = 1 linear in contrast. Interval on `prob` is the 95% profile-F.

| Run | Weights | `features.0` | conv median | `classifier.4` | `prob` | mean R² |
|---|---|---|---|---|---|---|
| [`vgg19-r250-s0-alllayers-fixed-caffe`](../results/vgg19-r250-s0-alllayers-fixed-caffe/notes.md) | trained · Caffe | +0.922 | +1.06 | +0.211 | **+0.059** [−0.05, +0.13] | 0.998 |
| [`vgg19-r250-s0-alllayers-fixed`](../results/vgg19-r250-s0-alllayers-fixed/notes.md) | trained · IN1K | +0.923 | +0.69 | +0.010 | **+0.165** [−0.09, +0.29] | 0.978 |
| [`vgg19-r250-s0-alllayers-linear`](../results/vgg19-r250-s0-alllayers-linear/notes.md) | trained · IN1K, linear grid | +0.934 | +0.75 | +0.086 | **+0.180** [+0.06, +0.29] | 0.984 |
| [`vgg19-scramble-r250-s0-alllayers-fixed-caffe`](../results/vgg19-scramble-r250-s0-alllayers-fixed-caffe/notes.md) | scrambled · Caffe | +0.926 | +1.29 | +2.748 | **+2.743** [+2.20, +3.54] | 0.989 |
| [`vgg19-scramble-r250-s0-alllayers-fixed`](../results/vgg19-scramble-r250-s0-alllayers-fixed/notes.md) | scrambled · IN1K | +0.926 | +0.35 | +0.163 | **+0.169** [−0.18, +0.63] | 0.918 |
| [`vgg19-scramble-r250-s0-alllayers-linear`](../results/vgg19-scramble-r250-s0-alllayers-linear/notes.md) | scrambled · IN1K, linear grid | +0.934 | +0.35 | +0.123 | **+0.128** [−1.47, +0.77] | 0.869 |

Four things to read off it before the prose:

- **`features.0` is +0.92–0.93 in all six — because λ cannot see that layer.**
  A model-free run on raw pixels returns the same +0.925 at the same 0.985 fit
  quality. That *column* is the metric's noise floor. The layer is not empty —
  its frequency profile is a real measurement — but its contrast axis is
  uninformative.
  [Why](#the-metric-has-a-noise-floor-and-features0-is-sitting-on-it).
- **The conv median separates the checkpoints** (+1.06 Caffe, +0.69 IN1K) far
  more than `prob` does.
- **λ at `prob` does not separate trained from scrambled on IN1K** — +0.165
  against +0.169. The R² column does (0.978 against 0.918), and the intervals do
  ([−0.09, +0.29] against [−0.18, +0.63]).
- **The linear-grid rows track the log-grid rows**, which is the grid control.

Regenerate the whole table with `--load <dir>`; nothing here needs a model.

## Where the log response appears along depth

All 45 leaf modules, `--reps 250`, both checkpoints, trained and scrambled —
four runs measured on identical code. **Always read λ against `lambda_r2`**: the
exponent locates a response only insofar as the family describes it.

**The checkpoints agree at the input because λ cannot see that layer.**
`features.0` (conv1_1) is **λ = 0.922** on `IMAGENET1K_V1` and **0.923** on the
converted Caffe weights; the two scrambled runs give 0.926 and 0.926. All four
agree to **0.01** — and so does a run with no network in it at all. That
agreement is forced by the metric, not a property of the checkpoints; see
[below](#the-metric-has-a-noise-floor-and-features0-is-sitting-on-it). The layer
itself is *not* empty — the same surface separates trained from scrambled
cleanly along its **other** axis.

It does still rule out one thing, which is what it was originally introduced
for. A preprocessing **gain** error in the Caffe conversion would rescale a
grating's effective contrast and slide the response along its own axis; the
noise floor's magnitude is `c · mean|W·ḡ|`, so a gain error would move it. It
does not move. That, plus `convert_weights.py --verify` (relative error 2.9e-8),
closes the conversion-artifact question.

**Divergence grows with depth and mostly closes again.** It reaches **0.622** at
`features.35` (Caffe +0.96, canonical +0.33), then falls to **0.041** at
`logits` and **0.106** at `prob`. Same input, different middle, nearly the same
output.

**Caffe's conv stack is not "less log-like" — it is flatly linear.** λ holds at
a median of **+1.06** across the 37 conv taps at R² 0.999. `IMAGENET1K_V1` sits
at **+0.69**, genuinely partway. This is a qualitative difference between the
checkpoints that the two-model race could only express as a position on an
arbitrary scale.

**One operation does the work, and it is a rectification.** On Caffe, one ReLU
takes `classifier.3` **+1.104 → classifier.4 +0.211**, and `prob` lands at
**λ = 0.059, CI [−0.05, +0.13], R² 0.992** — the log law, measured, with an
interval that contains 0. `IMAGENET1K_V1` does the same more gradually (+0.623 →
+0.010 at that ReLU, `prob` +0.165).

**But the per-layer sawtooth is one checkpoint's, not VGG-19's.** Averaged over
the `features.*` stack of the trained runs:

| | conv → ReLU | ReLU → conv |
|---|---|---|
| `IMAGENET1K_V1` | mean **−0.155**, 14/16 negative | mean **+0.166**, 10/11 positive |
| converted Caffe | mean +0.023, 5/16 negative | mean −0.015, 7/11 negative |

`IMAGENET1K_V1` shows the alternation cleanly; Caffe's steps are noise about
zero and its conv stack is simply flat at λ ≈ 1. Per rule 4 both are stated
rather than averaged together.

> **Corrected 2026-07-26.** An earlier version of this section asserted the
> sawtooth as a general mechanism — "convolutions push λ toward linear and ReLUs
> push it back". That was carried over from the retired metric's write-up
> without re-checking it per checkpoint, and it is false for Caffe. The
> `classifier.4` crossing, which is what the surrounding claims rest on, is
> unaffected: it is large on both checkpoints.

**Why λ ≈ 1 can survive 33 layers.** The grating is a *perturbation*
`gray + c·g` about a fixed operating point, and a ReLU network is piecewise
linear — so as long as the perturbation does not flip ReLU gates,
`D = |J·(c·g)| = c·|J·g|`, exactly linear in contrast at any depth. Read that
way, λ < 1 is the signature of gates switching with contrast, and the log
response is what appears once they do. Caffe stays in the locally-linear regime
for its whole conv stack; `IMAGENET1K_V1` leaves it from mid-stack. This is a
**hypothesis, not a measurement** — the direct test is to count ReLU sign flips
between gray and grating against `c`, which needs a forward pass.

There is now a **second reading of the same evidence** that has to be excluded
first, because it predicts the identical λ: a tap with no signal at all also
reports λ ≈ 1 at high fit quality. 26 of the 37 Caffe conv taps sit within 0.15
of the noise-floor value at a mean power-family R² of 0.9988. The two readings
are separated not by shape but by **magnitude against repetition count**, and
that is a much cheaper job than counting gates — see the next section.

**The two controls differ in kind, not degree.** Scrambled Caffe runs *away*
from the log law with depth, to a classifier median of **λ = +2.75** at R² 0.972
— strongly supralinear, `c^2.75`, and log-like at **0 of 45** taps. Scrambled
`IMAGENET1K_V1` instead returns a log-*looking* **λ ≈ +0.17** — and this is the
honest limit of the statistic: **λ alone does not separate it from the trained
net.** What separates them is R², 0.918 against 0.978, and the non-monotonicity
underneath it. Gating those cells out would separate them cleanly and was
deliberately rejected — dropping inconvenient data is not a metric fix. Quote λ
with its R², always.

**`prob` carries no information beyond `logits` in a scrambled net.** In every
scrambled run the two surfaces correlate at **r = 1.000000** with ratio exactly
**1/1000**: with 1000 classes the softmax is in its affine regime, so
`Δprob = Δlogits/1000`. The trained net gives r = 0.961, so its softmax is doing
real work. This is a structural reason the control cannot reproduce the trained
net's final-layer behaviour, and it was invisible until λ returned identical
values at both taps.

### The metric has a noise floor, and `features.0` is sitting on it

Phase is drawn `U[0, 2π)`, so **`E[grating] = gray` exactly** — the sinusoid
averages away. The metric is the distance of the class-**mean** representation
from gray, so at any layer that is an affine function of the input the
population value of D is identically **zero**, and a finite run measures
sampling noise of order 1/√reps. This is a property of the metric as the paper
defines it (eq. 4), not of this implementation.

[`data-r250-s0`](../results/data-r250-s0/notes.md) measures that floor directly,
on raw pixels, with no network:

| | raw pixels | trained Caffe `features.0` | trained IN1K `features.0` | scrambled Caffe | scrambled IN1K |
|---|---|---|---|---|---|
| λ | **+0.925** | +0.922 | +0.923 | +0.926 | +0.926 |
| power R² | **0.985** | 0.985 | 0.986 | 0.985 | 0.985 |
| mean R² vs log c | **0.754** | 0.756 | 0.756 | 0.754 | 0.754 |

Two checkpoints, trained and scrambled, and a model-free control agree to three
decimals — because none of them is measuring a network. `features.0` is the only
VGG-19 tap upstream of every nonlinearity.

**The floor is exactly linear in contrast by construction.** `D = c · mean_i|W·ḡ|_i`
with `ḡ` independent of `c`. So **λ ≈ 1 at high fit quality is also what a dead
tap looks like**, and the reading "this layer responds linearly to contrast"
needs separate evidence. For scale, R² of a perfectly linear response against
log c on this 14-point grid is **0.736** — a property of the grid alone, which is
most of what the 0.754/0.756 column above is reporting.

**The separating test is repetition count, not shape.** A real response holds D
when reps change; a floor falls as 1/√reps. Against
[`data-r50-s0`](../results/data-r50-s0/notes.md) the raw-pixel surfaces give a
median ratio of **2.237** against √5 = **2.236**. Applied to the r50/r250 pair on
`IMAGENET1K_V1`, `D(50)/D(250)` per contrast:

| contrast | `features.0` | `features.19` | `classifier.3` | `logits` | `prob` |
|---|---|---|---|---|---|
| 0.0078 | **2.19** | 1.04 | 1.03 | 1.00 | 1.03 |
| 0.0625 | **1.77** | 1.00 | 1.00 | 1.01 | 1.01 |
| 0.258 | **1.12** | 1.00 | 1.00 | 1.01 | 1.00 |
| 1.0 | 1.01 | 1.00 | 1.01 | 1.01 | 1.01 |

Everything except `features.0` is flat at 1.00 — real signal, at every contrast.
`features.0` decays from the noise-floor value to 1.0 as contrast rises, which is
the crossover from noise-dominated to signal-dominated. Note that this pair
predates the tap fix, so its `features.0` holds the *post*-ReLU value: a
rectified tap has a non-zero population D, but only through the Jensen gap,
which requires the perturbation to actually flip gates. At low contrast it does
not, the ReLU is locally affine, and the floor argument applies unchanged.

**What this does and does not touch.** `prob`, `logits`, `classifier.*` and the
deeper `features.*` taps are rep-invariant and carry real signal — every headline
number stands. What changes is the reading of the shallow end: λ at `features.0`
is uninformative and the four-way agreement there is forced. Deeper convolutions
are **not** covered by the argument, because they sit downstream of a ReLU where
`E[a(x)] ≠ a(gray)`.

#### The floor is on the contrast axis only — `features.0` still measures conv1_1

The floor argument constrains `D = c · mean_i|W·ḡ_f|_i` in two of its three
factors and leaves the third alone. The magnitude is 1/√reps and the contrast
dependence is forced to λ = 1 — but `ḡ_f` is a sum of gratings all at frequency
`f`, so it stays spectrally concentrated there, and the **frequency profile is
conv1_1's radial amplitude response**. That is real, weight-dependent, and
training-dependent. `D` at `c = 1`, each run normalised to its own mean:

| run | f=1 | 3.5 | 14 | 56 | 75 | max/min |
|---|---|---|---|---|---|---|
| [raw pixels, no model](../results/data-r250-s0/notes.md) | 0.64 | 1.00 | 1.14 | 1.12 | 1.13 | **1.78** |
| trained · Caffe | 0.22 | 0.45 | 1.04 | 2.02 | 1.74 | **9.09** |
| trained · `IMAGENET1K_V1` | 0.17 | 0.36 | 0.88 | 2.19 | 2.20 | **12.89** |
| scrambled · Caffe | 0.58 | 0.91 | 1.12 | 1.25 | 1.24 | **2.16** |
| scrambled · `IMAGENET1K_V1` | 0.61 | 0.96 | 1.13 | 1.18 | 1.20 | **1.96** |

Trained conv1_1 is strongly band-pass at 9–13×, far outside the ~1.8× noise
scale. **Scrambling collapses it onto the raw-pixel profile** (r = 0.970 and
0.989 against the model-free run), and the two trained checkpoints agree with
each other at r = 0.995 — both learn a similar first-layer filter bank, as they
should.

So the free calibration point does exist; it is just on the frequency axis, not
the contrast axis. Trained weights give a 9× band-pass profile at `features.0`;
random, scrambled or unloaded weights give a flat ~1.8× one. That **is** a check
that trained weights reached the model, which λ at the same tap is not.

> **Corrected 2026-07-27.** Committed earlier the same day as "`features.0` is
> not a measurement of conv1_1" and "neither checkpoint is being measured
> there". Both overstate: the demotion holds for λ and for R² against log c, and
> does not hold for the frequency profile. This also matters for the paper,
> whose §5 iso-output comparison reports `conv1_1` at R² = 96% — that analysis
> reads the frequency axis, so it is not affected by the floor.

**Not yet done:** the same test at each of the 45 taps, which needs one
`--reps 1000` run per checkpoint and would settle how much of Caffe's flat λ ≈ 1
conv stack is locally-linear response and how much is floor.

### Is the log response at `prob` just the softmax?

The obvious objection: a softmax is a squashing nonlinearity, so a compressive
response at `prob` might be the output layer rather than anything learned.
Decomposing the last four taps says no — but with two real caveats.

| | `classifier.3` | `classifier.4` (ReLU) | `logits` | `prob` | ReLU's share | softmax's share |
|---|---|---|---|---|---|---|
| Caffe | +1.104 | +0.211 | +0.271 | +0.059 | **85.5%** | 20.3% |
| `IMAGENET1K_V1` | +0.623 | +0.010 | +0.230 | +0.165 | **134%** | 14.3% |

The largest single step toward the log law is a **ReLU, one layer before the
softmax**, on both checkpoints. On `IMAGENET1K_V1` it is starker: `classifier.4`
reaches λ = 0.010 — the log law almost exactly — and then the final linear layer
and the softmax push it back *away* to 0.165.

**The control settles the structural version of the objection.** In both
scrambled runs the softmax changes λ by less than 0.01 (2.748 → 2.743;
0.162 → 0.169) because it never leaves its affine regime. A softmax does not
compress by construction; it compresses only once the logits are large and
structured enough, which takes training. Were the squashing an artifact of the
output nonlinearity, it would appear in the scrambled net too.

What survives of the objection:

- **The softmax does contribute**, 14–20% of the move to log, and measurably so
  — the trained ratio is 1.235/1000 (Caffe) and 1.094/1000 against exactly
  1.000/1000 scrambled.
- **`prob` is measured against its own ceiling.** The bound derived in
  [Method](Method.md) is `D_prob ≤ 0.002`; the measurement reaches **0.001929,
  96.5% of it** (87.6% on `IMAGENET1K_V1`). It has not flattened — the top
  increment is still 1.75× the mean of the others, so the response is
  accelerating rather than saturating — but there is no headroom left to verify
  that, and `c = 1` is maximum contrast by definition, so the experiment cannot
  be extended to find out.

**Therefore `prob` is a poor headline tap.** `classifier.4` is more log-like on
`IMAGENET1K_V1`, contains no softmax, and has no ceiling. This bears directly on
the open disagreement in [Method](Method.md), whose "Expected results" table
names `prob` as the peak.

**What the metric change overturned.** The old statistic said the scrambled
control was *nearly as log-like* as the trained net (+0.120 vs +0.151 at `prob`,
exceeding it at 32 of 45 taps). That was normalisation by total variance failing
to distinguish "fits log better" from "fits nothing". The qualitative findings —
input agreement, the crossing at `classifier.4`, the two controls behaving
differently — survived both metric changes. What did not survive is any reading
of the scrambled column as log-like, and the sawtooth as a *general* mechanism
(see the correction above).

Regenerate the profile from the committed surfaces:

```bash
python -m log_response.run --load results/vgg19-r250-s0-alllayers-fixed --panels out/p.png
```

### The log-spaced grid is not what produces this — measured, not argued

The one methodological caveat under every number above: the default contrast
grid is log-spaced, which is **not neutral** between the two shapes being
distinguished. It hands the log end evenly spread leverage while bunching the
linear end's points near zero, so a log-shaped verdict could in principle be an
artifact of where the axis was sampled.

The control is the same 45 taps at `--reps 250` with `--contrasts linear` —
identical endpoints, sampled evenly instead of geometrically, nothing else
changed. Committed as
[`vgg19-r250-s0-alllayers-linear`](../results/vgg19-r250-s0-alllayers-linear/notes.md)
and
[`vgg19-scramble-r250-s0-alllayers-linear`](../results/vgg19-scramble-r250-s0-alllayers-linear/notes.md).

**Every claim above survives, and more cleanly on λ than on anything before
it.** Across the 45 layers, mean **|Δλ| is 0.045** trained and 0.024 scrambled,
against a profile spanning ~2.7 from conv stack to output — and **44 of 44**
consecutive steps move in the same direction on both grids. Not 43 of 44:
all of them. `prob` moves +0.165 → +0.180; `classifier.4`, the crossing ReLU,
+0.010 → +0.086. The individual shifts range −0.049 to +0.142.

λ is grid-free by construction — the exponent of a response is a property of the
response, pinned in `test_lambda_survives_the_grid_it_is_measured_on` — so
unlike either version of `logness`, this comparison is between two measurements
rather than two rulers.

> **Retraction (2026-07-26), left standing as a record.** An earlier version of
> this section claimed the grid shift was "systematically negative through the
> conv stack", negative at *every one* of the 37 conv taps. That was an artifact
> of comparing raw `R²_log − R²_lin` across two grids with different ceilings — a
> comparison that statistic did not support. It was then restated on the
> residual-ratio form (−0.092 mean, range −0.178 to +0.107, not uniform), and is
> now measured on λ above. The headline — that the depth profile survives the
> change of contrast grid — has been unaffected throughout, and now rests on a
> statistic that is comparable across grids by construction rather than by
> argument.

## VGG-19, full grid on `IMAGENET1K_V1`

> **This is not the paper's checkpoint.** §8.1 of the paper used MatConvNet's
> *"imported pre-trained original version"* of VGG-19 — the Oxford/Caffe
> weights, which is what `convert_weights.py` produces. On those weights the
> paper's numbers reproduce; on this one they do not. See
> [below](#which-checkpoint-the-paper-used-and-what-reproduces-on-it). This
> section is kept as the measured behaviour of the torchvision checkpoint, which
> is a different model, not as the reference run.

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

On this checkpoint the run disagrees with [Method](Method.md) on three counts:

1. **`prob` reaches 0.917, not 0.98.**
2. **R² does not peak at `prob`.** It peaks one layer earlier, at `classifier.3`
   (fc7, 0.928), and *dips* at `logits` (0.878) before the softmax lifts it
   again. "Highest at `prob`" does not survive the full grid.
3. **The scrambled control beats the trained net at the early and middle taps**
   (0.604 vs 0.548, 0.924 vs 0.869; italicised above). The learned contribution
   is confined to the classifier end, where the ordering does hold — and even
   there the gap is 0.149, far short of the documented 0.98 − 0.60 = 0.38.

All three are **specific to this checkpoint** — the next section runs the same
comparison on the paper's, where 1 and 2 do not arise. Note also that the
`features.0` row is on the noise floor (see above), so its trained-vs-scrambled
ordering in point 3 is not a fact about learned weights.

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

They do genuinely differ, and the depth profile above says where: diverging
through the conv stack and re-converging at the classifier. (They also "agree at
conv1_1", but that agreement is forced by the noise floor and is not evidence
about the checkpoints either way.) A real difference between two legitimately
different checkpoints.

### Which checkpoint the paper used, and what reproduces on it

§8.1 of [the paper](1701.04674-adaptation-as-readout.pdf) says the models were
run in **MatConvNet 1.0-beta20**, and that *"for VGG-19 and ResNet-152, we used
the imported pre-trained original versions"*. The original VGG-19 is
Simonyan & Zisserman's Caffe release — the weights `convert_weights.py`
converts. **Torchvision's `IMAGENET1K_V1` is a different training run**, and the
paper never used it.

Checked against the paper's four §5 claims, on the 45-tap runs at `--reps 250`:

| paper §5 | converted Caffe | `IMAGENET1K_V1` |
|---|---|---|
| `prob` R² = **98%** | **0.980** ✓ | 0.917 ✗ |
| *"much lower … up to layer fc7"* | fc7 0.750, conv1_1 0.756 ✓ | fc7 0.869 ✗ |
| `prob` is the top | peak of all 45 taps ✓ | peak is `classifier.4`, 0.928 ✗ |
| scrambled = **60%** | 0.429 ✗ | 0.768 ✗ |

**Three of the four reproduce on the paper's own weights, to three decimals.**
The [Method](Method.md) "expected results" table was not wrong; it was being
checked against a checkpoint the paper never used. This closes what had been
recorded there as a contested number.

The scrambled control remains a genuine disagreement — 0.429 against 0.60, with
the paper naming no permutation seed and the four-seed sweep above spanning
0.169. Per rule 4 that one stays stated, not reconciled.

This also reverses the framing of the section above it. "The converted Caffe run
is the outlier" is true only among the torchvision runs; measured against the
paper, `IMAGENET1K_V1` is the outlier and Caffe is the reference.

### Reproduced at curve level, not just at the summary numbers

Matching `prob` R² = 0.980 against a documented 98% is one number against one
number. The paper's Figure 3b holds the whole measurement — four
representations × 14 contrasts × 8 frequencies — so it can be compared directly.
`python -m log_response.figure3 --compare` does it: the curves survive as vector
polylines, the `c = 0` line fixes each panel's origin *and* proves the y-axis is
linear, so each panel needs one free scale factor and nothing else.

| panel | our tap | r | median resid | **r, frequency only** | resid | cells |
|---|---|---|---|---|---|---|
| `data` | [raw pixels](../results/data-r250-s0/notes.md) | 0.9649 | 12.2% | **−0.047** | 11.4% | 112 |
| `conv1_1` | `features.0` | 0.9935 | 8.1% | **0.982** | 9.4% | 112 |
| `fc8` | `logits` | 0.9998 | 0.6% | **0.999** | 0.5% | 112 |
| `prob` | `prob` | 0.9998 | 0.5% | **0.999** | 0.4% | 112 |

"Frequency only" divides each contrast row by its own mean first, removing the
contrast trend that both sides share and would otherwise be flattered by. It is
the column that means something.

**`fc8` and `prob` reproduce to 0.4–0.5% across 112 cells each** — the paper's
plotted curves and this repo's Caffe run are the same measurement, not merely
the same summary statistic.

**And `data` collapses to r = −0.047, exactly as predicted.** That was stated
before running it: if the `data` panel is the [noise floor](#the-metric-has-a-noise-floor-and-features0-is-sitting-on-it),
its cells are independent random draws in the paper's run and in ours, so they
*cannot* correlate. Uncorrelated is what a broken digitisation would also
produce — but the other three panels rule that out, since the same extraction
and the same contrast pairing give 0.982–0.999 there. This is the strongest
evidence for the noise floor in the repo, and it comes from the paper's own
figure.

`conv1_1` sits in between at 0.982, which is the same story from the other side:
noise-floor magnitude, real frequency structure.

### Contrast constancy — the paper's actual §5 headline

The log-linearity result is introduced in the paper as *"an interesting,
unexpected observation"*. Its stated headline for that section is contrast
constancy: *"Contrast constancy: bandpass transduction in first layers is later
corrected"* — a response strongly modulated by spatial frequency at low
contrast, converging to frequency-invariant at high contrast.

It reproduces, and now against the figure rather than against the prose. Ratio
of the largest to the smallest D across the 8 frequencies, at the extremes of
the contrast axis:

| tap | at lowest contrast | at c = 1 | paper's Figure 3b |
|---|---|---|---|
| Caffe `logits` (= `fc8`) | 67.5× | **1.78×** | 64.5× → **1.77×** |
| Caffe `prob` | 85.2× | **1.40×** | — → **1.40×** |
| Caffe `classifier.4` | 51.1× | 1.48× | not plotted |
| IN1K `prob` | 375× | 2.33× | (different checkpoint) |

Strongly band-pass at low contrast, near frequency-invariant at high contrast —
the paper's claim, and our Caffe run lands on its digitised values. (The paper's
`prob` at the lowest contrast has a curve that rounds onto the baseline, below
the figure's own resolution, so there is no ratio to read there.)

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
