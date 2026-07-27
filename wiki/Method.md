# Exact method

The exact contrast / log-response procedure, so the implementation in this
directory can be checked against it.

> **Provenance.** The metric, the 250 random-orientation/phase draws, the
> within-layer scrambling control, the before-ReLU taps and the 224/227 input
> size are all specified in [the paper](1701.04674-adaptation-as-readout.pdf)
> (eq. 4 and §§8.1–8.2, 8.5). The grids are not stated in prose — §8.5 says only
> that images "depicted sine gratings at different contrast, spatial frequency,
> sine phase, and sine orientation combinations" — but they are recoverable from
> Figure 3b's vector geometry, and both match. See
> [below](#where-the-grids-come-from).

## Inputs

Full-image sinusoidal gratings, one per `(contrast c, frequency f, orientation
θ, phase φ)`:

```
I(c,f,θ,φ; x,y) = μ · [ 1 + c · sin( 2π f (x cosθ + y sinθ) / W + φ ) ]
```

* `μ` = mean gray level; `f` = cycles per image; `W` = image width.
* `c` is the Michelson contrast about `μ`.

**Contrast grid (14 values), ≈ log-spaced (≈ half-octave apart above the low end):**

```
c ∈ {1,2,3,4,6,8,11,16,23,33,46,64,92,128} / 128
  = {0.0078, 0.0156, 0.023, 0.031, 0.047, 0.063, 0.086, 0.125,
     0.180, 0.258, 0.359, 0.500, 0.719, 1.000}
```

The near-geometric spacing is exactly what makes `log c` nearly evenly spaced.

**Frequency grid (8 values), cycles/image:**

```
f ≈ {1, 1.75, 3.5, 7, 14, 28, 56, 75}
```

**Nuisance sampling:** 250 images per `(c,f)` with **random orientation** `θ ~
U[0,π)` and **random phase** `φ ~ U[0,2π)`. Total ≈ 14 × 8 × 250 = 28,000
forward passes per model + one gray reference.

### Where the grids come from

The paper's §8.5 does not list them, and Figure 3's axis labels were converted
to outlines when the MATLAB figure was embedded, so there is no text to read.
But the plotted curves survive as vector polylines, and the grids fall out of
their geometry. `Figure3.pdf` is object 470 in the PDF; its four sub-panels
(`data`, `conv1_1`, `fc8`, `prob`) each hold **15 polylines of 8 vertices**.

`python -m log_response.figure3` does the extraction and prints both grids;
`--compare` additionally checks the panels against the committed runs (see
[Results](Results.md#reproduced-at-curve-level-not-just-at-the-summary-numbers)).
`test_figure3_digitisation_recovers_the_documented_grids` pins it offline.

**Frequencies — recovered exactly.** All curves share one set of 8 x-positions.
Their spacing is six equal steps with a shorter one at each end, and reading the
equal step as a doubling gives:

```
1, 1.750, 3.501, 7.005, 14.009, 28.018, 56.062, 74.731
```

against the `{1, 1.75, 3.5, 7, 14, 28, 56, 75}` above — agreement to 0.007 in
units of the doubling step. The frequency grid is the paper's.

**Contrasts — 14, plus a zero reference.** 15 curves in a jet colormap, and the
lowest is flat to 0.0000 at the same height in all four panels: that is `c = 0`,
leaving 14 contrasts. Their values read off the two panels that are *linear* in
contrast, where curve height is proportional to `c` (`data` is the image itself;
`conv1_1` is a convolution). Normalising by the top curve, from `conv1_1`:

```
×128:  1.02  1.97  2.93  3.82  5.96  8.00  11.17  16.60  22.21  31.96  44.67  64.33  93.35  128
grid:  1     2     3     4     6     8     11     16     23     33     46     64     92     128
```

r = 0.9999, median error 1.8%, worst 4.5% (`data` gives the same at 2.6% median).
That confirms the span (exactly 1/128 to 1), the count, and the near-geometric
half-octave spacing — but the residual scatter is larger than the gap between
adjacent integers up top, so the *individual* values are the repo's reading, not
a measurement.

The scatter itself is informative. In the paper's `data` panel the
frequency-to-frequency spread is a **median 44%** of the curve height and shows
no trend with contrast — a multiplicative random factor, not a response. That is
the [noise floor](#the-noise-floor) signature, and `results/data-r250-s0`
reproduces it:

| | paper's `data` panel (digitised) | `data-r250-s0` |
|---|---|---|
| relative spread across frequency | median **44.1%** | median **41.4%** |
| max/min at `c = 1` | **1.64** | **1.78** |

So the paper's own `data` row is showing sampling noise, and this repo
reproduces it to within the precision of reading a figure.

## The metric (this is the subtle part)

For layer ℓ, scalar unit i (channel × spatial position flattened), the 250
randomized images `x_{c,f,r}`, and gray image `x₀`:

```
μ_ℓi(c,f) = (1/250) Σ_r  a_ℓi(x_{c,f,r})      # mean activation over the 250 images
b_ℓi      = a_ℓi(x₀)                          # activation for uniform gray
D_ℓ(c,f)  = (1/N_ℓ) Σ_i | μ_ℓi(c,f) − b_ℓi |  # mean |·| over units
```

**Order of operations matters.** The metric is
`| mean_r a_i(x_r) − a_i(gray) |` (distance of the *class-mean* representation
from gray), **not** `mean_r | a_i(x_r) − a_i(gray) |`. By Jensen's inequality
`‖E[A] − b‖₁ ≤ E[‖A − b‖₁]`, so these are different metrics: phase/orientation-
specific activity can cancel before the absolute value (matters most for signed,
pre-ReLU activations). D_ℓ measures the distance between the *expected* grating-
class representation and gray, not per-image response energy.

Convolutional "units": flatten channel and spatial dims so every scalar
activation counts.

**Which representations.** The paper's Figure 3b plots four: `data` (raw image
pixels), `conv1_1`, `fc8` and `prob`. Here those are `--model data`,
`features.0`, `logits` (verified bit-identical to `classifier.6`) and `prob`.
The paper's *primary* §5 analysis is a comparison of DNN iso-output curves
against human psychophysics; there is no human data in this repo, so only the
DNN half is implemented.

### The other ordering, recorded alongside

Every run from 2026-07-27 also records the metric with the operations reversed:

```
D_mod(c,f) = (1/250) Σ_r  (1/N_ℓ) Σ_i | a_ℓi(x_{c,f,r}) − b_ℓi |
```

This is **not** the paper's metric and does not replace it — `D` stays the
headline everywhere. It is recorded because the two disagree in a diagnostic
way. `D` has population value zero at any layer affine in the input (the
[noise floor](#the-noise-floor)); `D_mod` takes the absolute value per image, so
nothing cancels and shallow layers keep a real signal. **Where a layer's two λ
disagree, the primary metric is reporting its own sampling noise.**

It costs nothing: each image's distance collapses to a scalar immediately, so it
is one accumulator *number* per layer rather than an array, and the forward pass
dominates either way. `result.json` carries it under `mean_of_distances`; runs
saved before this date simply do not have it, and load unchanged.

**It also buys an exact calibration point, which `D` cannot provide.** For a
grating of contrast `c` about mean `μ`, the mean absolute deviation from gray is
`μ·c·mean|sin| = μ·c·(2/π)` — independent of frequency, orientation and phase.
At `μ = 0.5` that is `c/π`. Measured on raw pixels
([`data-r250-s0`](../results/data-r250-s0/notes.md)) it matches to **0.04%**
across all 14 contrasts, giving λ = **1.000** at R² = **1.000**. Any error in
the grating generator, the contrast convention or the metric shows up as a
deviation from a closed form. `D`'s population value at raw pixels is zero, so
there is nothing there to check a measurement against.

**Probability layer bound:** with p,q the 1000-class softmax vectors,
`D_prob = (1/1000) Σ_i |p_i − q_i| = (2/1000)·TV(p,q)`, so `0 ≤ D_prob ≤ 0.002`.
This bound means no *global* log law is possible — D must →0 near zero contrast
and saturate; the relation holds only over the finite sampled range (~1/128..1).

## Log-response regression

Per layer ℓ and per frequency f_k:

```
x_j = log c_j        (14 nonzero contrasts; log base is irrelevant to R²)
y_j = D_ℓ(c_j, f_k)
fit  y = α_ℓk + β_ℓk x   (linear, with intercept)
R²_ℓk = 1 − SSE/SST      (= squared Pearson corr of D vs log c)
R̄²_ℓ = (1/8) Σ_k R²_ℓk   # average R² across the 8 frequencies
```

"Averaged across spatial frequencies" = one regression per frequency, then
average the R² (each frequency keeps its own α, β). A rigorous run should also
report the alternative (average responses across frequency, then one fit).

## Contrast-exponent regression (λ) — the headline statistic

The regression above answers *how well does a straight line in log c fit?* It
cannot answer *is log the right shape at all?*, which is the first caveat below.
λ answers that one directly, and is what the code reports.

Per layer ℓ and frequency f_k, fit the nested power family:

```
y_j = a + b·(c_j^λ − 1)/λ      the λ → 0 limit is  a + b·ln c
```

`a` and `b` enter linearly at fixed λ, so they are profiled out by OLS and the
search over λ is one-dimensional (grid scan for the basin, golden section
inside it). Per layer, report:

```
λ_ℓ  = median over the 8 frequencies    # median: a non-monotone frequency
R²_ℓ = mean   over the 8 frequencies    #   returns an arbitrary exponent
CI   = { λ : RSS(λ) ≤ RSS_min·(1 + F(1, n−3, 0.95)/(n−3)) }   # profile-F
```

| λ | response |
|---|---|
| 0 | `a + b·ln c` — **the log law** |
| 0.5 | `∝ √c` |
| 1 | `a + b·c` — **linear in contrast** |
| < 0 | saturating |
| > 1 | accelerating |

One property the R²-of-a-log-fit does not have:

- **An uninformative fit is visible.** Pure noise returns the entire search
  range as its interval rather than a confident number.

And one that needed narrowing. `features.0` measures λ = 0.922–0.926 across all
four 45-tap runs, trained and scrambled, which was written up here as a forced
calibration point. **λ there checks the grating generator and the fitter, not
the model** — `features.0` sits on the metric's [noise floor](#the-noise-floor)
and a model-free run on raw pixels returns the same +0.925 at the same fit
quality, so nothing about the network can move it.

The calibration point does exist, on the *other* axis. The floor fixes the
magnitude and the contrast dependence of `D = c · mean_i|W·ḡ_f|_i` but not its
frequency profile, which is conv1_1's radial amplitude response. Trained weights
give a strongly band-pass `features.0` — max/min across frequency **9.09**
(Caffe), **12.89** (`IMAGENET1K_V1`) — while scrambled weights collapse to
**2.16** / **1.96**, essentially the model-free run's **1.78**. That is a check
that trained weights actually reached the model.

## The noise floor

Phase is drawn `U[0,2π)`, so `E[grating] = gray` **exactly**. Since D is the
distance of the class-*mean* representation from gray, any layer that is an
affine function of the input has population D = **0**, and a finite run measures
sampling noise of order 1/√reps. It follows from the metric as defined above,
so it applies to any implementation of it.

Consequences for reading a profile:

- **λ ≈ 1 at high fit quality is also what an empty tap looks like** — the floor
  is `D = c·mean_i|W·ḡ|_i` with `ḡ` independent of `c`, i.e. exactly linear in
  contrast whatever the weights.
- **The test is repetition count, not shape.** A real response holds D when reps
  change; a floor falls as 1/√reps. Measured on raw pixels: median D(50)/D(250)
  = 2.237 against √5 = 2.236.
- In VGG-19 only `features.0` is upstream of every nonlinearity. Deeper
  convolutions sit after a ReLU, where `E[a(x)] ≠ a(gray)`, and carry real
  signal — `features.19` holds its D across a 5× rep change.
- **The floor is on the contrast axis only.** `ḡ_f` stays spectrally
  concentrated at `f`, so an affine layer's *frequency* profile is still a real
  measurement of its filter bank even where its magnitude and its λ are not.

`--model data` measures the floor directly; see
[`results/data-r250-s0`](../results/data-r250-s0/notes.md) and
[Results](Results.md#the-metric-has-a-noise-floor-and-features0-is-sitting-on-it).

**Always quote λ with its R².** λ locates a response only insofar as the family
describes it, and this is where the scrambled control bites: at `prob` on
`IMAGENET1K_V1` the trained and scrambled runs return λ 0.165 and 0.169 —
indistinguishable — and only R² (0.952 against 0.823) separates them.

This replaced a statistic called `logness` on 2026-07-26, which raced the log
fit against a linear-in-contrast fit and reported which lost less. See
[Results](Results.md) for why that framing was wrong and what it changed.

## Expected results (VGG-19)

**Which checkpoint matters.** §8.1 of the paper ran MatConvNet 1.0-beta20 with
*"the imported pre-trained original version"* of VGG-19 — Simonyan & Zisserman's
Caffe release, which `convert_weights.py` converts. Torchvision's
`IMAGENET1K_V1` is a different training run and reproduces a different profile.
The table below is the paper's, so it is the Caffe number that tests it.

| Representation | Mean R² (D vs log c) | measured, Caffe | measured, IN1K |
|---|---|---|---|
| `prob` (1000-way softmax) | **0.98** | **0.980** ✓ | 0.917 |
| early/middle layers through `fc7` | much lower | 0.750 at fc7 ✓ | 0.869 |
| `prob` is the peak | — | peak of 45 taps ✓ | peak is `classifier.4` |
| `prob`, **weights scrambled within each layer** | **0.60** | 0.429 ✗ | 0.768 |

Three of the four reproduce on the paper's checkpoint. The scrambled control
does not, and is left as a stated disagreement per rule 4 — the paper names no
permutation seed, and four permutations at fixed settings span 0.169. See
[Results](Results.md#which-checkpoint-the-paper-used-and-what-reproduces-on-it).

Stronger than the table: digitising Figure 3b and comparing curve for curve puts
`fc8` and `prob` at **0.4–0.5% median residual over 112 cells each**, with the
shared contrast trend divided out. Four summary numbers become 448.

The paper's own headline for this section is not the log law but **contrast
constancy** — band-pass in spatial frequency at low contrast, converging to
frequency-invariant at high contrast. That reproduces too: at `prob` on Caffe
the spread across frequencies goes 85.2× → 1.40× along the contrast axis.

Controls to run: within-layer weight scrambling; comparison of `logits` vs
`prob`; comparison of distance-of-means vs mean-of-distances. Inputs 224×224 (or
227×227 for archs that expect it).

## Caveats

* `D = α + β log c` is **not** a power law (`log P = log k + γ log c`), and a high
  in-sample R² over 14 points does **not** uniquely identify a logarithm — a
  soft-log `log(1+c/σ)`, a power law `c^γ`, or a saturating form `c^n/(σ^n+c^n)`
  can fit similarly over this range.
  - **Partly addressed by λ**, which nests the power-law alternative and
    measures its exponent instead of leaving it as a possibility: the log law is
    λ = 0, `c^γ` is λ = γ, and a saturating form gives λ < 0. Measured, the
    conv stack of the converted Caffe checkpoint is λ ≈ 1 (linear in contrast,
    R² 0.999) and only `prob` reaches λ = 0.059 — so over most of the network
    the answer to this caveat is that the response is **not** a logarithm. The
    soft-log and Naka-Rushton forms are still not nested and remain untested.
  - **But λ ≈ 1 is ambiguous**, because it is also what the noise floor reads
    (above). 26 of the 37 Caffe conv taps sit within 0.15 of the floor's λ. A
    reps sweep separates the two; it has not been run per-tap.
* Scrambling dropping R² 0.98→0.60 shows learned organisation *strengthens* the
  effect; the residual 0.60 (architecture + softmax + metric) is substantial, so
  "learning adds 0.38" is not a clean causal statement. Measured on the paper's
  checkpoint the drop is larger still — 0.980 → 0.429 — which sharpens the
  direction of the claim without making the causal reading any cleaner.
* No individual unit computes a logarithm; the log-likeness is a property of the
  pooled response across units, not of any single unit.

## Stronger tests to add

**A per-tap reps sweep** is the cheapest and is now the most informative: one
`--reps 1000` run per checkpoint, compared against the committed r250, says
which of the 45 taps carry signal and which sit on the noise floor. It settles
how much of Caffe's flat λ ≈ 1 conv stack is a locally-linear response and how
much is an empty measurement — a question the shape of the curve cannot answer.

Then: fit and **hold out contrasts** to compare candidate laws — log, soft-log
`α+β log(1+c/σ)`, power `α+β c^γ`, saturating `α+β c^n/(σ^n+c^n)`; bootstrap
phase/orientation samples for CIs; repeat across init/scramble seeds; compare
**logits vs softmax**; analyse individual units vs the pooled response; extend
contrast below 1/128 to find where the apparent log breaks.

*(Distance-of-means vs mean-of-distances is done — both are now recorded on
every run, see above. What is still missing is a **model** run carrying both;
the only committed pair so far is raw pixels.)*
