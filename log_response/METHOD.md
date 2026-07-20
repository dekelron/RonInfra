# Exact method

The exact contrast / log-response procedure, so the implementation in this
directory can be checked against it.

## Stimuli

Full-field sinusoidal gratings, one per `(contrast c, frequency f, orientation
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

**Nuisance sampling:** 250 stimuli per `(c,f)` with **random orientation** `θ ~
U[0,π)` and **random phase** `φ ~ U[0,2π)`. Total ≈ 14 × 8 × 250 = 28,000
forward passes per model + one gray reference.

## The metric (this is the subtle part)

For layer ℓ, scalar unit i (channel × spatial position flattened), the 250
randomized stimuli `x_{c,f,r}`, and gray image `x₀`:

```
μ_ℓi(c,f) = (1/250) Σ_r  a_ℓi(x_{c,f,r})      # mean activation over the 250 stimuli
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

## Expected results (VGG-19)

| Representation | Mean R² (D vs log c) |
|---|---|
| `prob` (1000-way softmax) | **0.98** |
| early/middle layers through `fc7` | much lower (log-like compression develops late) |
| `prob`, **weights scrambled within each layer** | **0.60** |

Controls to run: within-layer weight scrambling; comparison of `logits` vs
`prob`; comparison of distance-of-means vs mean-of-distances. Inputs 224×224 (or
227×227 for archs that expect it).

## Caveats

* `D = α + β log c` is **not** a power law (`log P = log k + γ log c`), and a high
  in-sample R² over 14 points does **not** uniquely identify a logarithm — a
  soft-log `log(1+c/σ)`, a power law `c^γ`, or a saturating form `c^n/(σ^n+c^n)`
  can fit similarly over this range.
* Scrambling dropping R² 0.98→0.60 shows learned organisation *strengthens* the
  effect; the residual 0.60 (architecture + softmax + metric) is substantial, so
  "learning adds 0.38" is not a clean causal statement.
* No individual unit computes a logarithm; the log-likeness is a property of the
  pooled population response, not of any single unit.

## Stronger tests to add

Fit and **hold out contrasts** to compare candidate laws — log, soft-log
`α+β log(1+c/σ)`, power `α+β c^γ`, saturating `α+β c^n/(σ^n+c^n)`; bootstrap
phase/orientation samples for CIs; repeat across init/scramble seeds; compare
**logits vs softmax**; compare **distance-of-means vs mean-of-distances**;
analyse individual units vs population; extend contrast below 1/128 to find where
the apparent log breaks.
