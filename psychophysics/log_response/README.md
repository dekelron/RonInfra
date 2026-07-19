# Log-contrast response — reverse-engineered from Dekel (2017)

A runnable reimplementation of the **log-response result** from Ron Dekel,
*"Human perception in computer vision"* (arXiv:1701.04674, ICLR-2017 submission),
built **without** the original experiment code (the paper's public package ships
no analysis code). The exact procedure — Section 5, **Equation 4**, Figures 3/10/11
— is written up in [`METHOD.md`](METHOD.md); this file covers the implementation.

## The result being reproduced

- Stimuli: **full-field sinusoidal gratings** over a grid of **14 log-spaced
  contrasts × 8 spatial frequencies**, each vs. a **uniform gray** reference.
  Per `(contrast, frequency)` cell, **250 stimuli** with **random orientation and
  random phase**.
- Metric (**Eq. 4**): for each unit, average its activation over the 250 stimuli,
  subtract its gray activation, take the absolute value, average over units —
  `Dℓ(c,f) = mean_i |mean_r a_i(x_r) − a_i(gray)|`. This is the **distance of the
  class-mean representation from gray**, *not* the mean per-stimulus distance
  (the ordering matters — see below).
- End-layer response is **band-pass in spatial frequency at low contrast** and
  comparatively **frequency-flat at high contrast** (contrast-constancy-like).
- `D` vs `log(contrast)` is close to **linear per frequency**; the paper reports
  **mean R² = 0.98** at the softmax **`prob`** layer (averaged over the 8
  frequencies). Scrambling learned weights within each layer drops it to **0.60**.
  Equivalently: log-spaced contrasts become **evenly spaced** in the end-layer
  representation — an emergent ratio-sensitive (Weber-like) population code.

## The subtle part: Eq. 4 is distance-of-means, not mean-of-distances

`‖E[A] − b‖₁ ≤ E[‖A − b‖₁]` (Jensen). The paper averages activations across the
random phase/orientation draws **before** the absolute value, so phase-specific
activity cancels first (important for signed, pre-ReLU activations). This module
implements that ordering, and `test_pipeline.py` has a dedicated test
(`test_experiment_uses_distance_of_means_not_mean_of_distances`) proving the
driver cancels a phase-varying signed unit rather than accumulating its energy.

## What's here

| File | Role |
|------|------|
| `METHOD.md`     | The exact paper procedure (Eq. 4), grids, reported numbers, guardrails. |
| `gratings.py`   | Sinusoidal gratings; Michelson contrast about mid-gray; the paper's 14-contrast / 8-frequency grids; random orient/phase sampling (on the fly). |
| `features.py`   | `TorchvisionModel` (real CNN; hook-tapped layers; `logits`+`prob`; within-layer weight **scrambling** control) and `SyntheticFrontEnd` (offline, weight-free pipeline verifier). |
| `fit.py`        | `D = a·log10(c) + b` least-squares fit, R² per-frequency and pooled; log-spacing uniformity (CV of consecutive gaps). |
| `experiment.py` | End-to-end driver → Eq. 4 `D(freq, contrast)` surface per layer, fits, figures. |
| `run.py`        | CLI. |
| `test_pipeline.py` | Offline self-tests (8, no downloaded weights). |

## Running it

Offline pipeline verification (runs anywhere, no model download). Use a small
repetition count so it finishes quickly:

```bash
python -m psychophysics.log_response.run --model synthetic --reps 12 --figures out/
python -m psychophysics.log_response.test_pipeline
```

Real ImageNet CNN. The paper's full grid is 14×8×250 ≈ 28k forward passes per
model (heavy on CPU; use a GPU and/or `--reps` below 250 to explore):

```bash
python -m psychophysics.log_response.run --model vgg19 --figures out/
python -m psychophysics.log_response.run --model vgg19 --scramble --figures out_scrambled/
python -m psychophysics.log_response.run --model resnet152 --weights r152.pth --figures out/
```

Any `torchvision.models` arch works. Layer taps span early→late and always add
`logits` (fc8) and the softmax `prob` — the layer of the paper's R² ≈ 0.98.

## Offline verification vs. the real phenomenon

This sandbox blocks the pretrained-weight hosts (`download.pytorch.org`,
`huggingface.co`), and the log response is a **consequence of training** — it
does not appear in a random-init net. So the real-CNN numbers must be produced
where weights are reachable (pass `--weights` with a local `state_dict` if the
hub is blocked).

To verify the **analysis** offline, `SyntheticFrontEnd` is a weight-free stand-in
(CSF-weighted radial band-pass energy → compressive `log(1+k·E)`). It is **not** a
model of any network and **not** a reproduction of the paper's numbers — only a
check that the measurement/fit code reads the intended quantities. On the paper
grid: the pre-compression `energy` stage is **not** log-linear (mean R² ≈ 0.5);
the compressive `output` stage **is** (mean R² ≈ 0.97), with CSF making the family
band-pass across frequency. Its frequency curves stay parallel rather than
converging at high contrast — true contrast constancy needs divisive
normalization and is expected to be measured on the trained CNN, not asserted
from this stand-in.

## Provenance / honesty note

Method and reported numbers (R² 0.98 at `prob`; 0.60 scrambled; the 14 contrasts
and 8 frequencies; Eq. 4's averaging order) are taken from a detailed
reconstruction of arXiv:1701.04674 §5/Eq. 4/Figs 3,10,11 — the manuscript and
figures, not analysis code (which the paper never released). Several stimulus
details are undocumented in the source and are marked as replication assumptions
in `METHOD.md` (grating equation, phase/orientation distributions, conv-unit
flattening). This module is a **conceptually faithful** reconstruction, not a
bit-exact copy, and nothing here should be presented as reproducing the paper's
numbers until it has been run on trained weights.
