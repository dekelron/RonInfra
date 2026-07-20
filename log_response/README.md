# Log-contrast response

A runnable measurement: feed an ImageNet-trained CNN sinusoidal gratings at
log-spaced contrasts and show that the mean absolute change in its late-layer
representation (relative to a gray reference) is a linear function of
`log(contrast)`. The exact procedure — inputs, metric, fit — is in
[`METHOD.md`](METHOD.md); this file covers the implementation.

## The result

- Inputs: **full-image sinusoidal gratings** over a grid of **14 log-spaced
  contrasts × 8 spatial frequencies**, each vs. a **uniform gray** reference.
  Per `(contrast, frequency)` cell, **250 images** with **random orientation and
  random phase**.
- Metric: for each unit, average its activation over the 250 images, subtract
  its gray activation, take the absolute value, average over units —
  `D(c,f) = mean_i |mean_r a_i(x_r) − a_i(gray)|`. This is the **distance of the
  class-mean representation from gray**, *not* the mean per-image distance
  (the ordering matters — see below).
- End-layer response is **band-pass in spatial frequency at low contrast** and
  comparatively **frequency-flat at high contrast**.
- `D` vs `log(contrast)` is close to **linear per frequency**; at the softmax
  **`prob`** layer the mean R² across the 8 frequencies reaches **≈ 0.98** on
  VGG-19. Scrambling learned weights within each layer drops it to **≈ 0.60**.
  Equivalently: log-spaced contrasts become **evenly spaced** in the end-layer
  representation.

## The subtle part: distance-of-means, not mean-of-distances

`‖E[A] − b‖₁ ≤ E[‖A − b‖₁]` (Jensen). Averaging activations across the random
phase/orientation draws **before** the absolute value lets phase-specific
activity cancel first (important for signed, pre-ReLU activations). This module
implements that ordering, and `test_pipeline.py` has a dedicated test
(`test_experiment_uses_distance_of_means_not_mean_of_distances`) proving the
driver cancels a phase-varying signed unit rather than accumulating its energy.

## What's here

| File | Role |
|------|------|
| `METHOD.md`     | The exact procedure: inputs, metric, fit, grids, expected numbers. |
| `gratings.py`   | Sinusoidal gratings; Michelson contrast about mid-gray; the 14-contrast / 8-frequency grids; random orient/phase sampling (on the fly). |
| `features.py`   | `TorchvisionModel` (real CNN; hook-tapped layers; `logits`+`prob`; within-layer weight **scrambling** control) and `SyntheticFrontEnd` (offline, weight-free pipeline verifier). |
| `fit.py`        | `D = a·log10(c) + b` least-squares fit, R² per-frequency and pooled; log-spacing uniformity (CV of consecutive gaps). |
| `experiment.py` | End-to-end driver → `D(freq, contrast)` surface per layer, fits, figures. |
| `run.py`        | CLI. |
| `test_pipeline.py` | Offline self-tests (8, no downloaded weights). |

## Running it

Offline pipeline verification (runs anywhere, no model download). Use a small
repetition count so it finishes quickly:

```bash
python -m log_response.run --model synthetic --reps 12 --figures out/
python -m log_response.test_pipeline
```

Real ImageNet CNN. The full grid is 14×8×250 ≈ 28k forward passes per model
(heavy on CPU; use a GPU and/or `--reps` below 250 to explore):

```bash
python -m log_response.run --model vgg19 --figures out/
python -m log_response.run --model vgg19 --scramble --figures out_scrambled/
python -m log_response.run --model resnet152 --weights r152.pth --figures out/
```

Any `torchvision.models` arch works. Layer taps span early→late and always add
`logits` (fc8) and the softmax `prob` — the layer where R² is highest.

## Offline verification vs. the real phenomenon

If the pretrained-weight hosts (`download.pytorch.org`, `huggingface.co`) are
blocked, note that the log response is a **consequence of training** — it does
not appear in a random-init net. So the real-CNN numbers must be produced where
weights are reachable (pass `--weights` with a local `state_dict` if the hub is
blocked).

To verify the **analysis** offline, `SyntheticFrontEnd` is a weight-free stand-in
(band-pass energy → compressive `log(1+k·E)`). It is **not** a model of any
network and **not** a source of the headline numbers — only a check that the
measurement/fit code reads the intended quantities. On the full grid: the
pre-compression `energy` stage is **not** log-linear (mean R² ≈ 0.5); the
compressive `output` stage **is** (mean R² ≈ 0.97), with the band-pass gain
making the family band-pass across frequency. Its frequency curves stay parallel
rather than converging at high contrast — a frequency-flat high-contrast regime
does not fall out of this stand-in and is expected to be measured on the trained
CNN, not asserted here.
