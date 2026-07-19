# Log-contrast response — reverse-engineered from Dekel (2017)

A runnable reimplementation of the **log-response result** from Ron Dekel,
*"Human perception in computer vision"* (arXiv:1701.04674, ICLR-2017 submission).
Built **without** the original experiment code — reconstructed from the paper's
description of the method and result.

## The result being reproduced

From the paper's contrast section (as reconstructed; see *Provenance* below):

- Stimuli are **sinusoidal gratings** spanning all combinations of **spatial
  frequency × contrast**, each compared against a **uniform gray** reference
  image.
- The DNN correlate is the **mean absolute change in the network's
  representation** — an **L1 distance** between the representation of a grating
  and of the gray reference, measured per layer.
- **End-computation layers** show a response that is **band-pass in spatial
  frequency at low contrast** (strongly frequency-modulated) and **near
  contrast-constant at high contrast**.
- **`L1` vs `log10(contrast)` is close to linear**, with the paper reporting
  **R² ≈ 0.98** at the final (`prob`) layer, **averaged across spatial
  frequency**. Equivalently: **log-spaced contrast inputs become linearly
  (evenly) spaced** in the end-layer representation — a **Weber–Fechner
  logarithmic transducer** emerging in a network trained only for object
  recognition.

This module measures exactly those quantities and reports the per-frequency and
pooled R², plus a "spacing CV" that quantifies how evenly the log-spaced
contrasts land in representation space (0 = perfectly even).

## What's here

| File | Role |
|------|------|
| `gratings.py`   | Sinusoidal grating stimuli; Michelson contrast about mid-gray; log-spaced contrasts × spatial-frequency grid; phase averaging. |
| `features.py`   | `TorchvisionModel` (real ImageNet CNN, layer taps via forward hooks, L1 metric) and `SyntheticFrontEnd` (offline, weight-free pipeline verifier). |
| `fit.py`        | `L1 = a·log10(contrast) + b` least-squares fit, R², per-frequency and pooled; log-spacing uniformity (CV of consecutive gaps). |
| `experiment.py` | End-to-end driver → L1 response surface `L1(freq, contrast)` per layer, fits, and figures. |
| `run.py`        | CLI. |
| `test_pipeline.py` | Offline self-tests (no downloaded weights). |

## Running it

Offline pipeline verification (runs anywhere, no model download):

```bash
python -m psychophysics.log_response.run --model synthetic --figures out/
python -m psychophysics.log_response.test_pipeline
```

Real ImageNet CNN (needs torchvision pretrained weights reachable, or a local
`state_dict`):

```bash
python -m psychophysics.log_response.run --model vgg19   --figures out/
python -m psychophysics.log_response.run --model resnet50 --weights resnet50.pth --figures out/
```

`vgg19`, `resnet50` etc. are any `torchvision.models` arch. Default layer taps
span early→end computation and always include the softmax `prob` output — the
layer at which the paper's R² ≈ 0.98 is reported.

## Reverse-engineering decisions (defensible choices, stated plainly)

Where the paper's exact settings aren't specified in what we could access, these
choices were made and are easy to vary in `GratingConfig`:

- **Contrast** is Michelson contrast about a mid-gray (0.5) background, so every
  stimulus and the reference share the same mean luminance; only the modulation
  changes. Contrasts are **log-spaced** from 0.01 to 1.0 (the log spacing is the
  substance of the result).
- **Spatial frequency** is in **cycles-per-image** (resolution-independent for a
  fixed input size): `{2,4,8,16,32,64}` by default.
- **Phase** is averaged over 4 quadrature phases to remove phase-specific
  sampling artifacts; **orientation** is fixed (vertical) by default.
- **L1** = mean absolute difference over all units of a tapped layer, between
  grating and gray representations; averaged over phases.
- **Preprocessing** is standard ImageNet normalization for the torch back-end.

## Offline verification vs. the real phenomenon

This sandbox blocks the usual pretrained-weight hosts (`download.pytorch.org`,
`huggingface.co`), and the log response is a **consequence of training** — it
does not appear in a randomly initialized net. So the real-CNN numbers must be
produced on a machine where weights are reachable (the command above is ready).

To verify the **analysis** offline, `SyntheticFrontEnd` is a weight-free
stand-in: CSF-weighted radial band-pass energy → compressive `log(1+k·E)`
nonlinearity. It is **not** a model of any network — it exists only to prove the
measurement/fit code reads the intended quantities. On it:

- the pre-compression **`energy`** stage is **not** log-linear (mean R² ≈ 0.50);
- the compressive **`output`** stage **is** (mean R² ≈ 1.00, spacing CV ≈ 0.01),
  and the CSF weighting makes the family **band-pass across spatial frequency**.

That contrast (linear stage fails the log fit, compressive stage passes it) is
the sanity check that the pipeline is measuring the right thing. It is a
demonstration of the *method*, and is labeled as such — **not** evidence about
any real network, and **not** a claim to have reproduced the 2017 numbers here.
The parallel (rather than converging) frequency curves reflect the stand-in's
lack of divisive normalization; true high-contrast constancy is expected to be
measured on the trained CNN, not asserted from this stand-in.

## Provenance / honesty note

The original paper PDF (arXiv, OpenReview) is blocked by this session's egress
policy; the method and the R² ≈ 0.98 figure were reconstructed from web-search
snippets of the paper. Treat the exact numbers as **to be verified against the
PDF** once it is accessible. The stimulus/metric/fit design here is a faithful,
defensible reconstruction of the described procedure, not a copy of the original
code. Nothing in this module should be presented as a reproduction of the
paper's numbers until it has been run on trained weights and checked against the
source PDF.
