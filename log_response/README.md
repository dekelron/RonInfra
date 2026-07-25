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
| `features.py`   | `TorchvisionModel` (real CNN; hook-tapped layers; `logits`+`prob`; within-layer weight **scrambling** control), `CLIPModel` (open_clip image tower; zero-shot-prompt `prob` layer), `HFVLMModel` (generative VLM; next-token `prob` layer), `SAMModel` (Segment Anything encoder; optional mask-decoder taps), and `SyntheticFrontEnd` (offline, weight-free pipeline verifier). |
| `fit.py`        | `D = a·log10(c) + b` least-squares fit, R² per-frequency and pooled; log-spacing uniformity (CV of consecutive gaps). |
| `experiment.py` | End-to-end driver → `D(freq, contrast)` surface per layer, fits, figures; save/load a run (`.npz` surfaces + `.json` fit summary). |
| `run.py`        | CLI. |
| `test_pipeline.py` | Offline self-tests (16, no downloaded weights; the VLM/SAM tests build tiny random-config models in memory and skip when transformers is absent). |

## Running it

Offline pipeline verification (runs anywhere, no model download). Use a small
repetition count so it finishes quickly:

```bash
python -m log_response.run --model synthetic --reps 12 --figures out/
python -m log_response.test_pipeline
```

Real ImageNet CNN. The full grid is 14×8×250 ≈ 28k forward passes per model —
**≈ 1.3 h on 4 CPU cores** for VGG-19, so no GPU is needed for the headline
result (drop `--reps` below 250 to explore faster):

```bash
python -m log_response.run --model vgg19 --figures out/
python -m log_response.run --model vgg19 --scramble --figures out_scrambled/
python -m log_response.run --model resnet152 --weights r152.pth --figures out/
```

[`COMPUTE.md`](COMPUTE.md) has measured per-arch costs and free ways to run
this unattended — including a dispatchable
[GitHub Actions workflow](../.github/workflows/log-response.yml) that runs the
grid on a free public-repo runner and uploads the surfaces and figures. The
`hf:` VLM and `sam:` back-ends are the ones that genuinely want a GPU.

### Saving and re-using a run

The `D(freq, contrast)` surfaces are the expensive product (a full grid is ~28k
forward passes). `--save` persists them so you never recompute to re-fit or
re-plot:

```bash
python -m log_response.run --model vgg19 --save runs/vgg19          # writes runs/vgg19.{npz,json}
python -m log_response.run --load runs/vgg19 --figures out_vgg19/   # re-fit + re-plot, no model
```

`<base>.npz` is the canonical artifact (surfaces + grids + metadata); the fits
are re-derived on load (`summarise_layer` is deterministic), so the reloaded
report is identical to the original. `<base>.json` is a human-readable fit
summary — per-layer/per-frequency slopes and R² — convenient for cross-model
aggregation (degenerate layers serialise as `null`). `--load` ignores all model
flags. In Python: `save_result(result, path)` / `load_result(path)`.

Any `torchvision.models` arch works. Layer taps span early→late and always add
`logits` (fc8) and the softmax `prob` — the layer where R² is highest. Default
intermediate taps exist for VGG and ResNet; for other archs pass module names
explicitly, e.g.:

```bash
python -m log_response.run --model vit_b_16 --layers encoder.layers.encoder_layer_5,encoder.ln
```

## CLIP (VLM image towers)

The same measurement runs on CLIP image encoders via `open_clip`
(`pip install open_clip_torch`):

```bash
python -m log_response.run --model clip:ViT-B-32 --figures out_clip/
python -m log_response.run --model clip:ViT-B-32 --scramble --figures out_clip_scr/
python -m log_response.run --model clip:ViT-B-32:laion2b_s34b_b79k --figures out_laion/
```

CLIP has no class probabilities, so the terminal layers are rebuilt from the
contrastive head: `embed` (image embedding), `zs_logits` (logit-scaled cosine
similarities against a fixed text prompt set), and `prob` (their softmax — a
zero-shot classifier). Two caveats:

- **`prob` is prompt-set-conditional.** Unlike VGG, where the 1000 classes are
  part of the trained model, the zero-shot classifier depends on the chosen
  prompts (default: a built-in 64-prompt spread; override with `--prompts
  file.txt`, one prompt per line). Report the prompt set with any numbers. The
  TV bound becomes `0 ≤ D_prob ≤ 2/N` for N prompts.
- **The headline VGG numbers do not transfer.** The R² ≈ 0.98 result is a
  property of *classification* training; whether contrastive language-image
  training produces the same log-contrast response is exactly the open question
  this back-end lets you measure, not something to assume.

Default intermediate taps are the first/middle/last visual transformer block
(or `visual.layer1..4` for ResNet towers); `--scramble` permutes weights in
both towers, with text features recomputed from the scrambled text encoder.

## Generative VLMs (LLaVA, Qwen-VL, ...)

The same measurement runs through a Hugging Face image-text-to-text model
(`pip install transformers pillow`):

```bash
python -m log_response.run --model hf:llava-hf/llava-1.5-7b-hf \
    --device cuda --dtype float16 --reps 50 --figures out_llava/
python -m log_response.run --model hf:Qwen/Qwen2-VL-2B-Instruct \
    --device cuda --dtype bfloat16 --frequencies 3.5,7,14,28 --reps 25
```

A generative VLM's forward pass needs text, so the measurement is
**conditional on a fixed instruction** (default `"Describe this image."`;
override with `--instruction`), rendered through the model's chat template.
The terminal layers are `logits` and `prob` — the **next-token distribution at
the final position**, i.e. the model's distribution over the first response
token given grating + instruction. Default intermediate taps: last
vision-tower block, the multimodal projector, and a middle + the last LLM
decoder layer (introspected; override with `--layers`). Caveats:

- **`prob` is instruction- and template-conditional** — report both with any
  numbers. The TV bound becomes `0 ≤ D_prob ≤ 2/V` for vocab size V.
- **Decoder taps mix image- and text-token positions**; the text is fixed, so
  shapes are stable and the distance-of-means metric applies unchanged, but a
  per-position analysis is future work.
- **Cost**: the full grid is ~28k forward passes. For a 7B model use a GPU
  with `--dtype float16`/`bfloat16` and shrink `--reps` / `--frequencies`
  (the driver is batch-1). If the HF hub is unreachable, point `--model
  hf:/path/to/local/dir` (or `--weights`) at a downloaded copy.
- As with CLIP: whether generative VLM training preserves the log-contrast
  response through the LLM is the open question this measures, not something
  the VGG numbers imply.

## Segment Anything (SAM)

The measurement also runs on the SAM image encoder (`facebook/sam-vit-base`
/`-large`/`-huge` via transformers):

```bash
python -m log_response.run --model sam --device cuda --reps 50 --figures out_sam/
python -m log_response.run --model sam:facebook/sam-vit-huge --mask-decoder \
    --device cuda --dtype float16 --frequencies 3.5,7,14,28 --reps 25
```

SAM has no classifier and no contrastive head, so there is **no `prob`
analogue** — the default measurement is encoder-only: vision-transformer
blocks plus `embed`, the final image embedding. This asks whether
log-contrast compression emerges in a representation trained for
*segmentation*, completing the training-objective comparison
(classification / contrastive / generative / dense prediction).
`--mask-decoder` additionally runs the decoder with a **fixed center-point
prompt**, adding `mask_logits` and `iou_scores` terminal layers — like the
VLM instruction, those are prompt-conditional. SAM's native input is
1024×1024, so forwards are heavy; shrink `--reps`/`--frequencies` to explore.
(Note for tinkerers: HF SAM configs default to `initializer_range=1e-10`, so
a *random-init* SAM produces vanishing activations — random-weight controls
should scramble pretrained weights via `--scramble` instead.)

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
