# Operating point — weight-space forensics on the two VGG-19 checkpoints

**Not a run.** No `D(freq, contrast)` surface, no repetition count, no
`result.npz`. These are the gauge-invariant statistics
`log_response.operating_point` reads off a checkpoint's weights and off a few
hundred forward passes, kept here because rule 1 applies to them exactly as to a
run: the numbers quoted in `wiki/Results.md` need a committed artifact.

| file | checkpoint |
|---|---|
| `vgg19-caffe.json` | the converted Oxford/Caffe weights (`convert_weights.py`, fold verified at 2.9e-8) |
| `vgg19-in1k.json` | torchvision `vgg19` `IMAGENET1K_V1` |

Both from GitHub Actions run **31034992278**, `--reps 4 --seed 0`, same job, same
runner — so no difference between them can be a runner difference. Job total 25
min, of which the two probes were 4m09s and 4m10s.

## What is in each file

Per pre-activation tap (the 18 Conv2d/Linear layers a ReLU follows):

- `margin_median`, `margin_quantiles`, `margin_below_one` — how far units sit
  from their own threshold, in units of what a full-contrast grating does to
  them. The headline statistic.
- `bias_drive_ratio` — rms bias over rms image-driven pre-activation at gray.
- `off_fraction` — units already off at the gray reference.
- `flip_fraction` — fraction of ReLU gates flipped against gray, per contrast.
- `predicted_lambda`, `predicted_lambda_r2` — the λ the measured margin
  distribution implies under a Gaussian-drive model. **A model, not a
  measurement** (see `margin_response()`); it assumes the drive is zero-mean and
  exactly proportional to contrast.
- `dc_fraction`, `sparsity`, `kurtosis`, `effective_rank` — weight shape, all
  ratios within a layer.
- `lambda_pre` / `relu_layer` / `lambda_relu` — the join onto the committed
  `--layers all` runs. `predicted_lambda` belongs next to `lambda_relu`: the
  probe taps pre-activations, the model predicts what the ReLU downstream does.

`gauge_check` re-measures everything after a random ReLU rescaling symmetry, to
which every statistic here is invariant by construction. Measured drift **2.2e-7**
(Caffe) and **3.5e-7** (IN1K) — the statistics are reading the training, not the
coordinate system.

## What they showed

Conv-stack medians, against the committed λ profiles:

| | λ | margin | bias/drive | sparsity | kurtosis | eff. rank | kernel DC |
|---|---|---|---|---|---|---|---|
| Caffe | **+1.065** | **0.035** | **0.0021** | 0.102 | **7.80** | 0.738 | 2.72 |
| `IMAGENET1K_V1` | **+0.613** | **0.158** | **0.0990** | 0.086 | **1.88** | 0.892 | 4.78 |
| Gaussian weights | — | — | — | 0.080 | 0.00 | 0.985 | 0.67 |

Three things, written up in full in `wiki/Results.md`:

1. **The gate-flip hypothesis is falsified.** Caffe flips 1.6–22.6% of its gates
   at the *lowest* contrast on the grid, and the two checkpoints' flip fractions
   overlap almost exactly (r = +0.085, p = 0.63 against λ over 34 rectifiers).
   Caffe does not hold λ ≈ 1 by keeping gates still.
2. **It holds λ ≈ 1 because its units sit on their thresholds.** Margin 0.035
   against IN1K's 0.158, and the margin model predicts Caffe's conv λ to a median
   of 0.042 with nothing fitted. On IN1K it misses by 0.753 — structurally, since
   a rectifier fed a symmetric contrast-proportional drive cannot give λ < 1.
3. **The weight-shape signature is the one 5× more weight decay would leave**,
   and at conv1_1 the whole difference is the bias (3.0× the drive on IN1K,
   0.048× on Caffe). But two checkpoints differ in every row of the recipe at
   once, and `RandomResizedCrop`'s low-frequency bias fits the DC-tuning
   difference just as well. Not settled — see the last subsection there.

## Reproducing

    python -m log_response.operating_point --model vgg19 --gauge-check \
        --compare results/vgg19-r250-s0-alllayers-fixed \
        --out results/operating-point/vgg19-in1k.json

    python -m log_response.operating_point --model vgg19 --weights <caffe.pth> \
        --gauge-check --compare results/vgg19-r250-s0-alllayers-fixed-caffe \
        --out results/operating-point/vgg19-caffe.json

The committed files carry the join (`compare_run`, `lambda_pre`, `relu_layer`,
`lambda_relu`), so they read on their own.

**They also reproduce.** The numbers here were first measured on run
**31034992278** and re-measured on **31037478037**, a different runner, when the
workflow gained `--compare`: every statistic came back identical. The Caffe half
was measured a third time in the sandbox, from an independently converted
checkpoint, and agrees with the runner to **1.7e-7** across all statistics and
**7.6e-6** across the flip fractions. Nothing here is runner-dependent.
