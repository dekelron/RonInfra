# Results

One directory per run, committed. Each holds `result.npz` (the canonical
`D(freq, contrast)` surfaces), `result.json` (fit summary, diffs readably),
`run.json` (provenance) and `notes.md` (what it was for, what it showed).

> **Runs from 2026-07-27 carry two surfaces.** `surfaces` is the paper's
> distance-of-means `D` and stays the headline; `mean_of_distances` is the other
> order of operations, `mean_r mean_i |a_i(x_r) − gray_i|`. It rides along
> because `D` has population value **zero** at any layer affine in the input
> while `D_mod` does not — so where a layer's two λ disagree, the primary metric
> is reporting its own sampling noise. `result.json` carries it under
> `mean_of_distances`. Earlier runs simply lack it and load unchanged; adding it
> left every committed surface bit-identical.

| Run | Model | Reps | Weights | Headline | Notes |
|---|---|---|---|---|---|
| [`vgg19-r250-s0-alllayers-linear`](vgg19-r250-s0-alllayers-linear/notes.md) | VGG-19, trained | 250 | `IMAGENET1K_V1` | 45 taps, linear grid | Grid control. Profile survives: mean \|Δλ\| 0.045, **44/44** steps agree in direction. |
| [`vgg19-scramble-r250-s0-alllayers-linear`](vgg19-scramble-r250-s0-alllayers-linear/notes.md) | VGG-19, scrambled | 250 | `IMAGENET1K_V1` | 45 taps, linear grid | Control for the above. Mean \|Δλ\| 0.024; read against its R² 0.72, which is what makes λ here uninformative. |
| [`vgg19-r250-s0-alllayers-fixed`](vgg19-r250-s0-alllayers-fixed/notes.md) | VGG-19, trained | 250 | `IMAGENET1K_V1` | all 45 taps | Depth profile. conv median +0.69, `prob` +0.165. (Its λ 0.922 at conv1_1 is the noise floor, not a measurement — see `data-r250-s0`.) |
| [`vgg19-scramble-r250-s0-alllayers-fixed`](vgg19-scramble-r250-s0-alllayers-fixed/notes.md) | VGG-19, scrambled | 250 | `IMAGENET1K_V1` | all 45 taps | Control. λ ≈ +0.17 looks log-like; its R² 0.918 and 41% non-monotone cells are what separate it. |
| [`vgg19-r250-s0-alllayers-fixed-caffe`](vgg19-r250-s0-alllayers-fixed-caffe/notes.md) | VGG-19, trained | 250 | converted Caffe | all 45 taps | Conv stack flatly **linear** (λ +1.06, R² 0.999); one ReLU takes it to +0.21, `prob` +0.059. |
| [`vgg19-scramble-r250-s0-alllayers-fixed-caffe`](vgg19-scramble-r250-s0-alllayers-fixed-caffe/notes.md) | VGG-19, scrambled | 250 | converted Caffe | all 45 taps | Control. Runs *away* to λ ≈ **+2.75** (R² 0.972) — supralinear, log-like at 0/45 taps. |
| [`vgg19-scramble-r250-s0-alllayers`](vgg19-scramble-r250-s0-alllayers/notes.md) | VGG-19, scrambled | 250 | `IMAGENET1K_V1` | all 45 taps | **Superseded** — ran before the tap fix; conv taps hold ReLU output. |
| [`vgg19-r250-s0`](vgg19-r250-s0/notes.md) | VGG-19, trained | 250 | `IMAGENET1K_V1` | **0.928** (`classifier.3`) | The documented grid. Disagrees with Method.md on three counts. |
| [`vgg19-scramble-r250-s0`](vgg19-scramble-r250-s0/notes.md) | VGG-19, scrambled | 250 | `IMAGENET1K_V1` | 0.924 (`features.19`) | Control. Exceeds the trained net at the early/middle taps. |
| [`vgg19-r50-s0-in1k`](vgg19-r50-s0-in1k/notes.md) | VGG-19, trained | 50 | `IMAGENET1K_V1` | 0.921 (`classifier.3`) | Separates weight lineage from reps: lineage is the cause. |
| [`vgg19-scramble-r50-s0-in1k`](vgg19-scramble-r50-s0-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 0.924 (`features.19`) | Control, seed 0, and the seed-sweep write-up. |
| [`vgg19-scramble-r50-s1-in1k`](vgg19-scramble-r50-s1-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 0.941 (`features.19`) | Seed 1. `prob` 0.863 — the high end. |
| [`vgg19-scramble-r50-s2-in1k`](vgg19-scramble-r50-s2-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 0.920 (`features.19`) | Seed 2. `prob` 0.704. |
| [`vgg19-scramble-r50-s3-in1k`](vgg19-scramble-r50-s3-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 0.843 (`features.19`) | Seed 3. `prob` 0.693 — the low end. |
| [`vgg19-r50-s0`](vgg19-r50-s0/notes.md) | VGG-19, trained | 50 | converted Caffe | 0.976 (`prob`) | Differs from the `IMAGENET1K_V1` runs because of its checkpoint, not its reps. |
| [`vgg19-scramble-r50-s0`](vgg19-scramble-r50-s0/notes.md) | VGG-19, scrambled | 50 | converted Caffe | 0.428 (`prob`) | Control. Sits 0.33 below both `IMAGENET1K_V1` controls. |
| [`vgg19-r50-s0-alllayers-caffe`](vgg19-r50-s0-alllayers-caffe/notes.md) | VGG-19, trained | 50 | converted Caffe | **1/45 taps on the floor** | The reps companion to the r250 run: only `features.0` falls with reps (2.222 ≈ √5). The flat λ≈1 conv stack is **real**. |
| [`vgg19-scramble-r50-s0-alllayers-caffe`](vgg19-scramble-r50-s0-alllayers-caffe/notes.md) | VGG-19, scrambled | 50 | converted Caffe | 1/45 on the floor | Control. Its supralinear λ ≈ +2.76 is a real measurement too. |
| [`vgg19-r50-s0-alllayers-in1k`](vgg19-r50-s0-alllayers-in1k/notes.md) | VGG-19, trained | 50 | `IMAGENET1K_V1` | 1/45 on the floor | `features.1`/`.2` are the only partial cases (31%, 36%) — and where λ +1.67 vs λ_mod +1.01 exposes it. |
| [`vgg19-scramble-r50-s0-alllayers-in1k`](vgg19-scramble-r50-s0-alllayers-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 1/45 on the floor | Control. Max noise fraction outside `features.0/1/2` is 1.8%. |
| [`data-r250-s0`](data-r250-s0/notes.md) | raw pixels | 250 | none | λ **+0.925**, R² 0.985 | The paper's `data` row, and the metric's **noise floor**. `features.0` reproduces it to 3 decimals on both checkpoints. |
| [`data-r50-s0`](data-r50-s0/notes.md) | raw pixels | 50 | none | D(50)/D(250) = **2.237** | Companion to the above: makes the 1/√reps scaling checkable (√5 = 2.236). |

> **`logness` was removed on 2026-07-26 and replaced by `λ`.** Prose in any
> `notes.md` written before that date quotes the retired statistic — a race
> between `D = a + b·log c` and `D = a + b·c`, which measured nothing useful
> because *neither* line fits this data (the trained net is convex in `log c` at
> 95% of cells, the scrambled control non-monotone at 41%). `λ` is the exponent
> of `D = a + b·(c^λ − 1)/λ`: **0 is the log law, 1 linear in contrast**, and it
> comes with a confidence interval. Nothing was re-run — `result.npz` holds the
> surfaces, so every directory here re-fits, and each `result.json` now records
> `lambda`, `lambda_ci` and `lambda_r2` where it recorded `logness`,
> `fit_quality` and `logness_r2diff`. **When a note and its `result.json`
> disagree, `result.json` is right.** Read λ against `lambda_r2`, never alone.
> See `wiki/Results.md` for what changed substantively.

Note that "best mean R²" is not `prob` for either r250 run — that is the finding,
not a slip. Read the r250 rows with the spacing CV in their notes: the scrambled
column reaches 0.76–0.92 with a CV of 3.5–4.1, so a high R² there is a line
through frequencies that disagree, not an even log ladder.

The r250 pair came from the GitHub path and committed themselves from the job;
the r50 pair were run in the sandbox on converted weights. `run.json`
distinguishes them (`environment.platform`, `weights.source`).

Re-fit and re-plot any run without torch, weights, or network:

```bash
python -m log_response.run --load results/vgg19-r50-s0 --panels out/panels.png
python -m log_response.run --load results/vgg19-r50-s0 --figures out/
```

## Conventions

- **Slug**: `<model>[-<variant>]-r<reps>-s<seed>` — the axes that actually vary.
  It does **not** encode weight lineage, so `vgg19-r50-s0` is ambiguous between
  the converted Caffe checkpoint and `IMAGENET1K_V1`. When they would collide,
  append a lineage tag (`-in1k`, `-caffe`); the workflow's `slug_suffix` input
  does this, and it refuses to start rather than overwrite an existing
  directory. Overwriting is worse than it looks: `save_run_dir` never clobbers
  `notes.md`, so the prose would survive while the numbers underneath changed.
- **Always committed**: the four files above. A surface is
  `n_layers × n_freq × n_contrast` floats — a few KB regardless of `--reps`, so
  even a 28 000-forward run costs ~7 KB. Keep every run.
- **Never committed**: figures (~100× larger than the data behind them, and
  regenerate from the npz), weights, activations.
- **`run.json` is the trust record.** `weights.pretrained_verified` must be
  `true` for any run quoted as a result; `false` means an untrained control and
  `null` means the question does not apply (synthetic back-end). `run.py`
  refuses to save a `false` run unless `--allow-random-init` was passed
  deliberately.

New runs write this layout themselves:

```bash
python -m log_response.run --model vgg19 --weights W --reps 250 \
    --save-run results/vgg19-r250-s0 --notes "full grid"
```

Runs produced on the GitHub-hosted path
([workflow](../.github/workflows/log-response.yml)) commit their own directory
here from the job — the sandbox cannot download Actions artifacts, so CI pushing
the result is what makes it reachable. Their `run.json` carries the runner's
provenance, so they are distinguishable from sandbox runs without a naming
convention for it.
