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

> **Runs from 2026-07-28 are the first that are not VGG-19.** `alexnet`,
> `vgg19_bn`, `resnet50` and `vit_b_16` — all `--layers all`, each with its
> scrambled control and its reps companion. `prob` λ across the six
> architecture/checkpoint combinations now measured:
>
> | +0.05 | +0.06 | +0.17 | −0.16 | −0.22 | −0.27 |
> |---|---|---|---|---|---|
> | AlexNet | VGG-19 Caffe | VGG-19 IN1K | ViT-B/16 | ResNet-50 | `vgg19_bn` |
>
> **Two of the four scrambled controls are invalid.** `vgg19_bn` and `resnet50`
> carry BatchNorm; `--scramble` permutes γ while the running statistics stay
> put, which decalibrates rather than degrades. Their numbers are recorded and
> explicitly not comparable — see either notes file.

| Run | Model | Reps | Weights | Headline | Notes |
|---|---|---|---|---|---|
| [`resnet50-r250-s0`](resnet50-r250-s0/notes.md) | ResNet-50, trained | 250 | `IMAGENET1K_V1` | peak **0.957 at `layer2.3.relu@2`** | The peak tap is a **reuse slot** — an activation the old hook discarded. `prob` λ −0.223. Residual-stream prediction **falsified**: 0% of deep taps near λ=1. |
| [`resnet50-r50-s0`](resnet50-r50-s0/notes.md) | ResNet-50, trained | 50 | `IMAGENET1K_V1` | 3/160 on the floor | `conv1` + `bn1` (the affine prefix) at 98.5/98.7% noise; ≤5.1% everywhere else. |
| [`resnet50-scramble-r250-s0`](resnet50-scramble-r250-s0/notes.md) | ResNet-50, scrambled | 250 | `IMAGENET1K_V1` | **not comparable** | Same BatchNorm decalibration as `vgg19_bn`: r(logits,prob) 0.673, ratio 1.7e-10, λ-R² 0.692. |
| [`resnet50-scramble-r50-s0`](resnet50-scramble-r50-s0/notes.md) | ResNet-50, scrambled | 50 | `IMAGENET1K_V1` | CI = whole search range | λ moves to +0.028 with interval [−3.00, +4.00]. Confirms the above measures nothing. |
| [`vit-b-16-r250-s0`](vit-b-16-r250-s0/notes.md) | ViT-B/16, trained | 250 | `IMAGENET1K_V1` | `prob` λ **−0.162** | **No ReLU anywhere** — GELU, LayerNorm — and λ still runs +0.93 → −0.62 mid-encoder. The gate-flip reading cannot be the mechanism. |
| [`vit-b-16-r50-s0`](vit-b-16-r50-s0/notes.md) | ViT-B/16, trained | 50 | `IMAGENET1K_V1` | 1/65 on the floor | Only `conv_proj`. The floor does **not** cross the first LayerNorm — LN is not affine in the input, BN in eval is. |
| [`vit-b-16-scramble-r250-s0`](vit-b-16-scramble-r250-s0/notes.md) | ViT-B/16, scrambled | 250 | `IMAGENET1K_V1` | `prob` λ **+0.711**, R² 0.981 | A **clean** control: r(logits,prob) 0.999975, ratio 1/1000. LayerNorm has no running statistics, so the scramble behaves. |
| [`vit-b-16-scramble-r50-s0`](vit-b-16-scramble-r50-s0/notes.md) | ViT-B/16, scrambled | 50 | `IMAGENET1K_V1` | λ +0.714 | Reproducible to three decimals — the contrast with the two BN controls. |
| [`alexnet-r250-s0`](alexnet-r250-s0/notes.md) | AlexNet, trained | 250 | `IMAGENET1K_V1` | `prob` **0.963**, λ **+0.053** | **The log law does not need depth.** 8 weight layers, and `prob` is the peak of 21 taps — the paper's structure, which among VGG runs only Caffe gave. |
| [`alexnet-r50-s0`](alexnet-r50-s0/notes.md) | AlexNet, trained | 50 | `IMAGENET1K_V1` | 1/21 on the floor | Only `features.0` (2.240 ≈ √5); ≤1.8% noise everywhere else. Cleaner than VGG-19. |
| [`alexnet-scramble-r250-s0`](alexnet-scramble-r250-s0/notes.md) | AlexNet, scrambled | 250 | `IMAGENET1K_V1` | `prob` 0.865, λ +0.015 | Control. λ is *closer to log* than trained — only R² (0.889 vs 0.985) separates them. Peak moves to `features.9`. |
| [`alexnet-scramble-r50-s0`](alexnet-scramble-r50-s0/notes.md) | AlexNet, scrambled | 50 | `IMAGENET1K_V1` | `prob` 0.857 | Control companion. Rep-invariant, so the 0.098 gap is a property of the weights. |
| [`vgg19-bn-r250-s0`](vgg19-bn-r250-s0/notes.md) | VGG-19+BN, trained | 250 | `IMAGENET1K_V1` | conv median λ **−0.071** | **BatchNorm moves the whole conv stack to the log law without adding a rectification.** Against Caffe's +1.06 and IN1K's +0.69. `prob` λ −0.268 — past log, saturating. |
| [`vgg19-bn-r50-s0`](vgg19-bn-r50-s0/notes.md) | VGG-19+BN, trained | 50 | `IMAGENET1K_V1` | 5/61 on the floor | `features.1` is a **BatchNorm layer reading pure noise** (99.3%) — the floor is a property of affineness, not of being first. Headline taps ≤0.8%. |
| [`vgg19-bn-scramble-r250-s0`](vgg19-bn-scramble-r250-s0/notes.md) | VGG-19+BN, scrambled | 250 | `IMAGENET1K_V1` | **not comparable** | The scramble decalibrates BN rather than degrading it: r(logits,prob) **0.162**, ratio 1e-10, softmax saturated. λ uninformative (R² 0.613). Do not table it against the others. |
| [`vgg19-bn-scramble-r50-s0`](vgg19-bn-scramble-r50-s0/notes.md) | VGG-19+BN, scrambled | 50 | `IMAGENET1K_V1` | CI spans the search range | Confirms the above: λ moves 1.5 across the rep change and the interval opens to [−3.00, +1.32]. |
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
| [`vgg19-scramble-r50-s0-p1-alllayers-caffe`](vgg19-scramble-r50-s0-p1-alllayers-caffe/notes.md) | VGG-19, scrambled | 50 | converted Caffe | `prob` **0.516** | Permutation sweep, `--seed 0` fixed. The high end — and peaks at `features.0`, so no tap beats the noise floor. |
| [`vgg19-scramble-r50-s0-p2-alllayers-caffe`](vgg19-scramble-r50-s0-p2-alllayers-caffe/notes.md) | VGG-19, scrambled | 50 | converted Caffe | `prob` 0.443 | Permutation 2. λ +3.00. |
| [`vgg19-scramble-r50-s0-p3-alllayers-caffe`](vgg19-scramble-r50-s0-p3-alllayers-caffe/notes.md) | VGG-19, scrambled | 50 | converted Caffe | `prob` **0.422** | Permutation 3, the low end. Sweep spans 0.422–0.516; the paper's 0.60 is outside it. |
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

> **The per-frequency λ is recorded too, since 2026-07-28.** Each
> `per_frequency` entry carries `lambda`, `lambda_ci` and `lambda_r2` next to
> the log fit's `r2`/`slope`/`intercept`. The top-level `lambda` is the *median*
> of those eight, and it discards more than the differences it tends to get
> compared on — within one run λ spans up to 1.75 across frequency against 0.43
> between architectures, so two runs with equal median λ can have quite
> different responses. Again nothing was re-run; the surfaces already held it.

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
