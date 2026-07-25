# Results

One directory per run, committed. Each holds `result.npz` (the canonical
`D(freq, contrast)` surfaces), `result.json` (fit summary, diffs readably),
`run.json` (provenance) and `notes.md` (what it was for, what it showed).

| Run | Model | Reps | Weights | Best mean R² | Notes |
|---|---|---|---|---|---|
| [`vgg19-r250-s0`](vgg19-r250-s0/notes.md) | VGG-19, trained | 250 | `IMAGENET1K_V1` | **0.928** (`classifier.3`) | The documented grid. Disagrees with Method.md on three counts. |
| [`vgg19-scramble-r250-s0`](vgg19-scramble-r250-s0/notes.md) | VGG-19, scrambled | 250 | `IMAGENET1K_V1` | 0.924 (`features.19`) | Control. Exceeds the trained net at the early/middle taps. |
| [`vgg19-r50-s0-in1k`](vgg19-r50-s0-in1k/notes.md) | VGG-19, trained | 50 | `IMAGENET1K_V1` | 0.921 (`classifier.3`) | Separates weight lineage from reps: lineage is the cause. |
| [`vgg19-scramble-r50-s0-in1k`](vgg19-scramble-r50-s0-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 0.924 (`features.19`) | Control, seed 0. Matches the r250 control, not the Caffe one. |
| [`vgg19-r50-s0`](vgg19-r50-s0/notes.md) | VGG-19, trained | 50 | converted Caffe | 0.976 (`prob`) | The outlier. Its checkpoint, not its reps, is why. |
| [`vgg19-scramble-r50-s0`](vgg19-scramble-r50-s0/notes.md) | VGG-19, scrambled | 50 | converted Caffe | 0.428 (`prob`) | Control. Sits 0.33 below both `IMAGENET1K_V1` controls. |

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
