# Results

One directory per run, committed. Each holds `result.npz` (the canonical
`D(freq, contrast)` surfaces), `result.json` (fit summary, diffs readably),
`run.json` (provenance) and `notes.md` (what it was for, what it showed).

| Run | Model | Reps | Best mean R² | Notes |
|---|---|---|---|---|
| [`vgg19-r50-s0`](vgg19-r50-s0/notes.md) | VGG-19, trained | 50 | **0.976** (`prob`) | Primary measurement; reproduces the documented 0.98. |
| [`vgg19-scramble-r50-s0`](vgg19-scramble-r50-s0/notes.md) | VGG-19, scrambled | 50 | 0.428 (`prob`) | Control. Disagrees with the documented 0.60 — see notes. |

Re-fit and re-plot any run without torch, weights, or network:

```bash
python -m log_response.run --load results/vgg19-r50-s0 --figures out/
```

## Conventions

- **Slug**: `<model>[-<variant>]-r<reps>-s<seed>` — the axes that actually vary.
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
