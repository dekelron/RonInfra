# Notes for Claude

## Where things are

| File | Contents |
|---|---|
| `README.md` | Entry point: one-paragraph summary, offline quickstart, index of `wiki/`. Keep short. |
| `wiki/Running.md` | Install, commands, back-ends, flags; cost/wait-time table; per-environment tips — this sandbox (blocked weight hosts) and GitHub-hosted runners (2 cores, 14 GB disk, 6 h limit). |
| `wiki/Results.md` | Measured numbers: the verified VGG-19 run and its scrambled control, the synthetic pipeline check, and open deviations from the documented expectations. |
| `wiki/Method.md` | The exact procedure — grating definition, contrast/frequency grids, the distance-of-means metric, the regression, caveats, and stronger tests to add. The spec the code is checked against. |
| `log_response/` | The implementation. `gratings.py` (stimuli), `features.py` (model back-ends), `fit.py` (regression), `experiment.py` (driver + save/load), `run.py` (CLI), `test_pipeline.py` (offline tests). |

Docs are intentionally few and short. Prefer extending an existing page over
adding a new one.

## Working on this repo

- Run from the repo root: `python -m log_response.run`, never as a script.
- `python -m log_response.test_pipeline` is the fast check — 14 tests, no
  downloaded weights, runs anywhere.
- Long runs: background them and wait on the output file rather than watching
  (see the cost table in `wiki/Running.md`). Always `--save`; the `D(freq,
  contrast)` surfaces are the expensive product and `--load` re-fits without a
  model.

## The trap worth remembering

A failed pretrained-weight download does **not** fail the run. `TorchvisionModel`
falls back to random init with only a `RuntimeWarning`, and the experiment then
reports plausible-looking but meaningless numbers — the log response only exists
in a *trained* net. In this sandbox `download.pytorch.org` and `huggingface.co`
are blocked, so this is the default outcome of `--model vgg19`. Check for the
warning, or pass `--weights`. On GitHub runners the hosts are reachable and this
does not apply.

Numbers quoted in `wiki/Results.md` came from converted weights, not
`IMAGENET1K_V1` directly; if you get access to the real checkpoint, re-run and
reconcile — including the scrambled control, which currently disagrees with
`wiki/Method.md` (0.428 measured vs 0.60 documented).
