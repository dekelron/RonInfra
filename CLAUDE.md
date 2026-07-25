# Notes for Claude

## Where things are

| File | Contents |
|---|---|
| `README.md` | Entry point: one-paragraph summary, offline quickstart, index of `wiki/`. Keep short. |
| `wiki/Running.md` | Install, commands, back-ends, flags; cost/wait-time table; per-environment tips — this sandbox (blocked weight hosts) and GitHub-hosted runners (2 cores, 14 GB disk, 6 h limit). |
| `wiki/Results.md` | Measured numbers: the verified VGG-19 run and its scrambled control, the synthetic pipeline check, and open deviations from the documented expectations. |
| `wiki/Method.md` | The exact procedure — grating definition, contrast/frequency grids, the distance-of-means metric, the regression, caveats, and stronger tests to add. The spec the code is checked against. |
| `results/` | Committed runs, one directory each: `result.npz` (surfaces), `result.json` (fits), `run.json` (provenance), `notes.md` (prose). `results/README.md` is the index and states the conventions. |
| `log_response/` | The implementation. `gratings.py` (stimuli), `features.py` (model back-ends), `fit.py` (regression), `experiment.py` (driver + save/load), `panels.py` (the two-row per-layer figure), `provenance.py` (commit/versions/weight digest), `run.py` (CLI), `test_pipeline.py` (offline tests). |

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

## The trap, and why it is now closed

The log response only exists in a *trained* net, so a run whose weights failed to
load measures nothing. This used to be silent: `TorchvisionModel` and `CLIPModel`
fell back to random init with only a `RuntimeWarning`, and the run reported
plausible-looking meaningless numbers that were indistinguishable from real ones
once saved. Both now **raise** instead, and `--save`/`--save-run` refuse to
persist an unverified run; `--allow-random-init` is the deliberate opt-in for an
untrained control, and stamps `pretrained_verified: false`.

Keep that property. When adding a back-end, set `weights_ok` (`True` pretrained,
`False` untrained, `None` where weights do not apply) and `weights_source` —
`run.py` reads them to decide whether a run may be saved.

Open threads:

- The scrambled control measured **0.428** against the **0.60** in
  `wiki/Method.md`. One seed, 50 reps — see
  `results/vgg19-scramble-r50-s0/notes.md` for what to test first.
- Results came from converted Caffe weights, not `IMAGENET1K_V1` directly. Given
  the real checkpoint, re-run and reconcile.
- The full `--reps 250` grid has not been run.
