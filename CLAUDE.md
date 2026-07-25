# Notes for Claude

## Where things are

| File | Contents |
|---|---|
| `README.md` | Entry point: one-paragraph summary, offline quickstart, index of `wiki/`. Keep short. |
| `wiki/Running.md` | Install, commands, back-ends, flags; cost/wait-time table; per-environment tips — this sandbox (blocked weight hosts) and GitHub-hosted runners (4 cores on this public repo, 14 GB disk, 6 h cap). |
| `wiki/Results.md` | Measured numbers: the verified VGG-19 run and its scrambled control, the synthetic pipeline check, and open deviations from the documented expectations. |
| `wiki/Method.md` | The exact procedure — grating definition, contrast/frequency grids, the distance-of-means metric, the regression, caveats, and stronger tests to add. The spec the code is checked against. |
| `results/` | Committed runs, one directory each: `result.npz` (surfaces), `result.json` (fits), `run.json` (provenance), `notes.md` (prose). `results/README.md` is the index and states the conventions. |
| `log_response/` | The implementation. `gratings.py` (stimuli), `features.py` (model back-ends), `fit.py` (regression), `experiment.py` (driver + save/load), `panels.py` (the two-row per-layer figure), `provenance.py` (commit/versions/weight digest), `run.py` (CLI), `test_pipeline.py` (offline tests). |

Docs are intentionally few and short. Prefer extending an existing page over
adding a new one.

## Working on this repo

- Run from the repo root: `python -m log_response.run`, never as a script.
- `python -m log_response.test_pipeline` is the fast check — 21 tests, no
  downloaded weights, runs anywhere.
- Long runs: background them and wait on the output file rather than watching
  (see the cost table in `wiki/Running.md`). Always `--save-run`; the `D(freq,
  contrast)` surfaces are the expensive product and `--load` re-fits without a
  model.

## Rules

Four, and the first one is the one that has actually been broken:

1. **Every number quoted in a doc must have a committed run behind it.** If you
   cannot link `results/<slug>/`, do not quote the number. A run that lives only
   in a CI artifact does not count — artifacts expire, and the sandbox cannot
   download them. Finish a run by committing its directory.
2. **Never commit figures.** They are ~100× the surfaces that generate them and
   regenerate with `--load <run> --panels out/panels.png`. `.gitignore` enforces
   this; do not add exceptions.
3. **A new back-end sets `weights_ok` and `weights_source`** (see the trap
   below). `run.py` reads them to decide whether a run may be saved at all.
4. **Do not assert a contested number.** Where runs disagree — as they currently
   do on the scrambled control — state the disagreement and link the runs.
   Editing one number to match another hides the finding.

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

## Two run paths

Runs come from one of two places, and `run.json` tells them apart after the fact
(`environment.platform`, `cpu_count`, `weights.source`):

- **This sandbox.** Weight hosts are blocked, so `--model vgg19` cannot fetch
  anything; pass `--weights` with a converted local `state_dict`. See
  `wiki/Running.md`.
- **GitHub Actions** (`.github/workflows/log-response.yml`). Weights download
  normally, so `--model vgg19` works directly — this is the path for a run
  that should be quoted. Standard runners are free and unmetered on this public
  repo, 4 cores, and the full `--reps 250` grid takes 58 min against a 6 h cap.

  The commit-back step runs on **every** event and pushes `results/<slug>/` to
  the ref the run was launched from, so a dispatch from `master` commits to
  `master`. It is verified working — the two r250 directories were pushed by
  `github-actions[bot]`, and the rebase-retry handled both matrix jobs landing
  minutes apart. Still **check the directory actually arrived** before quoting
  numbers; the artifact is a fallback the sandbox cannot read.

## Needs doing

Ordered. Nothing here is blocking — every quoted number now has a committed run.

1. **`IMAGENET1K_V1` at `--reps 50`** — the single cell separating "weight
   lineage" from "repetition count" as the cause of the disagreement below.
   ~12–23 min on a runner; dispatch it and it commits itself.
2. **Vary the scramble seed** before any control value is trusted. Three
   disagreeing values exist and no seed has been repeated.
3. **Reconcile `wiki/Method.md` with the r250 result.** Its "Expected results"
   table still states 0.98 at `prob` and calls `prob` the peak; the measured
   grid says 0.917 and peaks at `classifier.3`. Per rule 4, do not quietly edit
   one to match the other — decide which is the claim and say why.

## Open threads

- **`--reps 250` on `IMAGENET1K_V1` disagrees with `wiki/Method.md` on three
  counts** — `prob` at 0.917 not 0.98, R² peaking at `classifier.3` rather than
  `prob`, and the scrambled control *exceeding* the trained net at the early and
  middle taps. See `wiki/Results.md`.
- The scrambled control has now measured **0.428** (Caffe weights, 50 reps) and
  **0.768** (`IMAGENET1K_V1`, 250 reps) against the **0.60** documented. Three
  values, no seed repeats yet — vary the scramble seed before trusting any.
- Runner wall time varies **2.0×** for identical work (58.6 / 69.7 / 116.5 /
  118.4 min over four jobs, byte-identical code). Size any new grid against the
  slow end or it can miss the 6 h cap.
- Weight lineage and repetition count changed together between those two runs.
  `IMAGENET1K_V1` at `--reps 50` is the single cell that separates them.
- R² is a poor summary for the scrambled column: its spacing CV runs 3.5–4.1
  (against 0.6–0.9 trained), i.e. a spike at the top contrast that a line fits.
  Quote the CV alongside it, or prefer a different statistic.
