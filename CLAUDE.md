# Notes for Claude

## Where things are

| File | Contents |
|---|---|
| `README.md` | Entry point: one-paragraph summary, offline quickstart, index of `wiki/`. Keep short. |
| `wiki/Running.md` | Install, commands, back-ends, flags; cost/wait-time table; per-environment tips — this sandbox (blocked weight hosts) and GitHub-hosted runners (4 cores on this public repo, 14 GB disk, 6 h cap). |
| `wiki/Results.md` | Measured numbers: the verified VGG-19 run and its scrambled control, the synthetic pipeline check, and open deviations from the documented expectations. |
| `wiki/Method.md` | The exact procedure — grating definition, contrast/frequency grids, the distance-of-means metric, the regression, caveats, and stronger tests to add. The spec the code is checked against. |
| `results/` | Committed runs, one directory each: `result.npz` (surfaces), `result.json` (fits), `run.json` (provenance), `notes.md` (prose). `results/README.md` is the index and states the conventions. |
| `log_response/` | The implementation. `gratings.py` (stimuli), `features.py` (model back-ends), `fit.py` (regression), `experiment.py` (driver + save/load), `panels.py` (the two-row per-layer figure), `provenance.py` (commit/versions/weight digest), `convert_weights.py` (Caffe/Keras VGG-19 → torchvision, with the preprocessing fold), `run.py` (CLI), `test_pipeline.py` (offline tests). |

Docs are intentionally few and short. Prefer extending an existing page over
adding a new one.

## Working on this repo

- Run from the repo root: `python -m log_response.run`, never as a script.
- `python -m log_response.test_pipeline` is the fast check — 24 tests, no
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

## Pre-activation taps were post-activation until 2026-07-25

Every hooked tap in every run committed before the `--layers all` sweep recorded
the **post**-activation value, whatever its name said. The hook did
`out.detach().float().cpu().numpy()`, which on CPU with float32 activations is
three no-ops and a view onto the module's own storage; torchvision builds VGG
with `nn.ReLU(inplace=True)`, so the ReLU overwrote the captured conv output
before the forward returned.

What this does and does not invalidate:

- **`logits` and `prob` are unaffected** — they come from the returned tensor,
  not from hooks. Every headline number (`prob` 0.917 / 0.976, the control
  series, the seed sweep, the lineage finding) stands.
- **`features.0`, `features.19` and `classifier.3` are mislabelled** in the
  committed runs. They are valid measurements of the *following* ReLU, not of
  the conv/linear output. Anything read off them as "pre-activation" is wrong.
- The paper's second representation level is specifically the *before-ReLU*
  conv1_1 output. It has never actually been measured here.

Fixed by cloning in the hook; `test_pre_activation_taps_survive_inplace_relu`
fails without it. Re-run before quoting any early- or middle-layer number.

## Needs doing

Ordered. Nothing here is blocking — every quoted number now has a committed run.

1. ~~**Audit the converted Caffe checkpoint.**~~ **Done — hypothesis tested and
   rejected.** Two distinct questions were being run together here; keep them
   apart:

   - *Is the conversion faithful?* — the audit. **Closed.** The suspect step was
     folding caffe preprocessing into conv1, because a gain error there rescales
     a grating's effective contrast and slides the contrast-response curve along
     its own axis, mimicking a checkpoint difference while argmax accuracy sees
     nothing. `convert_weights.py --verify` compares the folded conv1 against a
     directly-computed caffe path: relative error **2.9e-8**, best-fit gain
     **1.000000001**. conv1 is the only layer touching the input, so no
     input-gain or channel-scaling error survives anywhere.
     `test_preprocessing_fold_is_exact` pins the arithmetic offline.
   - *Do the two checkpoints genuinely differ, and where?* — the science, in
     item 2 below. The audit result makes this the only live explanation.
2. ~~**Measure the depth profile on both checkpoints.**~~ **Done.** All 45 taps,
   both checkpoints, trained and scrambled. They agree at conv1_1 to 0.001,
   diverge through the conv stack, and re-converge at the classifier. The
   crossover to log-like is carried by *rectifications*, not by depth: each conv
   pushes `logness` down, each ReLU pushes it up, and on Caffe the whole network
   crosses at one ReLU (`classifier.4`, -0.241 -> +0.133). See `wiki/Results.md`.
3. **Re-run the seed sweep with `--scramble-seed` fixed against `--seed`.** The
   done sweep varied both at once (one flag drove both until now), so it bounds
   permutation variance rather than isolating it.
4. **Reconcile `wiki/Method.md` with the measured grid.** Its "Expected results"
   table still states 0.98 at `prob` and calls `prob` the peak; two independent
   `IMAGENET1K_V1` runs say 0.913–0.917 and peak at `classifier.3`. Per rule 4,
   do not quietly edit one to match the other — decide which is the claim.

## Open threads

- **`--reps 250` on `IMAGENET1K_V1` disagrees with `wiki/Method.md` on three
  counts** — `prob` at 0.917 not 0.98, R² peaking at `classifier.3` rather than
  `prob`, and the scrambled control *exceeding* the trained net at the early and
  middle taps. See `wiki/Results.md`.
- **Weight lineage, not repetition count, drives the disagreement.** At fixed
  reps, changing the checkpoint moves `prob` 0.063 and the control 0.332; at a
  fixed checkpoint, changing reps 5× moves them 0.004 and 0.008. The converted
  Caffe run is the outlier; everything measured on `IMAGENET1K_V1` agrees.
- **The scrambled control is not a single number.** Four permutations at
  identical settings give `prob` 0.760 / 0.863 / 0.704 / 0.693 — spread 0.169,
  sd 0.078. The trained net is 0.913 there, so the learned contribution reads
  anywhere from 0.050 to 0.220 depending on the draw. Treat every single-seed
  control value (0.428, 0.768, and the documented 0.60) as one sample, and do
  not quote a trained-minus-scrambled gap from one seed.
- Runner wall time varies **2.0×** for identical work (58.6 / 69.7 / 116.5 /
  118.4 min over four jobs, byte-identical code). Size any new grid against the
  slow end or it can miss the 6 h cap.
- The log-like behaviour is produced by **rectifications**, not accumulated
  depth: convolutions push `logness` toward linear-in-contrast and ReLUs push it
  back, a sawtooth the three-tap view could not show.
- ~~The **linear-vs-log contrast grid** is the main untested caveat.~~ **Closed —
  tested, and the profile survives.** The default grid is log-spaced, which is
  not neutral between the two laws `logness` compares, so the whole depth profile
  was re-measured on `--contrasts linear` (same endpoints, even spacing, nothing
  else changed). Mean |Δ `logness`| 0.037 trained / 0.024 scrambled against
  effects of 0.2–0.4; 1/45 sign flips trained, 0/45 scrambled; 44/44 consecutive
  steps agree in direction, so the sawtooth is reproduced layer for layer. The
  residual shift is small but *systematic* — slightly toward "linear" through the
  conv stack, the direction even sampling predicts. See `wiki/Results.md` and the
  two `-alllayers-linear` runs.
- R² is a poor summary for the scrambled column: its spacing CV runs 3.5–4.1
  (against 0.6–0.9 trained), i.e. a spike at the top contrast that a line fits.
  Quote the CV alongside it, or prefer a different statistic.
