# Notes for Claude

## Where things are

| File | Contents |
|---|---|
| `README.md` | Entry point: one-paragraph summary, offline quickstart, index of `wiki/`. Keep short. |
| `wiki/Running.md` | Install, commands, back-ends, flags; cost/wait-time table; per-environment tips — this sandbox (blocked weight hosts) and GitHub-hosted runners (4 cores on this public repo, 14 GB disk, 6 h cap). |
| `wiki/Results.md` | Measured numbers: the verified VGG-19 run and its scrambled control, the synthetic pipeline check, and open deviations from the documented expectations. |
| `wiki/Method.md` | The exact procedure — grating definition, contrast/frequency grids, the distance-of-means metric, the regression, caveats, and stronger tests to add. The spec the code is checked against. |
| `wiki/1701.04674-adaptation-as-readout.pdf` | The source paper. Its "mean absolute change in DNN representation between a gray image and sinusoidal gratings" is our `D`, and its "R² = 98% … for prob" is the contested number in `Method.md`'s expected-results table. |
| `results/` | Committed runs, one directory each: `result.npz` (surfaces), `result.json` (fits), `run.json` (provenance), `notes.md` (prose). `results/README.md` is the index and states the conventions. |
| `log_response/` | The implementation. `gratings.py` (stimuli), `features.py` (model back-ends, incl. `RawPixelModel` = the noise floor), `fit.py` (regression), `experiment.py` (driver + save/load), `panels.py` (the two-row per-layer figure and the λ depth profile), `provenance.py` (commit/versions/weight digest), `convert_weights.py` (Caffe/Keras VGG-19 → torchvision, with the preprocessing fold), `figure3.py` (digitises the paper's Figure 3b and compares it to a run), `run.py` (CLI), `test_pipeline.py` (offline tests). |

Docs are intentionally few and short. Prefer extending an existing page over
adding a new one.

## Working on this repo

- Run from the repo root: `python -m log_response.run`, never as a script.
- `python -m log_response.test_pipeline` is the fast check — 30 tests, no
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
   regenerate from a committed run: `--load <run> --panels out/panels.png` for
   the summary, `--load <run> --figures out/` for the per-layer set plus
   `lambda_profile.png`. `.gitignore` enforces this; do not add exceptions.
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
   both checkpoints, trained and scrambled. They diverge through the conv stack
   and re-converge at the classifier. (They also "agree at conv1_1 to 0.001",
   which was written up here as a finding and is not one — that tap is the
   metric's noise floor and agrees for every checkpoint, trained or not.) The
   crossover to log-like is carried by *rectifications*, not by depth. On λ:
   Caffe holds λ ≈ 1.0 (flatly **linear in contrast**, R² 0.999) through the
   whole conv stack, then one ReLU takes it `classifier.3` 1.10 → `classifier.4`
   0.21, and `prob` lands at **λ = 0.06 [−0.05, +0.13]** — the log law, measured.
   `IMAGENET1K_V1` starts lower (conv median 0.69) and drifts to 0.16. See
   `wiki/Results.md`.
3. **Re-run the seed sweep with `--scramble-seed` fixed against `--seed`.** The
   done sweep varied both at once (one flag drove both until now), so it bounds
   permutation variance rather than isolating it.
4. ~~**Reconcile `wiki/Method.md` with the measured grid.**~~ **Done — the table
   was right and the checkpoint was wrong.** The paper (§8.1) ran MatConvNet's
   *imported pre-trained original* VGG-19, i.e. the Oxford/Caffe weights, not
   torchvision's `IMAGENET1K_V1`. On the Caffe run `prob` = **0.980** against
   the documented 0.98, `prob` **is** the peak of all 45 taps, and fc7 sits at
   0.750 ("much lower", as documented). Nothing was edited to match: the
   disagreement was an artifact of testing the claim on a checkpoint the paper
   never used. The scrambled control (0.429 vs 0.60) stays open per rule 4.

5. **Run the per-tap reps sweep.** One `--reps 1000` run per checkpoint against
   the committed r250. A real response holds `D` when reps change; the noise
   floor falls as 1/√reps. This is what says which of the 45 taps are measuring
   anything — in particular how much of Caffe's flat λ ≈ 1 conv stack is a
   locally-linear response and how much is an empty tap. Cheaper and sharper
   than the gate-flip count below.

## Open threads

- **The paper's checkpoint is the converted Caffe one, and on it the paper
  reproduces.** §8.1 used MatConvNet's imported *original* VGG-19. Measured:
  `prob` 0.980 vs the documented 98%, `prob` the peak of 45 taps, fc7 0.750
  ("much lower") — three of four §5 claims, to three decimals. Only the
  scrambled control disagrees (0.429 vs 0.60). On `IMAGENET1K_V1` none of the
  three hold. **Do not describe the Caffe run as "the outlier"** — it is the
  outlier only among the torchvision runs; against the paper it is the
  reference and `IMAGENET1K_V1` is the deviation.
  - **It reproduces at curve level, not just on the summary numbers.**
    `python -m log_response.figure3 --compare` digitises Figure 3b and matches
    it panel by panel: with the shared contrast trend divided out, `fc8` and
    `prob` agree at **r = 0.999, 0.4–0.5% median residual over 112 cells each**.
    Four numbers become 448.
  - **And the `data` panel anti-confirms the noise floor**, which is the
    sharpest evidence for it in the repo: frequency-only correlation
    **−0.047**, i.e. uncorrelated, exactly as independent noise draws must be.
    Not a broken extraction — the same code and pairing give 0.982–0.999 on the
    other three panels.
- **Weight lineage, not repetition count, drives the difference between the two
  checkpoints.** At fixed reps, changing the checkpoint moves `prob` 0.063 and
  the control 0.332; at a fixed checkpoint, changing reps 5× moves them 0.004
  and 0.008. The `IMAGENET1K_V1` runs are internally consistent across reps and
  grids — which is not the same as agreeing with the paper.
- **Both grids are the paper's, recovered from Figure 3b's geometry.** §8.5 does
  not list them and the axis labels were flattened to outlines, but the curves
  survive as polylines in PDF object 470: four sub-panels (`data`, `conv1_1`,
  `fc8`, `prob`), each 15 polylines × 8 vertices. Frequencies come back exactly
  — 1, 1.750, 3.501, 7.005, 14.009, 28.018, 56.062, 74.731 — and the lowest
  curve is flat at the same height in all four panels, so it is `c = 0` and
  there are **14 contrasts**. Their values, read off the two panels linear in
  `c`, match `{1,2,3,4,6,8,11,16,23,33,46,64,92,128}/128` at r = 0.9999, median
  error 1.8%. Span and count are confirmed; the individual integers are within
  the figure's noise, so those stay the repo's reading.
  - **Corrected 2026-07-26.** Committed earlier the same day as "the grids are
    not in the paper — treat them as this repo's choice", on the basis of §8.5's
    prose alone, without checking the figure.
  - Bonus: the paper's `data` panel has a median 44% frequency-to-frequency
    spread with no trend in contrast — the noise floor, and
    `results/data-r250-s0` reproduces it at 41%.
- **Both orderings of the metric are now recorded on every run** (2026-07-27).
  `D` (distance of means) stays the headline — it is the paper's, eq. 4. `D_mod`
  = `mean_r mean_i |a_i(x_r) − gray_i|` rides along for free (each image's
  distance collapses to a scalar, so it is an accumulator *number* per layer).
  Why: `D` has population value zero at any affine layer, `D_mod` does not, so
  **where a layer's two λ disagree the primary metric is reporting its own
  noise**. `result.json` carries it under `mean_of_distances`; runs saved before
  this date lack it and load with it `None`. Adding it left every committed
  surface bit-identical.
  - **It brings an exact calibration point**, which the repo never had. On raw
    pixels `D_mod = μ·c·(2/π) = c/π` in closed form — matched to **0.04%** over
    all 14 contrasts, λ **1.000**, R² **1.000**. `D`'s population value there is
    zero, so it can calibrate nothing.
  - Only `data-r250-s0` / `data-r50-s0` carry both so far; **no model run does
    yet.**
- **The metric has a zero-population floor at every affine layer, and
  `features.0` is on it.** Phase ~ U[0,2π) makes `E[grating] = gray` exactly, so
  the distance-of-*means* metric has population `D` = 0 wherever the layer is
  affine in the input; what a finite run measures there is 1/√reps sampling
  noise. `results/data-r250-s0` (raw pixels, no model) returns λ **+0.925**,
  power-R² 0.985, log-R² 0.754 — and `features.0` returns 0.922–0.926 /
  0.985–0.986 / 0.7545–0.7558 in all four 45-tap runs, trained and scrambled,
  both checkpoints. Consequences: **λ ≈ 1 at high R² is also what a dead tap
  looks like**; λ at `features.0` checks the grating generator and the fitter,
  not the model; and R² 0.736 is what a perfectly linear response scores against
  log c on this grid regardless of anything. Everything from `features.19`
  outward is rep-invariant and carries real signal, so no headline number moves.
  - **The floor is on the contrast axis only — do not say `features.0` measures
    nothing.** `D = c·mean|W·ḡ_f|` and `ḡ_f` stays spectrally concentrated at
    `f`, so the *frequency* profile is conv1_1's radial amplitude response: real,
    weight-dependent, training-dependent. Trained is band-pass at max/min **9.09**
    (Caffe) / **12.89** (IN1K); scrambled collapses to **2.16** / **1.96** against
    the model-free run's **1.78**, at r = 0.97–0.99 with it. The two trained
    checkpoints agree at r = 0.995. So the free calibration point is real, on the
    frequency axis: trained weights give 9× band-pass, unloaded or scrambled ones
    give a flat floor. The paper's §5 iso-output `conv1_1` result (R² = 96%) reads
    that axis and is unaffected.
    - **Corrected 2026-07-27**, having been committed earlier the same day as
      "`features.0` is not a measurement of conv1_1".
- **The scrambled control is not a single number — and not a single shape.**
  Four permutations at identical settings give `prob` R² 0.760 / 0.863 / 0.704 /
  0.693 (spread 0.169, sd 0.078). On λ the spread is qualitative, not just
  numeric: seeds 0–2 give λ 0.19 / 0.19 / 0.16 with R² 0.77–0.91 (nothing fits),
  while **seed 3 gives λ = 1.04 [0.96, 1.13] at R² 0.992** — cleanly *linear in
  contrast*, because its response is flat until c ≈ 0.25 and then rises sharply.
  Per-frequency λ within seed 3 runs 1.77 → 0.80. Treat every single-seed
  control value as one sample, and do not quote a trained-minus-scrambled gap
  from one seed.
- Runner wall time varies **2.0×** for identical work (58.6 / 69.7 / 116.5 /
  118.4 min over four jobs, byte-identical code). Size any new grid against the
  slow end or it can miss the 6 h cap.
- **The ReLU sawtooth is real on `IMAGENET1K_V1` and absent on Caffe.** Measured
  per transition type on λ, trained runs, `features.*`:

  | | conv → ReLU | ReLU → conv |
  |---|---|---|
  | `IMAGENET1K_V1` | mean **−0.155**, 14/16 negative | mean **+0.166**, 10/11 positive |
  | converted Caffe | mean +0.023, 5/16 negative | mean −0.015, 7/11 negative |

  So "convolutions hold λ near 1 and ReLUs push it down" describes one
  checkpoint, not the mechanism. Caffe's conv stack is flat at λ ≈ 1 with no
  sawtooth at all; it leaves the linear regime only at `classifier.4`. Per rule
  4 this is stated as a disagreement between the runs — do not quote the
  `IMAGENET1K_V1` sawtooth as a property of VGG-19.
  - **Corrected 2026-07-26.** The general form of this claim was committed
    earlier the same day, carried over from the retired metric's write-up
    without re-checking it per checkpoint. It was wrong for Caffe.
- **Why λ ≈ 1 survives 33 layers — the live hypothesis.** The grating is a
  *perturbation* `gray + c·g` about a fixed operating point, and a ReLU net is
  piecewise linear, so while the perturbation does not flip ReLU gates,
  `D = |J·(c·g)| = c·|J·g|` exactly — linear in contrast at any depth. On this
  reading λ < 1 is the signature of gates actually switching with contrast, and
  the log response is what emerges once they do. Caffe stays in that regime for
  the whole conv stack; `IMAGENET1K_V1` leaves it gradually from mid-stack.
  **Untested.** The direct check is to count ReLU sign flips between gray and
  grating against `c`; that needs a forward pass, so it is an Actions job.
  - **A competing explanation has to be excluded first**, because it predicts
    the identical λ: an empty tap also reads λ ≈ 1 at high fit quality. 26 of
    the 37 Caffe conv taps sit within 0.15 of the noise-floor λ at mean R²
    0.9988. The reps sweep (item 5 in "Needs doing") separates the two and is
    far cheaper than gate counting — do that one first.
- ~~The **linear-vs-log contrast grid** is the main untested caveat.~~ **Closed —
  tested, and the profile survives.** The whole depth profile was re-measured on
  `--contrasts linear` (same endpoints, even spacing, nothing else changed).
  On λ: mean |Δλ| **0.045** trained / 0.024 scrambled against a profile spanning
  ~2.7, and **44/44** consecutive steps agree in direction. `prob` moves +0.165
  → +0.180. See `wiki/Results.md` and the two `-alllayers-linear` runs.
- **`logness` was removed on 2026-07-26 and replaced by `λ`.** Not redefined —
  removed. It had already been redefined once that day, from `R²_log − R²_lin`
  to a residual ratio, and the second look showed the whole framing was wrong:
  it raced two straight lines against each other, and **neither line describes
  the data**. The trained net is convex in `log c` at **95%** of
  layer-frequency cells; the scrambled control is not even monotone at **41%**
  of them. Because the race summed *squared* residuals, one contrast point out
  of 14 carried **20–55%** of the verdict.
  - What replaced it: `D = a + b·(c^λ − 1)/λ`, the Box-Tidwell/Box-Cox nested
    family, fitted by profiling out `a, b` and searching λ. **λ = 0 is the log
    law, λ = 1 linear in contrast, 0.5 a square root, negative saturating.** One
    measured parameter with a profile-F interval, not a choice between two
    guesses. It fits **0.92–0.998** everywhere the straight lines did not.
  - It is **grid-free by construction**, so the cross-grid comparisons the old
    statistic could not support are now valid.
  - Pure noise returns the **entire search range** as its interval instead of a
    confident number. That is the property the ±1 form never had: it answered 0
    both for "the two laws tie" and for "neither law fits".
  - Nothing was re-run — `result.npz` holds the surfaces, so every committed
    directory re-fits. `result.json` now carries `lambda`, `lambda_ci`,
    `lambda_r2` where it carried `logness`, `fit_quality`, `logness_r2diff`.
    **`logness` is gone from the code entirely**; notes written before this date
    quote it, and `result.json` is the authority where they disagree.
- **Always quote λ with its R².** λ locates a response only insofar as the
  family describes it, and this is exactly where the scrambled control bites:
  scrambled `IMAGENET1K_V1` returns a log-*looking* λ ≈ 0.17 while fitting at
  0.918 against the trained net's 0.978. λ alone does not separate them; λ with
  R² does. (Gating out the non-monotone cells would separate them too — that
  was considered and **deliberately rejected**: do not silently drop data.)
- **`prob` carries no information beyond `logits` in a scrambled net.** Their
  surfaces correlate at r = 1.000000 with ratio exactly 1/1000 in every
  scrambled run — with 1000 classes the softmax is in its affine regime, so
  `Δprob = Δlogits/1000`. In the trained net r = 0.961 and the softmax is doing
  real work. This is why the control cannot reproduce the trained net's
  final-layer behaviour, and it was invisible until λ returned identical values
  at the two taps.
- **The log response at `prob` is not just the softmax — but `prob` is still the
  wrong tap to headline.** Decomposed on λ, the ReLU at `classifier.4` carries
  **85.5%** (Caffe) / **134%** (`IMAGENET1K_V1`) of the move to log and the
  softmax 20.3% / 14.3%. On `IMAGENET1K_V1` `classifier.4` reaches λ = 0.010 and
  the softmax pushes it back *away* to 0.165. The structural objection is
  answered by the control: the softmax shifts λ by < 0.01 in both scrambled runs
  because it never leaves its affine regime, so squashing is not automatic — it
  requires trained, confident logits.
  - **What survives:** the softmax does contribute 14–20%, and `prob` is
    measured at **96.5% of its own ceiling** (`D_prob ≤ 0.002`, measured
    0.001929). It has not flattened — the top increment is 1.75× the mean of the
    others — but `c = 1` is maximum contrast, so there is no way to check
    further. Prefer `classifier.4`: more log-like, no softmax, no ceiling. This
    is more evidence for item 4 below.
