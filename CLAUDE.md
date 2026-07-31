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

## One module can be several taps — this bites ResNet, not VGG

Same failure family as the above, found 2026-07-27 while opening the
cross-architecture work, and **no committed run is affected**. A forward hook
fires once per *call*, not once per module, and torchvision's ResNet holds a
single `nn.ReLU` per block that it calls 2–3× (`BasicBlock` after conv1 and
after the residual add; `Bottleneck` three times). The hook assigned to
`self._acts[name]`, so one name kept only the **last** firing and the earlier
activations were dropped — on `resnet50`, 32 of 158 activations gone and the
surviving taps mislabelled, exactly the pre-activation bug's signature.

Every firing after the first now gets its own slot: `<name>`, `<name>@2`,
`<name>@3`. VGG-19 gives each ReLU its own module and calls it once, so its
45-tap set is byte-for-byte what it was — pinned by
`test_vgg_all_layers_is_unchanged_by_the_reuse_fix`, and
`test_reused_modules_are_not_collapsed_into_one_tap` fails without the fix.

Two neighbours, same theme — a tap's name has to match what it recorded:

- **A hooked module that never fires now warns** instead of vanishing.
  `nn.MultiheadAttention` hands `out_proj.weight`/`bias` to
  `F.multi_head_attention_forward` rather than calling the module, so
  `--layers all` on `vit_b_16` registers 75 modules and 12 produce no tap.
- **`--layers all` works on every back-end now.** The expansion lived inside
  `TorchvisionModel`, so `--layers all` on `clip:`/`hf:`/`sam:` raised
  `KeyError: layer 'all' not found` — the depth profile, which is the figure a
  result is read off, could not be measured on any of them. On CLIP it stops at
  the `visual` tower and on SAM at `vision_encoder` unless `--mask-decoder`,
  since the rest does not run per grating.

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
3. ~~**Re-run the seed sweep with `--scramble-seed` fixed against `--seed`.**~~
   **Done, on the Caffe checkpoint — and 0.60 is out of reach.** Four
   permutations at `--seed 0` (identical images, permutation the only variable)
   give `prob` mean R² **0.428 / 0.516 / 0.443 / 0.422** — spread 0.095, sd
   0.044. The paper's 0.60 is **outside** that range, and outside the
   `IMAGENET1K_V1` range (0.693–0.863) too: the two checkpoints miss it in
   opposite directions and 0.60 sits in the gap between them. So the
   disagreement is real rather than a one-permutation accident, which is
   precisely what a single value could not establish. It stays stated per rule 4.
   The paper's *direction* survives and strengthens: trained 0.976 against
   0.42–0.52 is a gap of 0.46–0.55, against the 0.38 the documented pair implies.
   The workflow gained a `scramble_seed` input (slug `-p<n>`) to make this
   possible.
4. ~~**Reconcile `wiki/Method.md` with the measured grid.**~~ **Done — the table
   was right and the checkpoint was wrong.** The paper (§8.1) ran MatConvNet's
   *imported pre-trained original* VGG-19, i.e. the Oxford/Caffe weights, not
   torchvision's `IMAGENET1K_V1`. On the Caffe run `prob` = **0.980** against
   the documented 0.98, `prob` **is** the peak of all 45 taps, and fc7 sits at
   0.750 ("much lower", as documented). Nothing was edited to match: the
   disagreement was an artifact of testing the claim on a checkpoint the paper
   never used. The scrambled control (0.429 vs 0.60) stays open per rule 4.

5. ~~**Run the per-tap reps sweep.**~~ **Done — and the conv stack is real.**
   Four `--reps 50` 45-tap runs against the committed r250 ones. **Only
   `features.0` is on the floor** (D(50)/D(250) = 2.22–2.24 against √5 = 2.236,
   98–100% noise), in all four runs; outside `features.0/1/2` the largest noise
   fraction anywhere is **3.4%**. So Caffe's flat λ ≈ 1 conv stack is measuring
   a locally-linear response, not empty taps — the competing explanation below
   is excluded. Went *down* in reps rather than up: √5 discrimination for a
   fifth of the r250 cost, where `--reps 1000` would have given √4 for 4×.

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
  - **It works as a per-tap diagnostic, from one run.** Across the 180 tap-runs
    of the four r50 45-tap runs, median |λ − λ_mod| is **0.039** where the noise
    fraction is under 5% (n=171) and **0.277** where it is over (n=9) — and all
    9 are `features.0/1/2`. Sharpest case: `features.1` on trained IN1K reads
    λ **+1.67** against λ_mod **+1.01**, i.e. the primary metric reports a
    strongly supralinear exponent that is its own sampling noise.
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
- **On the paper's checkpoint the control is bounded away from 0.60.** Four
  Caffe permutations at fixed `--seed` give 0.428 / 0.516 / 0.443 / 0.422
  (spread 0.095). `IMAGENET1K_V1` gives 0.693–0.863. **Both miss the documented
  0.60, in opposite directions**, so it is a genuine disagreement, not sampling.
  λ at `prob` is +1.76 to +3.00 in all four, so scrambled Caffe's supralinear
  classifier is robust and not a seed-0 artifact. Caveat that repeats: permutation
  p1 peaks at `features.0`, i.e. no tap beats the noise floor — single-seed
  control *shapes* have now misled twice.
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
  - **AlexNet reproduces it (2026-07-28): −0.218 / +0.216, 4/5 and 0/2.** So it
    is not VGG-specific either. Both nets that show it carry **torchvision**
    weights and the one that does not carries the original Oxford/Caffe
    weights — the sawtooth tracks the training recipe, not the architecture.
    Three runs, so this is a correlation, not a mechanism.

- **The scrambling control breaks on nets with running statistics** (2026-07-28,
  sharpened the same day). It permutes every `*weight*` tensor, which on a
  BatchNorm net permutes γ across channels while `running_mean`/`running_var`
  stay put — each channel then gets one channel's statistics and another's
  scale. That **decalibrates** rather than degrades, saturating the softmax to
  one-hot:

  | scrambled | running stats? | r(logits, prob) | D_prob/D_logits |
  |---|---|---|---|
  | VGG-19 (either ckpt), AlexNet | no | 1.000000 | 1/1000 |
  | **ViT-B/16** (LayerNorm) | **no** | 0.999975 | 1/1000 |
  | **`vgg19_bn`** | **yes** | **0.162** | **1.1e-10** |
  | **`resnet50`** | **yes** | **0.673** | **1.7e-10** |

  The split is **not** "normalised vs not" — ViT is normalised throughout and
  scrambles cleanly, because LayerNorm has no buffers to desynchronise. The BN
  pair's numbers (`vgg19_bn` 0.214, `resnet50` 0.658) are **not comparable** to
  VGG-19's 0.429, AlexNet's 0.865 or ViT's 0.797; both r50 companions return
  most or all of the search range as their λ interval. Fix, if a BN control is
  wanted: scramble the running statistics with the weights, or leave both alone.

- **The metric's floor is a property of affineness, not of depth-one** — and
  BatchNorm counts while LayerNorm does not. VGG-19 put only `features.0` on it,
  which made "the first conv" a tempting shorthand. `vgg19_bn` puts `features.1`
  there too (a **BatchNorm layer at 99.3% noise**) and `resnet50` puts `bn1`
  there, because BN in eval uses fixed running statistics and is therefore
  affine in the input. **ViT-B/16 stops at `conv_proj`**: LayerNorm normalises
  by the input's own mean and variance, so it is not affine and has no
  zero-population floor. Predicting otherwise for ViT was wrong; the rule is
  affineness, exactly.
- **Why λ ≈ 1 survives 33 layers — the live hypothesis, now constrained.** The
  grating is a *perturbation* `gray + c·g` about a fixed operating point, and a
  ReLU net is piecewise linear, so while the perturbation does not flip ReLU
  gates, `D = |J·(c·g)| = c·|J·g|` exactly — linear in contrast at any depth. On
  this reading λ < 1 is the signature of gates actually switching with contrast,
  and the log response is what emerges once they do. Caffe stays in that regime
  for the whole conv stack; `IMAGENET1K_V1` leaves it gradually from mid-stack.
  The direct check is still to count ReLU sign flips between gray and grating
  against `c`; that needs a forward pass, so it is an Actions job.
  - **"Rectifications carry it" is now insufficient — measured 2026-07-28.**
    `vgg19_bn` has VGG-19's topology, ReLU count, task and stimulus, and
    BatchNorm in eval is a per-channel **affine** map that cannot add gates. Its
    conv stack nonetheless sits at **λ = −0.071** (R² 0.971) against Caffe's
    **+1.06** and IN1K's +0.69 — a shift of ~1.1, larger than the entire Caffe
    depth profile, with zero rectifications added. So the controlling variable
    is the **operating point** units sit at relative to their ReLU, not the
    number of rectifiers passed. The perturbation reading survives; the
    "rectifications, not depth" phrasing does not.
  - **Depth was already ruled out independently**: AlexNet, 8 weight layers,
    reaches `prob` λ **+0.053** at R² 0.985 and peaks at `prob` over all 21
    taps.
  - **And rectifiers are not needed at all — ViT-B/16, 2026-07-28.** No ReLU
    anywhere (GELU is smooth, no hard gate), and λ still runs **+0.926** at
    `conv_proj` to **−0.617** mid-encoder, −0.162 at `prob` (R² 0.933). Gate
    flipping cannot be the mechanism because there are no gates. What survives
    across all **22** architectures now measured is the **operating point**
    reading.
  - **Skip connections do not preserve linearity either.** ResNet-50 was run to
    test whether the identity path keeps an affine component alive deep — it
    does not: `layer3`/`layer4` median λ **−0.262** with **0/85** taps within
    0.15 of λ = 1, and no deep tap on the floor (max 5.1% noise outside the
    first three modules).
  - ~~**A competing explanation has to be excluded first**~~ — **excluded.** An
    empty tap reads the same λ ≈ 1 at high fit quality, and 26 of the 37 Caffe
    conv taps sit within 0.15 of the noise-floor λ. The reps sweep (item 5) says
    they are real: only `features.0` falls with repetition count, everything
    else holds `D` to within 3.4%. So gate-flip counting is now testing a
    hypothesis with no live rival.
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
- **λ moves more across frequency than across architecture** (2026-07-28). Every
  λ in the docs is a **median over the 8 frequencies**. Per-frequency at `prob`:
  median λ spans **0.43** across the six trained series, while *within* a run λ
  spans **0.49** (Caffe) to **1.75** (`vgg19_bn`). Four dip in the mid band
  (7–28 cyc/img) and two peak there. **Seed-swept, 24 runs, 3 seeds each**:
  every series is sign-consistent and the weakest effect is **9× its own
  seed-to-seed sd** — `vgg19_bn` +0.533±0.036, AlexNet +0.409±0.021, ResNet-50
  +0.404±0.014, IN1K +0.198±0.022, Caffe **−0.335**±0.025, ViT **−0.284**±0.022.
  **It is not the training-recipe split**: ViT carries torchvision
  `IMAGENET1K_V1` like the four that dip. Per-frequency λ is now in
  `result.json` under `per_frequency[].lambda`. See `wiki/Results.md`.
  - **Corrected the same day.** Committed first as "four resolved dips plus two
    nulls", on the 95% profile-F intervals alone. Both "nulls" reproduce
    cleanly; the interval was simply the wrong test. **For "is this frequency
    profile real", prefer seed replication** — per-frequency, the mean CI
    half-width exceeds the across-seed sd by **2.9× to 7.1×**. The interval
    answers a different question: how well 14 contrast points pin λ in one run.
  - **The dip does not generalise — 17 more architectures, 2026-07-29.** A
    contributed 23-run screen gives **5 monotone** in frequency (no band shape
    at all: ConvNeXt-T/S, Swin-T, Swin-V2-T, GoogLeNet, ρ −0.76 to −0.90),
    9 dipping, 3 peaking, band contrast −1.53 to +0.62. The clean mid-band dip
    belongs to the six seed-swept series, **not to ImageNet-trained nets
    generally**. One seed each.
  - **But frequency structure needs trained weights, and now has a null.** The
    screen's six scrambled controls are all on BN-free nets, so all six are
    valid. λ's range across frequency is **3.4×–20.8× larger trained than
    scrambled, 6/6 pairs**; scrambled is nearly flat (0.07–0.26) against
    trained 0.89–2.62. The *shape* is architecture-dependent; the *existence*
    of the structure is a property of learning.
- **The compression needs neither convolution nor attention** (2026-07-29,
  `timm:` back-end). gMLP-S16 and ResMLP-12 are token/channel-mixing **MLPs** on
  a patch embedding — no conv past the stem, no attention anywhere — and reach
  `prob` λ **−0.250** (R² 0.934) and **−0.315** (R² 0.892). With ViT (no
  rectifiers) and AlexNet (no depth), the operating-point reading is the only
  survivor. PoolFormer-S12 −0.034; **FocalNet-T +0.420, the highest λ of all 29
  combinations**; XCiT-nano −0.469 but at **λ-R² 0.745, the worst fit in the
  repo** — quote it with the R² or not at all. One seed each.
- **And it does not need the classification objective** (2026-07-30). Every
  other run here is an ImageNet classifier with a 1000-way softmax.
  SmolVLM-256M is **generative** — language-modelling objective, SigLIP tower,
  no classification head — and its three hidden taps sit at the log law with
  the scrambled control **cleanly separated** (non-overlapping 95% intervals at
  all three): `vision…layers.11` **+0.047** vs +0.560, `text_model.layers.14`
  **+0.020** vs +0.524, `text_model.layers.29` **−0.120** vs +0.549, trained
  λ-R² 0.955–0.965. Every trained interval contains 0; no scrambled one does.
  Also a **7th matched pair** for frequency structure needing trained weights
  (trained λ range 0.89–1.03 across frequency vs scrambled 0.22–0.27, ratio
  3.6–4.5×, inside the classifiers' 3.4–20.8×). Control valid by the
  renormalisation rule: 0 BN, 25 LayerNorm + 61 RMSNorm, no pinned taps,
  max D_logits 2.93. **reps = 2**, one seed, one instruction — heavy caveats.
  - **Do not quote this run's `prob`.** λ +0.485 at λ-R² 0.857, interval
    overlapping the control's, per-frequency λ from +0.03 to **+2.77** — noise
    at 2 reps over a 49 280-way softmax. `D_prob` is 7.6% of its 2/V ceiling
    trained against **92.3%** scrambled, a saturated language head rather than
    a response. On a VLM, preferring a pre-softmax tap is mandatory, not merely
    better.
  - **`prob`'s bound is 2/V, not 2/1000.** `HFVLMModel` now records
    `vocab_size`, the chat-templated `conditioning_text` and a normalisation
    census, so a VLM `D` is interpretable from its own directory. The
    instruction was already recorded (`run.py`); the vocabulary was not.
- **VLM forward-pass cost varies ~9× run to run, and the cause is not
  established** (2026-07-30). Four measurements of the same model, SmolVLM-256M:

  | | dtype | grid | s/pass |
  |---|---|---|---|
  | probe | bf16 | 6 cells | 14.6 |
  | full run (cancelled at the cap) | bf16 | 8×14 | **130** |
  | probe | fp32 | 14 cells | 31.1 |
  | full run (landed) | fp32 | 8×14 | **14.9** |

  **Corrected 2026-07-30**, having been committed earlier the same day as
  "a checkpoint's own dtype is not the fast one — bf16 cost 4.2×". That number
  divided the bf16 *full run* by the fp32 *probe* — different grids **and**
  different runners, so it isolated nothing. The within-dtype spread is **8.9×**
  for bf16 and **2.1×** for fp32; the only grid-matched pair is 130 vs 14.9, one
  run each on different runners. bf16 emulation without AVX512-BF16 remains a
  plausible contributor, but runner variance alone spans enough to produce the
  whole gap. Per rule 4, stated as unresolved. Isolating it needs the two dtypes
  on the same runner, which the workflow cannot currently pin.
  - `llava-interleave` is separately unfixable on CPU — 62.8 s/pass from anyres
    tiling, and **grating size does not change it** (455 s at 384 px vs 462 s at
    224 px; the processor resizes to its own resolution before tiling). That
    one *is* clean: same runner class, same 6-cell grid, size the only variable.
  - **Size a long run from a probe that crosses a progress line.** A 7-pass
    probe read 14.6 s/pass where the real grid ran at 130 — a **9×** error that
    no safety factor would have absorbed. The cancelled run's own log (two
    11-cell intervals, 119.6 and 117.0 min) was the trustworthy measurement.
    Job-status green is not progress: the cell counter is. This survives the
    correction above intact, and is the more useful lesson.
  - The workflow gained `size`, `dtype` and `instruction` inputs — `run.py` had
    all three flags, no dispatch could reach them.
- **"No BatchNorm" is not sufficient for a valid scramble** (2026-07-29,
  corrects the rule below). `resmlp-12-scramble` has **zero** BN modules and is
  broken anyway: max |D_logits| **2409** against ~2 elsewhere, r(logits,prob)
  **0.590**, and **42/114 taps pinned at the λ=+4 bound** (trained companion:
  0). ResMLP uses a learned per-channel **Affine** instead of LayerNorm, which
  never renormalises by the input, so nothing absorbs the permuted scales. The
  real rule is **renormalisation**: the scramble is safe only where the net
  rescales by statistics of the *current input*. LayerNorm yes; BatchNorm-eval
  no (fixed buffers); learned Affine no (fixed learned scale). `TimmModel`
  refuses on BN, which is right but would not have caught this.
- **Preprocessing moves per-frequency λ far more than it moves the median**
  (2026-07-29). `gmlp-s16` run twice, differing only in input normalisation
  (shared ImageNet 0.485/0.229 vs native 0.5/0.5): `prob` λ moves just **0.096**
  but per-frequency **mean |Δλ| = 0.320, max 0.707**, signs disagree at some
  frequencies, and band contrast goes −0.71 → −0.05. That is comparable to the
  mid-band dips (0.20–0.53) and ~10× their seed-to-seed sd. The seed sweep shows
  the dips survive **redrawing the images**; nothing yet shows they survive **the
  normalisation**. All seed-swept series used shared constants, so none of them
  is invalidated. One architecture, one comparison — a flag, not a result.
- **Some taps are not functions of the input at all** (2026-07-29, found while
  plotting depth profiles). Swin-V2-T's `features.*.attn.cpb_mlp.*` — the
  continuous position-bias MLP — takes *relative-position coordinates*, not the
  image, so its `D` surface is **exactly zero at every cell**. The fitter
  answers a constant: λ = −2.95 (near the bound) with `lambda_r2` = **nan**.
  **36 of Swin-V2-T's 136 taps (26%) are these**, in both the trained and the
  scrambled run. They are worse than the noise floor — that is a real layer
  measured at zero SNR, this is a module the image never reaches. Filter on
  `lambda_r2` being non-finite before any depth profile or cross-tap average.
  No other committed run has one.
- **λ pinned at a search bound is not a measurement** (2026-07-29). `fit.py`
  searches λ over [−3, 4]; a fit that hits the edge returns the bound as its
  point estimate, and it reads like a value. `regnet_y_400mf` at 7 cyc/img and
  `mobilenet_v3_large` at 75 both return exactly −3.000. **Drop such cells
  before averaging**: RegNet-Y's band contrast is **+1.10 with the pinned cell
  and +0.05 without** — the whole effect was the bound. This is the
  point-estimate analogue of the interval-spans-the-range case, which the code
  already reports honestly; the point estimate does not.
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
