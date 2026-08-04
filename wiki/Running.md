# Running

Run everything from the **repo root** — `log_response` is a package, invoked as
`python -m log_response.run`, not as a script.

## Install

```bash
pip install numpy matplotlib                     # offline mode only
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision   # real models, CPU
```

Use the CPU wheel index. Plain `pip install torch` from PyPI drags in ~3.4 GB of
CUDA/triton packages that are dead weight on a CPU box (measured: 2.7 GB
`nvidia/` + 691 MB `triton/`).

## Commands

```bash
python -m log_response.test_pipeline                                  # fast check, no weights
python -m log_response.run --model synthetic --reps 12 --figures out/ # offline check
python -m log_response.run --model vgg19 --reps 50 --figures out/ --save runs/vgg19
python -m log_response.run --model vgg19 --reps 50 --scramble         # control
python -m log_response.run --load runs/vgg19 --figures out/           # re-plot, no model
python -m log_response.run --load results/vgg19-r50-s0 --panels out/panels.png
```

`--panels` writes the summary figure: one column per layer, contrast **linear**
on the top row and **log** on the bottom with the per-frequency fits. The two
rows together are the result — on a linear axis every layer looks like the same
saturating curve, and only the log axis separates the late layers (straight)
from the early ones (visibly bent). Frequency is encoded as a single-hue
light→dark ramp, since it is ordered.

`--figures <dir>` writes the per-layer figures **and `lambda_profile.png`**, the
depth profile: λ per tap against the λ = 1 (linear in contrast) and λ = 0 (log
law) reference lines, with the 95% interval as a band and the fit's R² in a row
underneath. That row is not optional decoration — at `prob` on `IMAGENET1K_V1`
the trained and scrambled runs both return λ ≈ 0.167 and only R² separates them,
so the figure never shows one without the other.

Each per-layer figure's right panel scales every frequency to its own range and
overlays its fitted power law against the same two references. **On a log
contrast axis the shapes are inverted from the intuition**: the *log* law is the
straight line and linear-in-contrast bends upward. The axis label says so.

Back-ends: `synthetic` (offline), `data` (raw pixels — the paper's `data` row and
the metric's noise floor, also offline), any `torchvision.models` arch,
`clip:ViT-B-32`, `hf:<model-id>` (generative VLM), `sam[:<model-id>]`.

Both offline back-ends run anywhere in seconds to minutes and need no weights,
so they are the two things worth running first in a new environment:

```bash
python -m log_response.run --model synthetic --reps 12   # pipeline check
python -m log_response.run --model data --reps 50        # noise floor
```

### Flags

| Flag | What it does |
|---|---|
| `--reps N` | Draws per cell, default 250. The cost knob — see the table below. |
| `--layers all` | Every leaf module (45 on VGG-19) instead of the three-tap default. How the depth profile is measured. Works on every back-end — but **not advisable on a VLM**: it hooks 471 modules on SmolVLM including attention internals, whose outputs at ~1 000 tokens are ~92 MB each, and two attempts were killed by the runner OOM 41 s in. Pass an explicit list of block outputs instead. |
| `--frequencies` | Override the 8 spatial frequencies (cycles/image). |
| `--contrasts linear` | Sample the same contrast endpoints evenly instead of geometrically. The grid control; λ should not move, and [measurably does not](Results.md). |
| `--size N` | Input side in pixels, default the back-end's native size (224 for the CNNs, 336–384 for the VLMs). Frequencies are in cycles per *image*, so this does **not** rescale the pattern — it changes how finely the pattern is sampled, and how much work a forward pass is. Comparable across sizes on λ in principle; the caveats are aliasing as the top frequency approaches Nyquist, and whatever the model's own resize does. On `llava-interleave` size buys nothing: 455 s at 384 px vs 462 s at 224 px, because the processor resizes to its own resolution before tiling. |
| `--scramble` | Permute weights within each layer — the control. Pair with `--scramble-seed`, because one permutation is one sample. |
| `--scramble-seed N` | Permutation seed, independent of `--seed`. Set both explicitly: a sweep that varies them together bounds the wrong thing. |
| `--weights PATH` | Local `state_dict`. Required in this sandbox, where weight hosts are blocked. |
| `--device` / `--dtype` | Placement, and the **model's** weight dtype — not the accumulator, which is float64 in every run regardless. Leave it alone on the torchvision back-ends. On `hf:` it is worth setting explicitly so the run records which dtype it used: VLM per-pass cost has been seen to vary ~9× across runs of the same model, and whether dtype or runner variance drives that is [unresolved](Results.md). |
| `--noise-blocks K` | Split the reps K ways round-robin and report the standard error of D from their spread. Costs no extra forward passes, only K extra accumulators. This is what turns "which law fits better" into "does *any* law fit within measurement error" — it adds a `chi2/dof` column. **No committed run has used it yet.** |
| `--unit-taps a,b,c` | Keep the **per-unit** vectors behind `D` and `D_mod` for those taps, plus each unit's gray value, in `result.units.npz`. `D` is an L1 *norm*, so λ says whether the response's magnitude is a power of contrast, not whether the response is — a mean shift growing exactly like `c` while its carriers rotate reads λ = 1. No extra forward passes; the cost is storage, ~3.4 MB per 4096-unit tap in git. A single VGG conv tap would be **1.4 GB**, so `--unit-budget-mb` (default 32) refuses an over-large list before the grid runs, and `all` is never appropriate here. See [Method](Method.md#per-unit-detail-for-the-taps-that-ask-for-it). |
| `--save` / `--save-run` / `--load` | See *Storing a run* below. |
| `--notes "..."` | Prose written into the run directory's `notes.md`. `--save-run` never overwrites an existing `notes.md`. |
| `--allow-random-init` | Deliberate opt-in for an untrained control; stamps `pretrained_verified: false`. |
| `--quiet` | Suppress the per-cell progress output. |

`--instruction`, `--prompts` and `--mask-decoder` apply only to the VLM and SAM
back-ends. Every VLM λ is conditional on the instruction, so it is recorded in
`run.json` both raw and chat-templated — though the prompt turns out not to
reach the representation at all, only the final-position readout
([why](Results.md)).

## Cost and wait times

The grid is 14 contrasts × 8 frequencies × `--reps` forward passes, batch-1.
Per-image cost is the whole driver loop — grating synthesis, the forward, and
the float64 accumulate over every tapped unit — not the forward alone. VGG-19,
224², default taps, measured on both paths:

| Path | s / forward | `--reps 50` (5 600) | full `--reps 250` (28 000) |
|---|---|---|---|
| This sandbox, 4 cores | 0.170 | ~16 min | ~1 h 20 min |
| GitHub-hosted runner   | 0.125–0.250 | 12–23 min | **59–118 min** (measured) |

**Budget for the top of that range, not the middle.** Four measured jobs, all
the same 28 000-pass grid on the same code at 4 cores, took 58.6, 69.7, 116.5
and 118.4 min — a 2.0× spread with no code difference (`run_experiment` is
byte-identical across the two runs). Runner speed varies that much, so a job
sized against the fast end can miss the 6 h cap. `--reps 1000` would be ~8 h at
the slow end and is not viable on a standard runner at all.

With `--layers all`, measured on the runner (4 cores, all eight jobs):

| Grid | measured | spread |
|---|---|---|
| 45 taps, `--reps 250` | 110.6 / 131.5 / 133.1 / 137.3 min | 1.24× |
| 45 taps, `--reps 50` | 26.0 / 30.6 / 30.8 / 30.9 min | 1.19× |

**Prefer going down in reps, not up.** The r50 runs exist to separate a real
response from the metric's noise floor, and for that `--reps 50` against a
committed `--reps 250` gives √5 = 2.24 discrimination for a *fifth* of the cost,
where `--reps 1000` would give √4 = 2.00 for four times it — 20× worse per unit
of answer, and over the cap besides. More reps buys a better estimate; fewer
reps buys the comparison.

The r50 numbers are the first runs carrying both orderings of the metric. They
came in at 26–31 min against the ~26 expected from scaling the r250 grid by 1/5,
so the second accumulator costs at most ~15% and is inside runner variance.

Other back-ends, scaled from the same sandbox loop: `alexnet` 0.016 s/forward,
`resnet50` 0.071, `vit_b_16` 0.157, `resnet152` 0.183. `synthetic --reps 12`
takes seconds. The `hf:` VLM and `sam:` back-ends are the only ones that really
want a GPU — a 7B model at 28 000 passes, and SAM's native 1024² input.

`--layers all` taps every leaf module — 43 for VGG-19, both sides of every
nonlinearity, Dropout skipped since it is the identity in eval mode. It costs
**1.65×** (85 min against 51 for the full grid), all of it in the float64
accumulate over 31M units per image rather than in the forward pass. The saved
surface is still only ~41 KB.

Two things it does that VGG-19 never made visible, because VGG-19 gives every
ReLU its own module and calls each module once:

- **A module called more than once per forward gets a slot per firing** —
  `<name>`, `<name>@2`, `<name>@3`. torchvision's ResNet is the case: each
  `BasicBlock`/`Bottleneck` holds a *single* `nn.ReLU` and calls it 2–3× (after
  conv1, after conv2, after the residual add). Before this, one name kept only
  the last firing and the rest were dropped — on `resnet50`, **32 of 158
  activations**, with the surviving taps mislabelled.
- **A hooked module that never fires is reported**, as a `RuntimeWarning` naming
  the taps. `nn.MultiheadAttention` passes `out_proj.weight`/`bias` to
  `F.multi_head_attention_forward` instead of calling the module, so
  `--layers all` on `vit_b_16` registers 75 modules and **12 never fire**. The
  run is correct — those taps do not exist — but a depth profile missing every
  attention output projection should not be discovered afterwards.

Measured tap counts and per-image accumulate load, for sizing a new run
(224², `--layers all`, random init — the shape is weight-independent):

| Arch | leaf modules | taps incl. `logits`/`prob` | reuse slots | units/image |
|---|---|---|---|---|
| `alexnet` | 19 | 21 | 0 | 1.1 M |
| `vgg19` | 43 | **45** | 0 | 31.3 M |
| `vgg19_bn` | 59 | 61 | 0 | 46.1 M |
| `resnet50` | 126 | 160 | **32** | 32.0 M |
| `vit_b_16` | 75 | 65 | 0 (12 unfired) | 20.3 M |
| `convnext_tiny` | 156 | 158 | 0 | 31.4 M |

VGG-19's 45 is unchanged, which is the point — every committed depth profile is
a 45-tap VGG-19 run and `test_vgg_all_layers_is_unchanged_by_the_reuse_fix`
pins it.

That 1.65× is reducible to ~1.10× by accumulating into torch tensors in place
instead of `numpy` with a per-layer `.astype(np.float64)`, which currently
copies 31M values three times per image. It is deliberately not done: the
committed runs were accumulated in numpy float64 and the point of the wide
sweep is comparability with them, so correctness outranks the 30 min. Measure
before adopting it — averaging 250 float32 activations is exactly where a
narrower accumulator would bite.

Cost is linear in `reps × contrasts × frequencies`, so `--reps` and
`--frequencies` scale it directly. Reps only average down the orientation/phase
nuisance within a cell, at `1/√reps`; the 14-point contrast axis the fit runs
along is untouched. `--reps 50` explores, `--reps 250` is the reported grid.

Anything past a few minutes should be backgrounded and waited on rather than
watched — progress prints every ~11 cells:

```bash
python -m log_response.run --model vgg19 --reps 50 --save runs/vgg19 > run.log 2>&1 &
until [ -f runs/vgg19.json ]; do sleep 5; done; tail -20 run.log
```

## Storing a run

**The `D(freq, contrast)` surfaces are the expensive product; everything else
re-derives from them.** A surface is `n_layers × n_freq × n_contrast` float64 —
a few KB whatever `--reps` was, so even a 28 000-forward run costs ~7 KB. Never
throw one away, and never recompute a fit you could re-load.

Since 2026-07-27 the npz carries **two** such arrays: `surfaces` (the paper's
distance-of-means `D`, the headline everywhere) and `mean_of_distances` (`D_mod`,
the other order of operations). Both are re-fitted on load, and `report()` gains
`lam(mod)` / `R^2(mod)` columns. Runs saved before that date have only the first
and load with `mean_of_distances` set to `None` — nothing else changes, and no
committed surface moved when it was added. Why both:
[Method](Method.md#the-other-ordering-recorded-alongside).

`--unit-taps` adds a **fifth** file, `result.units.npz`, and it is the one
exception to "a run directory is a few KB": 3.4 MB per 4096-unit tap, because it
stores the per-unit vectors rather than their mean. It sits beside `result.npz`
instead of inside it precisely so that `--load`, the re-fits and the test that
walks every committed run do not pay for it. Runs without it are unchanged.

`--save <base>` writes `<base>.npz` (the surfaces) plus `<base>.json` (the fit
summary). Pass it on any long run.

`--save-run results/<slug>` is what to use for a run worth keeping. It writes
the committable four-file directory — `result.npz`, `result.json`, `run.json`
(provenance), `notes.md` — described in [results/](../results/README.md), which
is also where the slug convention and the "never commit figures" rule live.

`--load <dir-or-base>` re-fits and re-plots from the stored surfaces with no
model, no weights and no network. This is how every number in
[Results](Results.md) survived two changes of metric without a single re-run:

```bash
python -m log_response.run --load results/vgg19-r250-s0-alllayers-fixed-caffe
python -m log_response.run --load results/vgg19-r250-s0 --figures out/
```

Two things worth knowing before you overwrite anything: `--save-run` refuses to
start rather than clobber an existing directory, and it never overwrites an
existing `notes.md` — so a careless re-run would otherwise leave prose describing
numbers that had changed underneath it.

## Trusting a run

The log response only exists in a *trained* net, so a run whose weights failed to
load measures nothing — and used to be indistinguishable from a real one. Two
guards now make that state unreachable:

- Failing to load pretrained weights **raises** rather than falling back to
  random init. Use `--allow-random-init` to measure an untrained control
  deliberately.
- `--save`/`--save-run` **refuse** to persist a run that is not verifiably
  pretrained, unless that flag was passed. Saved runs record
  `weights.pretrained_verified` (`true` / `false` / `null` for weight-free
  back-ends) alongside the commit, package versions, and the weight file's
  sha256.

## This sandbox: weights are the blocker

`download.pytorch.org` and `huggingface.co` are **blocked** by the network
policy, and PyPI is not. So `--model vgg19` cannot fetch weights here and now
exits with an error (before the guard described above, it silently fell back to
random init and reported meaningless numbers). Pass `--weights` with a local
`state_dict`.

The original Oxford VGG-19 ImageNet weights are mirrored on
`storage.googleapis.com` (reachable) in Keras HDF5 form, so one command produces
a usable checkpoint — this is what `results/vgg19-r50-s0` ran on, and it works on
a runner too:

```bash
pip install h5py
python -m log_response.convert_weights --out vgg19_caffe.pth --verify
python -m log_response.run --model vgg19 --weights vgg19_caffe.pth --reps 50
```

It bridges four differences: kernel layout `(kh,kw,in,out)→(out,in,kh,kw)`,
conv1 input channels BGR→RGB, the fc1 flatten `(H,W,C)→(C,H,W)`, and caffe
preprocessing (`x*255 − mean`, no std division) folded into conv1 so the net
takes the repo's normalised RGB input.

`--verify` checks that last one against a directly-computed caffe path and fails
above 1e-5; it measures **2.9e-8**. That check matters more than it looks: a
*gain* error in the fold would rescale a grating's effective contrast and slide
the contrast-response curve along its own axis, which would masquerade as a
difference between checkpoints. conv1 is the only layer touching the input, so
its exactness rules that out everywhere. The arithmetic also has an offline test
(`test_preprocessing_fold_is_exact`).

Other reachable hosts: `pypi.org`, `github.com`, `raw.githubusercontent.com`,
`storage.googleapis.com`. Check policy state with
`curl -sS "$HTTPS_PROXY/__agentproxy/status"`.

## GitHub-hosted runners

The second run path, and the better one for a real grid: **weights download
fine** (no egress restriction), so no conversion is needed — `--model vgg19`
works directly, which is the whole reason to prefer it over the sandbox.

[`.github/workflows/log-response.yml`](../.github/workflows/log-response.yml)
is the entry point. Actions → *log-response run* → **Run workflow**; by default
it launches the pretrained run and its scrambled control as two parallel jobs,
prints each fit table on the run summary page, and **commits `results/<slug>/`
back to the ref it was launched from** — the artifact is the fallback copy, not
the delivery mechanism, since the sandbox cannot reach the artifact blob host.
Nothing to download: `git pull` and the run is there.

The commit-back runs on every event, so a dispatch from `master` commits to
`master`. Both matrix jobs push to the same ref minutes apart; the step rebases
and retries up to five times, which is how the two r250 directories landed as
consecutive commits.

**Dispatch inputs.** `model`, `reps`, `seed`, `frequencies`, `contrasts`,
`layers`, `size`, `dtype`, `instruction`, `notes` map straight onto the CLI
flags above — the last three were added 2026-07-31, having existed in
`run.py` from the start with no way to reach them from a dispatch. The rest
are about where the run lands and which weights it uses:

| Input | What it does |
|---|---|
| `weights` | `canonical` = torchvision `IMAGENET1K_V1`; `caffe` = the original Oxford VGG-19, fetched and converted on the runner with `--verify` (fails above 1e-5, so the fold is re-established per run rather than inherited). **The paper used the Caffe one** — see [Results](Results.md#which-checkpoint-the-paper-used-and-what-reproduces-on-it). |
| `variants` | `both` (default) launches the pretrained run and its scrambled control in parallel; `pretrained` / `scrambled` for one. |
| `scramble_seed` | Permutation seed, held separate from `seed`. Sweep this to isolate permutation variance — sweeping `seed` moves the image draws too. Enters the slug as `-p<n>`, so a sweep cannot collide. |
| `slug_suffix` | Appended to the computed slug. **Needed whenever weight lineage differs**, because the slug encodes model/variant/reps/seed but not which checkpoint produced it. |

The slug-collision check runs before the measurement, so a name clash costs
seconds rather than the two hours the grid takes — and it refuses rather than
overwriting, because `--save-run` preserves an existing `notes.md` and would
otherwise leave prose describing numbers that had changed underneath it.

Facts worth knowing, measured rather than assumed (the workflow's *Runner facts*
step prints them on every run):

- **4 cores, 16 GB** on standard `ubuntu-latest` — that is the *public*
  repository allocation, and this repo is public. Private repos get 2 cores and
  are roughly twice as slow. Standard runners are free and unmetered on public
  repos.
- **~14 GB disk.** Install the CPU torch wheel; the CUDA packages alone would eat
  a fifth of it.
- **6 h job limit.** The full `--reps 250` grid takes 58 min, so it fits with
  ~5 h of headroom — shrink `--reps` rather than sharding `--frequencies`, since
  nothing merges per-frequency shards back into one surface.
- Cache both pip and the torch hub weights (`~/.cache/torch`) — the VGG-19
  download is ~550 MB per run otherwise.
- `--load` re-fits and re-plots from the artifact's `result.npz` with neither
  torch nor weights, so iterate on the analysis locally and for free.

Provenance separates the two paths after the fact: `run.json` records
`environment.platform`, `cpu_count`, `wall_seconds` and `weights.source`, so a
runner result (`torchvision vgg19 IMAGENET1K_V1`) is distinguishable from a
sandbox one (`local state_dict: ...`) without relying on memory.

### When a runner is not enough

Only the GPU back-ends need more than the above. Free options, in order of
how well they suit an unattended grid:

- **Kaggle notebooks** — ~30 GPU-h/week (T4 ×2 or P100), 12 h sessions, and
  *Save & Run All* executes headless, so nothing depends on a live tab. The
  right home for `hf:` and `sam:` runs. Write to `/kaggle/working/`.
- **Colab** — free T4 under a dynamic quota, but the ~90 min idle timeout keys
  on browser-tab interaction rather than on whether the code is running. Fine
  for poking at a short `--reps`, unreliable for a full grid.
- **Oracle Cloud Always Free** — no wall-clock cap at all, so it suits "slow
  CPU, just let it grind"; the free Ampere A1 allowance was halved to 2 OCPU /
  12 GB in June 2026 and capacity is scarce in busy regions.
