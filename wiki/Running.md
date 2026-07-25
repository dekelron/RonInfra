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

Back-ends: `synthetic` (offline), any `torchvision.models` arch, `clip:ViT-B-32`,
`hf:<model-id>` (generative VLM), `sam[:<model-id>]`. Useful flags: `--reps`
(draws per cell, default 250), `--frequencies`, `--layers`, `--device`,
`--weights` (local `state_dict`), `--save`/`--load`.

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

Other back-ends, scaled from the same sandbox loop: `alexnet` 0.016 s/forward,
`resnet50` 0.071, `vit_b_16` 0.157, `resnet152` 0.183. `synthetic --reps 12`
takes seconds. The `hf:` VLM and `sam:` back-ends are the only ones that really
want a GPU — a 7B model at 28 000 passes, and SAM's native 1024² input.

`--layers all` taps every leaf module — 43 for VGG-19, both sides of every
nonlinearity, Dropout skipped since it is the identity in eval mode. It costs
**1.65×** (85 min against 51 for the full grid), all of it in the float64
accumulate over 31M units per image rather than in the forward pass. The saved
surface is still only ~41 KB.

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

`--save` writes `<base>.npz` (the expensive surfaces) plus `<base>.json` (fit
summary). Always pass it on a long run so a re-fit or re-plot never recomputes.
For a run worth keeping, use `--save-run results/<slug>` instead: it writes the
committable directory layout described in [results/](../results/README.md), with
full provenance in `run.json`.

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

Workaround used for the run in [Results](Results.md): the original Oxford VGG-19
ImageNet weights are mirrored on `storage.googleapis.com` (reachable) in Keras
HDF5 form, and convert to a torchvision `state_dict` by transposing kernels
`(kh,kw,in,out)→(out,in,kh,kw)`, swapping conv1 input channels BGR→RGB,
reordering the fc1 flatten from `(H,W,C)` to `(C,H,W)`, and folding caffe
preprocessing into conv1 so the net accepts the repo's normalised RGB input.
Verify any such conversion before trusting it — classifying a known image is
enough; a layout or channel-order error yields obvious garbage.

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
