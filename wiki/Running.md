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
python -m log_response.test_pipeline                                  # 14 tests, no weights
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
Measured on 4 CPU cores, VGG-19 at ~0.10 s/forward:

| Run | Forwards | Wall time (4 cores) |
|---|---|---|
| `--model synthetic --reps 12` | 1 344 | seconds |
| `--model vgg19 --reps 50` | 5 600 | ~6 min |
| `--model vgg19 --reps 50 --scramble` | 5 600 | ~6 min |
| `--model vgg19` (full, `--reps 250`) | 28 000 | ~30 min (not yet run) |

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

Different trade-offs from the sandbox — **weights download fine** (no egress
restriction), so no conversion is needed; use `--model vgg19` directly. The
limits are compute and disk:

- **2 cores** on standard `ubuntu-latest` (4 on larger runners) — roughly double
  the wall times in the table above. Budget ~15 min for `--reps 50`.
- **~14 GB disk.** Install the CPU torch wheel; the CUDA packages alone would eat
  a fifth of it.
- **6 h job limit.** A full `--reps 250` grid at ~1–2 h on 2 cores fits, but keep
  headroom; prefer sharding over `--frequencies` across matrix jobs.
- Cache both pip and the torch hub weights (`~/.cache/torch`) — the VGG-19
  download is ~550 MB per run otherwise.
- Upload `runs/*.npz` as an artifact; re-fit and re-plot later with `--load`,
  which needs neither torch nor the weights.
