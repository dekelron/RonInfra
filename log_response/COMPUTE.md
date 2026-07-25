# Where to run this for free

The headline measurement is **CPU-sized**. A full 14 contrasts × 8 frequencies
× 250 reps grid on VGG-19 — 28,001 forward passes — takes **≈ 1.3 h on 4 CPU
cores**, no GPU. That fits inside every major free tier, unattended.

## Measured cost

Wall-clock per image for the real driver loop (grating synthesis + batch-1
forward + the float64 accumulate `experiment.py` does), Intel Xeon @ 2.8 GHz,
torch 2.13 CPU build, 224×224, default layer taps. Timing is weight-independent,
so these hold for pretrained and `--scramble` runs alike.

| arch | ms / image (4 threads) | full grid (28k passes) | `--reps 50` | `--reps 25` |
|---|---|---|---|---|
| `alexnet`   |  16 | **0.12 h** | 2 min  | 1 min  |
| `resnet50`  |  71 | **0.55 h** | 7 min  | 3 min  |
| `vit_b_16`  | 157 | **1.22 h** | 15 min | 7 min  |
| `vgg19`     | 170 | **1.33 h** | 16 min | 8 min  |
| `resnet152` | 183 | **1.42 h** | 17 min | 9 min  |

On 2 threads instead of 4, `vgg19` goes to 270 ms/image → **2.1 h** for the full
grid. So the complete headline result — the pretrained run *plus* its
`--scramble` control — is **under 3 h of CPU** even on the smallest free runner.

The cost is linear in `reps × contrasts × frequencies`, so `--reps` and
`--frequencies` scale it directly. Reps only average down the nuisance
(orientation/phase) noise in each cell, at `1/√reps`; the 14-point contrast axis
that the fit runs along is untouched. `--reps 50` is a sound explore setting,
`--reps 250` the reported one.

Two things are *not* CPU-sized: the generative-VLM back-end (a 7B model at 28k
passes) and SAM (native 1024×1024, ~20× a 224² forward). Those want a GPU
session and a shrunk grid — see the Kaggle route below.

## Route 1 — GitHub Actions (recommended; unattended, nothing new to sign up for)

This repository is public, and **standard GitHub-hosted runners are free and
unmetered on public repositories** — 4 vCPU, 16 GB, with a hard **6 h per job**
cap that the full VGG-19 grid clears with ~4.5 h to spare.

[`.github/workflows/log-response.yml`](../.github/workflows/log-response.yml)
runs it: Actions → *log-response run* → **Run workflow**, set `model` / `reps` /
`frequencies`, and the `.npz` surfaces, `.json` fit summary and figures come back
as a downloadable artifact (90-day retention), with the fit table on the run
summary page. It installs the CPU-only torch wheel, runs `test_pipeline` first
as a smoke check, and stops itself at 330 min so the upload still fires if you
dispatch something oversized.

By default it launches the pretrained run **and** its weight-scrambled control
as two parallel jobs — that pair is the actual claim (R² ≈ 0.98 vs ≈ 0.60 at
`prob`), and on free runners the pair costs the same wall-clock as one. The
`variants` input narrows it to one or the other.

Note that GitHub only registers `workflow_dispatch` once the file is on the
**default branch**; dispatching a workflow that exists only on a feature branch
404s. Until this merges to `master`, the branch-scoped `push` trigger is what
launches it — delete that trigger block once merged.

Pull the artifact and re-fit or re-plot locally for free — no model needed:

```bash
python -m log_response.run --load runs/run --figures out/
```

Caveat: one job = one run, and there is no helper that merges per-frequency
shards into a single surface. If a back-end can't finish inside 6 h, shrink
`--reps` rather than splitting `--frequencies` across jobs.

## Route 2 — Kaggle Notebooks (for the GPU back-ends)

The most generous free tier for the heavy models: **~30 GPU-h/week** (T4 ×2 or
P100), **12 h max per session**, and — the part that matters — sessions can be
committed to run **headless in the background**, so you don't babysit a tab.

```python
!pip install -q open_clip_torch
!git clone -q https://github.com/dekelron/RonInfra && cd RonInfra && \
  python -m log_response.run --model clip:ViT-B-32 --device cuda \
    --save /kaggle/working/clip --figures /kaggle/working/out/
```

Anything under `/kaggle/working/` persists as notebook output. Use this for
`hf:` VLMs and `sam:` — with `--device cuda --dtype float16` and a trimmed
`--frequencies 3.5,7,14,28 --reps 25`.

## Route 3 — Google Colab (interactive poking)

Free tier gives a T4 under a dynamic weekly quota, ~12 h max session — but the
**~90 min idle timeout keys on browser-tab interaction, not on whether your code
is running**. A 1.3 h unattended run will usually survive; a longer one won't if
you walk away. Fine for exploring, worse than Actions or Kaggle for the real
grid.

```python
!git clone -q https://github.com/dekelron/RonInfra
%cd RonInfra
!python -m log_response.run --model vgg19 --reps 50 --save /content/vgg19 --figures /content/out/
```

Mount Drive first if you want the `.npz` to outlive the VM.

## Route 4 — Oracle Cloud Always Free (a box that never times out)

An Always Free ARM VM has no wall-clock cap at all, so it suits "slow CPU is
fine, just let it grind." Note the free-tier Ampere A1 allowance was **halved to
2 OCPU / 12 GB in June 2026** (PAYG accounts reportedly keep 4/24 at no charge),
and A1 capacity is often unavailable in busy regions. At 2 cores expect ~2.1 h
for the VGG-19 grid. Run it under `tmux`/`nohup` and collect the `.npz`.

## The one real blocker: weight downloads

Compute is the easy part; **reaching the weight hosts is the constraint that
actually bites**. The log response is a consequence of training and does *not*
appear in a random-init net, so a sandbox that blocks
`download.pytorch.org` / `huggingface.co` cannot produce the real numbers —
`TorchvisionModel` warns and falls back to random init rather than failing.
All four routes above have open outbound network, so this only matters on a
locked-down network. If you hit it, download the `state_dict` somewhere with
access and pass `--weights r152.pth` (or `--model hf:/path/to/local/dir`).

To sanity-check the *analysis* with no weights at all, the offline verifier runs
anywhere in seconds:

```bash
python -m log_response.run --model synthetic --reps 12 --figures out/
python -m log_response.test_pipeline
```
