# hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-vlm

`hf:HuggingFaceTB/SmolVLM-256M-Instruct`, 2 reps/cell, best mean R² 0.925 at `model.text_model.layers.14`.

## What this run was for

First VLM measurement. The training objective is the one axis the other 27 architectures never vary - all are ImageNet classifiers with a 1000-way softmax, and convolution, attention, depth, rectifiers, normalisation and skip connections have each been varied and excluded already. SmolVLM-256M is generative: language-modelling objective, SigLIP tower, no classification head, so prob is the softmax over a 49280-token vocabulary and the TV bound is 2/V rather than 2/1000. Comparable to the other runs on lambda, which is dimensionless; NOT comparable on the magnitude of D_prob. Standard 8x14 grid, so reps is the only deviation from the corpus. dtype float32 is deliberate and is worth 4.2x: the checkpoint ships bfloat16, which these runners emulate, measured at 130 s per forward pass against 31.1 s in float32. reps=2 is the price of fitting the full grid at that rate - 225 passes, 117 min expected against a 330 min cap, still inside it at the documented 2x runner variance. reps=2 is far below the corpus's 50, so treat single-tap lambdas as provisional and read the lambda vs lambda_mod gap as the per-tap check on whether the primary metric is reporting its own noise. Two previous full-grid attempts were cancelled at the cap, sized from an unrepresentative 7-pass probe.

## What it showed

**The compression survives a change of training objective.** The three hidden
taps sit at the log law and their weight-scrambled companions do not, with
**non-overlapping** 95% profile-F intervals at all three:

| tap | trained λ | scrambled λ | separated |
|---|---|---|---|
| `model.vision_model.encoder.layers.11` | **+0.047** [−0.20, +0.30] R² 0.955 | +0.560 [+0.43, +0.70] R² 0.984 | yes |
| `model.text_model.layers.14` | **+0.020** [−0.21, +0.27] R² 0.965 | +0.524 [+0.37, +0.70] R² 0.978 | yes |
| `model.text_model.layers.29` | **−0.120** [−0.37, +0.13] R² 0.964 | +0.549 [+0.37, +0.74] R² 0.971 | yes |
| `logits` | +0.134 [−0.36, +0.75] R² 0.853 | +0.632 [+0.45, +0.83] R² 0.970 | no |
| `prob` | +0.485 [+0.16, +1.05] R² 0.857 | +0.938 [+0.59, +1.28] R² 0.927 | no |

All three trained intervals contain λ = 0; none of the scrambled ones do.

**`prob` is not usable on this run and should not be quoted.** λ = +0.485 at
λ-R² 0.857, its interval overlaps the control's, and per-frequency it runs
+0.03 to **+2.77** — that is sampling noise at 2 reps over a 49 280-way
softmax, not a response. The repo already preferred a pre-softmax tap on the
classifiers; on a VLM it is mandatory. `D_prob` reaches 7.6% of its 2/V
ceiling (V = 49 280), against the scrambled run's 92.3%.

**Frequency structure needs trained weights — a 7th matched pair.** Trained λ
falls monotonically with frequency while scrambled is nearly flat:

| tap | trained, 1 → 75 cyc/img | range | scrambled range | ratio |
|---|---|---|---|---|
| `…encoder.layers.11` | +0.27 → −0.62 | 0.89 | 0.25 | 3.6× |
| `…text_model.layers.14` | +0.30 → −0.70 | 1.00 | 0.22 | 4.5× |
| `…text_model.layers.29` | +0.27 → −0.76 | 1.03 | 0.27 | 3.8× |

3.6–4.5× sits inside the 3.4–20.8× band the six classifier pairs give, so the
generative objective behaves like the classifiers on this axis.

**Caveats, and they are heavy.** **reps = 2**, 25× below the corpus's 50 — the
price of fitting the full 8×14 grid at 31.1 s per forward pass. One seed, one
instruction (`'Describe this image.'`, chat template recorded in `run.json`).
The λ vs λ_mod gap is the per-tap noise check and stays ≤ 0.072 at the three
hidden taps, which is why those are quoted and `prob` (gap 0.005 but λ-R²
0.857 and a wild frequency profile) is not. Intervals here are profile-F
within one run; they answer "do these two runs differ given their own contrast
sampling", **not** "does this reproduce across seeds".

**On dtype — the run's own `--notes` string overstates this.** It says float32
"is worth 4.2x". That compared this run's dtype against a 14-cell *probe* on a
different runner, so it isolated nothing. What is measured: this run did 225
passes in 3 353 s = **14.9 s/pass**, while the cancelled bfloat16 full-grid
attempt ran at 130 s/pass — but a bfloat16 probe ran at 14.6 and a float32 probe
at 31.1, i.e. within-dtype spreads of 8.9× and 2.1×. Runner variance covers the
gap on its own, so the dtype effect is **not established**; see
[`wiki/Results.md`](../../wiki/Results.md#vlm-forward-pass-cost-varies-9-and-why-is-unresolved).
The `--notes` text is left as recorded because it is provenance.

## Reproduce

```
run.py --model hf:HuggingFaceTB/SmolVLM-256M-Instruct --reps 2 --seed 0 --save-run results/hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-vlm --notes First VLM measurement. The training objective is the one axis the other 27 architectures never vary - all are ImageNet classifiers with a 1000-way softmax, and convolution, attention, depth, rectifiers, normalisation and skip connections have each been varied and excluded already. SmolVLM-256M is generative: language-modelling objective, SigLIP tower, no classification head, so prob is the softmax over a 49280-token vocabulary and the TV bound is 2/V rather than 2/1000. Comparable to the other runs on lambda, which is dimensionless; NOT comparable on the magnitude of D_prob. Standard 8x14 grid, so reps is the only deviation from the corpus. dtype float32 is deliberate and is worth 4.2x: the checkpoint ships bfloat16, which these runners emulate, measured at 130 s per forward pass against 31.1 s in float32. reps=2 is the price of fitting the full grid at that rate - 225 passes, 117 min expected against a 330 min cap, still inside it at the documented 2x runner variance. reps=2 is far below the corpus's 50, so treat single-tap lambdas as provisional and read the lambda vs lambda_mod gap as the per-tap check on whether the primary metric is reporting its own noise. Two previous full-grid attempts were cancelled at the cap, sized from an unrepresentative 7-pass probe. --figures out/ --dtype float32
```

Code: `2ac15b5c2b08`. Weights: transformers HuggingFaceTB/SmolVLM-256M-Instruct.
