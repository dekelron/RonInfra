# hf-HuggingFaceTB-SmolVLM-256M-Instruct-scramble-r2-s0-vlm

`hf:HuggingFaceTB/SmolVLM-256M-Instruct`, 2 reps/cell, best mean R² 0.867 at `model.text_model.layers.14`.

## What this run was for

First VLM measurement. The training objective is the one axis the other 27 architectures never vary - all are ImageNet classifiers with a 1000-way softmax, and convolution, attention, depth, rectifiers, normalisation and skip connections have each been varied and excluded already. SmolVLM-256M is generative: language-modelling objective, SigLIP tower, no classification head, so prob is the softmax over a 49280-token vocabulary and the TV bound is 2/V rather than 2/1000. Comparable to the other runs on lambda, which is dimensionless; NOT comparable on the magnitude of D_prob. Standard 8x14 grid, so reps is the only deviation from the corpus. dtype float32 is deliberate and is worth 4.2x: the checkpoint ships bfloat16, which these runners emulate, measured at 130 s per forward pass against 31.1 s in float32. reps=2 is the price of fitting the full grid at that rate - 225 passes, 117 min expected against a 330 min cap, still inside it at the documented 2x runner variance. reps=2 is far below the corpus's 50, so treat single-tap lambdas as provisional and read the lambda vs lambda_mod gap as the per-tap check on whether the primary metric is reporting its own noise. Two previous full-grid attempts were cancelled at the cap, sized from an unrepresentative 7-pass probe.

## What it showed

The control for the first VLM run. **It is valid**, by the renormalisation
rule: SmolVLM has **zero BatchNorm** (census: 25 LayerNorm, 61 RMSNorm, all of
which renormalise by the current input), no tap is pinned at a λ search bound,
and `max D_logits` = 2.93 against the trained run's 0.72 — ordinary magnitudes,
nothing like `resmlp-12-scramble`'s 2409.

The softmax is only *partly* in its affine regime: `ratio × V` = 0.612 where a
pure affine softmax gives 1.0 and the broken ResMLP control gave 0.127, with
r(logits, prob) = 0.947. `D_prob` reaches **92.3%** of its 2/V ceiling against
the trained run's 7.6% — a scrambled language head flips between confident
tokens, so the two distributions are nearly disjoint. Read `prob` here as a
saturation artifact, not a response.

What the control establishes, at the three hidden taps where the intervals do
not overlap the trained run's:

| tap | scrambled λ | trained λ |
|---|---|---|
| `model.vision_model.encoder.layers.11` | +0.560 [+0.43, +0.70] R² 0.984 | +0.047 [−0.20, +0.30] |
| `model.text_model.layers.14` | +0.524 [+0.37, +0.70] R² 0.978 | +0.020 [−0.21, +0.27] |
| `model.text_model.layers.29` | +0.549 [+0.37, +0.74] R² 0.971 | −0.120 [−0.37, +0.13] |

And λ is nearly **flat across frequency** (+0.34 → +0.59, range 0.22–0.27)
where the trained run spans 0.89–1.03 — the 7th matched pair showing frequency
structure is a property of learning.

Same caveats as its trained companion: reps = 2, one seed, one instruction. The
`--notes` string recorded above claims float32 "is worth 4.2x"; that comparison
was confounded and the dtype effect is not established — see the trained
companion's notes and [`wiki/Results.md`](../../wiki/Results.md#vlm-forward-pass-cost-varies-9-and-why-is-unresolved).

## Reproduce

```
run.py --model hf:HuggingFaceTB/SmolVLM-256M-Instruct --reps 2 --seed 0 --save-run results/hf-HuggingFaceTB-SmolVLM-256M-Instruct-scramble-r2-s0-vlm --notes First VLM measurement. The training objective is the one axis the other 27 architectures never vary - all are ImageNet classifiers with a 1000-way softmax, and convolution, attention, depth, rectifiers, normalisation and skip connections have each been varied and excluded already. SmolVLM-256M is generative: language-modelling objective, SigLIP tower, no classification head, so prob is the softmax over a 49280-token vocabulary and the TV bound is 2/V rather than 2/1000. Comparable to the other runs on lambda, which is dimensionless; NOT comparable on the magnitude of D_prob. Standard 8x14 grid, so reps is the only deviation from the corpus. dtype float32 is deliberate and is worth 4.2x: the checkpoint ships bfloat16, which these runners emulate, measured at 130 s per forward pass against 31.1 s in float32. reps=2 is the price of fitting the full grid at that rate - 225 passes, 117 min expected against a 330 min cap, still inside it at the documented 2x runner variance. reps=2 is far below the corpus's 50, so treat single-tap lambdas as provisional and read the lambda vs lambda_mod gap as the per-tap check on whether the primary metric is reporting its own noise. Two previous full-grid attempts were cancelled at the cap, sized from an unrepresentative 7-pass probe. --figures out/ --dtype float32 --scramble
```

Code: `2ac15b5c2b08`. Weights: transformers HuggingFaceTB/SmolVLM-256M-Instruct.
