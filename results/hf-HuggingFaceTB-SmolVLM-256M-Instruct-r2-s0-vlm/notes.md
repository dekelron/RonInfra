# hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-vlm

`hf:HuggingFaceTB/SmolVLM-256M-Instruct`, 2 reps/cell, best mean R² 0.925 at `model.text_model.layers.14`.

## What this run was for

First VLM measurement. The training objective is the one axis the other 27 architectures never vary - all are ImageNet classifiers with a 1000-way softmax, and convolution, attention, depth, rectifiers, normalisation and skip connections have each been varied and excluded already. SmolVLM-256M is generative: language-modelling objective, SigLIP tower, no classification head, so prob is the softmax over a 49280-token vocabulary and the TV bound is 2/V rather than 2/1000. Comparable to the other runs on lambda, which is dimensionless; NOT comparable on the magnitude of D_prob. Standard 8x14 grid, so reps is the only deviation from the corpus. dtype float32 is deliberate and is worth 4.2x: the checkpoint ships bfloat16, which these runners emulate, measured at 130 s per forward pass against 31.1 s in float32. reps=2 is the price of fitting the full grid at that rate - 225 passes, 117 min expected against a 330 min cap, still inside it at the documented 2x runner variance. reps=2 is far below the corpus's 50, so treat single-tap lambdas as provisional and read the lambda vs lambda_mod gap as the per-tap check on whether the primary metric is reporting its own noise. Two previous full-grid attempts were cancelled at the cap, sized from an unrepresentative 7-pass probe.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model hf:HuggingFaceTB/SmolVLM-256M-Instruct --reps 2 --seed 0 --save-run results/hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-vlm --notes First VLM measurement. The training objective is the one axis the other 27 architectures never vary - all are ImageNet classifiers with a 1000-way softmax, and convolution, attention, depth, rectifiers, normalisation and skip connections have each been varied and excluded already. SmolVLM-256M is generative: language-modelling objective, SigLIP tower, no classification head, so prob is the softmax over a 49280-token vocabulary and the TV bound is 2/V rather than 2/1000. Comparable to the other runs on lambda, which is dimensionless; NOT comparable on the magnitude of D_prob. Standard 8x14 grid, so reps is the only deviation from the corpus. dtype float32 is deliberate and is worth 4.2x: the checkpoint ships bfloat16, which these runners emulate, measured at 130 s per forward pass against 31.1 s in float32. reps=2 is the price of fitting the full grid at that rate - 225 passes, 117 min expected against a 330 min cap, still inside it at the documented 2x runner variance. reps=2 is far below the corpus's 50, so treat single-tap lambdas as provisional and read the lambda vs lambda_mod gap as the per-tap check on whether the primary metric is reporting its own noise. Two previous full-grid attempts were cancelled at the cap, sized from an unrepresentative 7-pass probe. --figures out/ --dtype float32
```

Code: `2ac15b5c2b08`. Weights: transformers HuggingFaceTB/SmolVLM-256M-Instruct.
