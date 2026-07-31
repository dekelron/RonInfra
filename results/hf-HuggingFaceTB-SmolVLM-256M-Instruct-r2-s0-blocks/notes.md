# hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks

`hf:HuggingFaceTB/SmolVLM-256M-Instruct`, 2 reps/cell, best mean R² 0.943 at `model.vision_model.encoder.layers.7`.

## What this run was for

VLM depth profile, default prompt, 42 block outputs (12 vision + 30 decoder) plus logits and prob. Replaces a --layers all attempt that was killed 41 s into the first forward pass with exit 143, the runner receiving a shutdown signal - almost certainly OOM: --layers all hooks all 471 modules including attention internals, whose outputs at ~1200 tokens are ~92 MB each. Block outputs are (1, seq, dim) only, roughly 240 MB for the whole tap set at reps 2. Taps cost no extra forward passes, so the 225 passes and the measured 14.9 s/pass are unchanged.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model hf:HuggingFaceTB/SmolVLM-256M-Instruct --reps 2 --seed 0 --save-run results/hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks --notes VLM depth profile, default prompt, 42 block outputs (12 vision + 30 decoder) plus logits and prob. Replaces a --layers all attempt that was killed 41 s into the first forward pass with exit 143, the runner receiving a shutdown signal - almost certainly OOM: --layers all hooks all 471 modules including attention internals, whose outputs at ~1200 tokens are ~92 MB each. Block outputs are (1, seq, dim) only, roughly 240 MB for the whole tap set at reps 2. Taps cost no extra forward passes, so the 225 passes and the measured 14.9 s/pass are unchanged. --figures out/ --layers model.vision_model.encoder.layers.0,model.vision_model.encoder.layers.1,model.vision_model.encoder.layers.2,model.vision_model.encoder.layers.3,model.vision_model.encoder.layers.4,model.vision_model.encoder.layers.5,model.vision_model.encoder.layers.6,model.vision_model.encoder.layers.7,model.vision_model.encoder.layers.8,model.vision_model.encoder.layers.9,model.vision_model.encoder.layers.10,model.vision_model.encoder.layers.11,model.text_model.layers.0,model.text_model.layers.1,model.text_model.layers.2,model.text_model.layers.3,model.text_model.layers.4,model.text_model.layers.5,model.text_model.layers.6,model.text_model.layers.7,model.text_model.layers.8,model.text_model.layers.9,model.text_model.layers.10,model.text_model.layers.11,model.text_model.layers.12,model.text_model.layers.13,model.text_model.layers.14,model.text_model.layers.15,model.text_model.layers.16,model.text_model.layers.17,model.text_model.layers.18,model.text_model.layers.19,model.text_model.layers.20,model.text_model.layers.21,model.text_model.layers.22,model.text_model.layers.23,model.text_model.layers.24,model.text_model.layers.25,model.text_model.layers.26,model.text_model.layers.27,model.text_model.layers.28,model.text_model.layers.29 --dtype float32
```

Code: `42af5439223f`. Weights: transformers HuggingFaceTB/SmolVLM-256M-Instruct.
