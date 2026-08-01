# hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks-contrastprompt

`hf:HuggingFaceTB/SmolVLM-256M-Instruct`, 2 reps/cell, best mean R² 0.943 at `model.vision_model.encoder.layers.7`.

## What this run was for

VLM depth profile, contrast-relevant prompt. Identical to the -blocks run in model, grid, seed, dtype and tap list; the ONLY difference is the instruction, 'How much contrast does this pattern have?' against the default 'Describe this image.' Tests whether asking the model about contrast changes its contrast response - a question no classifier run can pose, since a classifier has no prompt. Both lambdas are conditional on their own prompt, so this is a paired comparison rather than a refinement, and the text token positions differ between the two runs by construction.

## What it showed

**The prompt does not change the representation — and structurally cannot.**
Identical to [`…-r2-s0-blocks`](../hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks/notes.md)
in model, grid, seed, dtype and tap list; the instruction is the only variable.

| taps | mean \|Δλ\| | max \|Δλ\| |
|---|---|---|
| 12 vision blocks | **0.0000** | **0.0000** |
| 30 decoder layers | 0.0006 | 0.0020 |

against a depth profile spanning 0.79. Two separate reasons, and the second is
the interesting one:

* the **vision tower never sees text**, so those taps are bit-identical;
* under **causal masking the image tokens precede the prompt**, so their hidden
  states cannot attend to it. `D` averages over ~1 000 image-token positions
  against ~10 text ones, so the prompt is invisible in the average even at
  decoder taps that do contain text positions.

**Only the final position sees the prompt**, which is exactly what `logits` and
`prob` read — and there the point estimates move a long way:

| tap | "Describe this image." | "How much contrast…?" |
|---|---|---|
| `logits` | +0.134 [−0.36, +0.75] R² 0.853 | **−0.627** [−1.13, +0.16] R² 0.875 |
| `prob` | +0.485 [+0.16, +1.05] R² 0.857 | +0.909 [−0.01, +4.00] R² **0.536** |

**That shift is not resolved and must not be quoted as a prompt effect.** The
two `logits` intervals overlap over [−0.36, +0.16]; this run's `prob` fits at
R² 0.536 — the worst in the repo — with an interval covering nearly the whole
λ search range, and it is flagged for λ vs λ_mod disagreeing by 0.31.

So the answer to "does asking about contrast change the contrast response" is:
not in the representation, where it cannot; and at the readout, this pair cannot
tell. Settling the readout question needs more reps and more than one seed.

**Caveats.** reps = 2, one seed. The two prompts differ in token count, so the
decoder taps average over slightly different sequence lengths — which is exactly
why the near-zero Δλ is evidence about causal structure rather than a null.

## Reproduce

```
run.py --model hf:HuggingFaceTB/SmolVLM-256M-Instruct --reps 2 --seed 0 --save-run results/hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks-contrastprompt --notes VLM depth profile, contrast-relevant prompt. Identical to the -blocks run in model, grid, seed, dtype and tap list; the ONLY difference is the instruction, 'How much contrast does this pattern have?' against the default 'Describe this image.' Tests whether asking the model about contrast changes its contrast response - a question no classifier run can pose, since a classifier has no prompt. Both lambdas are conditional on their own prompt, so this is a paired comparison rather than a refinement, and the text token positions differ between the two runs by construction. --figures out/ --layers model.vision_model.encoder.layers.0,model.vision_model.encoder.layers.1,model.vision_model.encoder.layers.2,model.vision_model.encoder.layers.3,model.vision_model.encoder.layers.4,model.vision_model.encoder.layers.5,model.vision_model.encoder.layers.6,model.vision_model.encoder.layers.7,model.vision_model.encoder.layers.8,model.vision_model.encoder.layers.9,model.vision_model.encoder.layers.10,model.vision_model.encoder.layers.11,model.text_model.layers.0,model.text_model.layers.1,model.text_model.layers.2,model.text_model.layers.3,model.text_model.layers.4,model.text_model.layers.5,model.text_model.layers.6,model.text_model.layers.7,model.text_model.layers.8,model.text_model.layers.9,model.text_model.layers.10,model.text_model.layers.11,model.text_model.layers.12,model.text_model.layers.13,model.text_model.layers.14,model.text_model.layers.15,model.text_model.layers.16,model.text_model.layers.17,model.text_model.layers.18,model.text_model.layers.19,model.text_model.layers.20,model.text_model.layers.21,model.text_model.layers.22,model.text_model.layers.23,model.text_model.layers.24,model.text_model.layers.25,model.text_model.layers.26,model.text_model.layers.27,model.text_model.layers.28,model.text_model.layers.29 --dtype float32 --instruction How much contrast does this pattern have?
```

Code: `42af5439223f`. Weights: transformers HuggingFaceTB/SmolVLM-256M-Instruct.
