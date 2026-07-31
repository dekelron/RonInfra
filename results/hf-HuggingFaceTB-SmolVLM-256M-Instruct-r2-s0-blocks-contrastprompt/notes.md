# hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks-contrastprompt

`hf:HuggingFaceTB/SmolVLM-256M-Instruct`, 2 reps/cell, best mean R² 0.943 at `model.vision_model.encoder.layers.7`.

## What this run was for

VLM depth profile, contrast-relevant prompt. Identical to the -blocks run in model, grid, seed, dtype and tap list; the ONLY difference is the instruction, 'How much contrast does this pattern have?' against the default 'Describe this image.' Tests whether asking the model about contrast changes its contrast response - a question no classifier run can pose, since a classifier has no prompt. Both lambdas are conditional on their own prompt, so this is a paired comparison rather than a refinement, and the text token positions differ between the two runs by construction.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model hf:HuggingFaceTB/SmolVLM-256M-Instruct --reps 2 --seed 0 --save-run results/hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks-contrastprompt --notes VLM depth profile, contrast-relevant prompt. Identical to the -blocks run in model, grid, seed, dtype and tap list; the ONLY difference is the instruction, 'How much contrast does this pattern have?' against the default 'Describe this image.' Tests whether asking the model about contrast changes its contrast response - a question no classifier run can pose, since a classifier has no prompt. Both lambdas are conditional on their own prompt, so this is a paired comparison rather than a refinement, and the text token positions differ between the two runs by construction. --figures out/ --layers model.vision_model.encoder.layers.0,model.vision_model.encoder.layers.1,model.vision_model.encoder.layers.2,model.vision_model.encoder.layers.3,model.vision_model.encoder.layers.4,model.vision_model.encoder.layers.5,model.vision_model.encoder.layers.6,model.vision_model.encoder.layers.7,model.vision_model.encoder.layers.8,model.vision_model.encoder.layers.9,model.vision_model.encoder.layers.10,model.vision_model.encoder.layers.11,model.text_model.layers.0,model.text_model.layers.1,model.text_model.layers.2,model.text_model.layers.3,model.text_model.layers.4,model.text_model.layers.5,model.text_model.layers.6,model.text_model.layers.7,model.text_model.layers.8,model.text_model.layers.9,model.text_model.layers.10,model.text_model.layers.11,model.text_model.layers.12,model.text_model.layers.13,model.text_model.layers.14,model.text_model.layers.15,model.text_model.layers.16,model.text_model.layers.17,model.text_model.layers.18,model.text_model.layers.19,model.text_model.layers.20,model.text_model.layers.21,model.text_model.layers.22,model.text_model.layers.23,model.text_model.layers.24,model.text_model.layers.25,model.text_model.layers.26,model.text_model.layers.27,model.text_model.layers.28,model.text_model.layers.29 --dtype float32 --instruction How much contrast does this pattern have?
```

Code: `42af5439223f`. Weights: transformers HuggingFaceTB/SmolVLM-256M-Instruct.
