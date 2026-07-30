# hf-llava-hf-llava-interleave-qwen-0-5b-hf-r1-s0-probe224

`hf:llava-hf/llava-interleave-qwen-0.5b-hf`, 1 reps/cell, best mean R² 0.973 at `model.vision_tower.encoder.layers.25`.

## What this run was for

COST PROBE 2, not a science run. Identical to the size-384 probe except --size 224, to measure whether feeding a smaller image collapses llava-interleave's anyres tiling and cuts the 62 s/forward-pass cost. Frequencies are in cycles per image so the stimulus is unchanged. Needs 7.4x to make the standard grid fit the 330 min cap.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model hf:llava-hf/llava-interleave-qwen-0.5b-hf --reps 1 --seed 0 --save-run results/hf-llava-hf-llava-interleave-qwen-0-5b-hf-r1-s0-probe224 --notes COST PROBE 2, not a science run. Identical to the size-384 probe except --size 224, to measure whether feeding a smaller image collapses llava-interleave's anyres tiling and cuts the 62 s/forward-pass cost. Frequencies are in cycles per image so the stimulus is unchanged. Needs 7.4x to make the standard grid fit the 330 min cap. --figures out/ --frequencies 1,7 --contrasts 0.25,0.5,1 --size 224
```

Code: `c03e8957db99`. Weights: transformers llava-hf/llava-interleave-qwen-0.5b-hf.
