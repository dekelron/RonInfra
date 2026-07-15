# VLM low-level-vision benchmarks — grouped notes

- **Cluster:** G — VLM low-level-vision wave (2024–2026)
- **Sourcing:** `snippet` for all; verify before citing

Grouped: these establish the **backdrop** — VLMs are being probed on low-level
perception and repeatedly found weak — without occupying our specific
transducer + causal niche.

## Members

- **MVP-Bench** (arXiv:2410.04345, 2024) — multi-level visual perception: can
  LVLMs perceive at multiple levels like humans? Benchmark of low→high perception.
- **SalBench** — saliency of **low-level features** (color, intensity,
  orientation); reports LVLMs miss obvious salient anomalies (e.g., GPT-4o ~47.6%).
- **VLMs vs Human: Perceptual Image Quality Assessment** (arXiv:2603.24578,
  2026) — six VLMs vs. psychophysical ground truth on **contrast, colorfulness,
  preference**; attribute-dependent alignment (contrast: GPT/Gemini ~10%
  disagreement; underperform on contrast vs. colorfulness).
- **Vision-Language Models Can't See the Obvious** (arXiv:2507.04741, 2025) —
  salient-feature blindness.
- **SenseBench** (arXiv:2605.10576, 2026) — remote-sensing low-level perception &
  description (domain-specific).
- **Do VLMs Measure Up?** (arXiv:2510.26865, 2025) — benchmarking visual
  measurement/quantity reading.
- **VLMs Using Language-Guided Inference Capture Context-Sensitivity of Human
  Object Recognition** (ScienceDirect 2026) — CLIP-style ViT + language
  supervision approaches human context-sensitivity.

## Relevance to our artifact

Rich related-work backdrop and a **methods quarry** (esp. 2603.24578's
contrast/colorfulness psychophysics against ground truth). None measures a
**suprathreshold transducer law** behaviourally + representationally with a
**causal** manipulation.

## How we cite / differentiate

Cite as the 2024–2026 wave establishing low-level fragility; position our niche as
the parametric-transducer + causal cell none of them fill. Note 2603.24578 also
touches contrast — read it closely to keep our distinction (transducer vs.
attribute-rating alignment) crisp.
