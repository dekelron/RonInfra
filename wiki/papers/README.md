# Papers — one file per work

Per-paper notes for the VLM-psychophysics artifact. Each file uses the same
template: bibliographic header, what it does, key findings, relevance to our
artifact, and how we cite / differentiate.

## Sourcing discipline (read this)

The session network policy **blocks arxiv.org, Semantic Scholar, and PDF fetch**
(403 at the proxy). Every file is tagged:
- **`verified`** — well-established work the summary reflects with confidence.
- **`snippet`** — reconstructed from web-search snippets/abstracts only;
  **numbers, methods, and author lists must be checked against the PDF** before
  they enter the preprint. arXiv IDs marked `(verify id)` are best-effort.

Do not cite any `snippet` figure in the paper without opening the source.

## Index

### A. Prior work (Ron's own — the lineage)
- [1701.04674 — Human perception in computer vision (Dekel 2017)](1701.04674-dekel-human-perception-cv.md)
- [1907.09019 — Scintillating Grid in a DNN (Sun & Dekel 2019)](1907.09019-sun-dekel-scintillating-grid.md)

### B. Direct neighbors (closest competition — must position against)
- [2508.10367 — Contrast Sensitivity in MLLMs (Malo group 2025)](2508.10367-vlm-contrast-sensitivity.md)
- [2603.20642 — Weber's Law in Transformer Magnitude Reps (Cacioli 2026)](2603.20642-weber-transformer-magnitude.md)
- [2604.04469 — Same Geometry, Opposite Noise (Cacioli follow-up 2026)](2604.04469-same-geometry-opposite-noise.md)
- [2502.15678 — Limits of Fine-Tuning for Visual Cognition in VLMs (2025)](2502.15678-finetuning-visual-cognition.md)

### C. Methodological siblings (psychophysics-for-models)
- [1611.06448 — PsyPhy evaluation framework (RichardWebster et al. 2016)](1611.06448-psyphy.md)
- [2404.05290 — MindSet: Vision toolbox (2024)](2404.05290-mindset-vision.md)
- [Brain-Score / Schrimpf et al. 2020 (Neuron)](brainscore-schrimpf-2020.md)
- [2305.17023 — Are DNNs Adequate Behavioral Models? (Wichmann & Geirhos 2023)](2305.17023-wichmann-geirhos-review.md)

### D. Human/DNN perception foundations
- [1811.12231 — Texture vs. shape bias (Geirhos et al. 2019)](1811.12231-geirhos-texture-bias.md)
- [1801.03924 — LPIPS perceptual similarity (Zhang et al. 2018)](1801.03924-lpips.md)
- [2310.13018 — Getting aligned on representational alignment (2023)](2310.13018-representational-alignment.md)

### E. Crowding cluster (the demoted add-on's prior art)
- [2004.12676 — Crowding: local vs global in humans & machines (Doerig et al. 2020)](2004.12676-doerig-crowding.md)
- [Volokitin et al. 2017 — Do DNNs suffer from crowding? (NeurIPS)](volokitin-2017-crowding.md)

### F. Illusion cluster (killed direction — awareness citations)
- [2311.00047 — Grounding Visual Illusions in Language (2023)](2311.00047-grounding-illusions-language.md)
- [2501.00848 — IllusionBench+ (2025)](2501.00848-illusionbench-plus.md)
- [Illusion benchmarks — grouped notes (VIA-Bench, Illusory VQA, others)](illusion-benchmarks-grouped.md)

### G. VLM low-level-vision & deficits wave (2024–2026)
- [2504.10786 — VLM visual deficits on neuropsychological tests (2025)](2504.10786-vlm-neuropsych-deficits.md)
- [2502.16435 — Human cognitive benchmarks / MLLM visual gaps (2025)](2502.16435-cognitive-benchmarks-gaps.md)
- [2504.09393 — ViTs exhibit human-like biases (2025)](2504.09393-vit-humanlike-biases.md)
- [2503.16264 — Do quality metrics model low-level human vision? (2025)](2503.16264-quality-metrics-lowlevel.md)
- [VLM low-level benchmarks — grouped (MVP-Bench, SalBench, VLMs-vs-Human IQA, "Can't See the Obvious")](vlm-lowlevel-benchmarks-grouped.md)

### H. Magnitude / number in LLMs
- [LLM number-sense — grouped (2502.16147, 2502.01540, 2402.03328)](llm-number-sense-grouped.md)

### I. Theory of log / Weber responses (the "why")
- [Efficient coding & the log transducer (Fechner; divisive normalization)](theory-efficient-coding-log.md)
- [2312293121 — Unified framework for magnitude & discriminability (PNAS 2024)](pnas-2024-unified-magnitude.md)
- [Kausik — Weber-Fechner in ML (2204.11834, 2208.11236)](kausik-weber-fechner-ml.md)
