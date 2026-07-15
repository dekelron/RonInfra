# 06 — Related Work, Red-Team, Interview Defense

Organized citation map + anticipated critiques with rebuttals + draft answers to
the two required interview questions. All arXiv/PDF numbers are snippet-sourced
(network policy blocks PDF fetch) — **verify before the preprint.**

> **Per-paper detail lives in [papers/](papers/README.md)** — one `.md` per work
> with exhaustive summary + differentiation, grouped A–I. This file is the
> high-level map; go to `papers/` for any single work.

## Related work, by cluster

### A. The direct competitor (behavioral VLM psychophysics)
- **arXiv:2508.10367** — Contrast Sensitivity in MLLMs (Hernández-Cámara,
  Gomez-Villa, Laparra, Malo; Aug 2025). Behavioral **detection** CSF via
  yes/no responses; fits **Weibull** psychometric functions; API models (GPT,
  Gemini). *Our distinction: suprathreshold transducer, not detection; behavior
  + representation; causal.* See [03-literature-niche.md](03-literature-niche.md).

### B. Crowding in neural networks (prior art for the add-on battery)
- **Doerig, Bornet, Choung, Herzog 2020** — "Crowding Reveals Fundamental
  Differences in Local vs. Global Processing in Humans and Machines"
  (arXiv:2004.12676; *Vision Research*). Feedforward CNNs reproduce crowding but
  **not uncrowding**; recurrent grouping/segmentation argued necessary.
- **Volokitin, Roig, Poggio 2017** — "Do Deep Neural Networks Suffer from
  Crowding?" (NeurIPS).
- "Crowding in humans is unlike that in convolutional neural networks"
  (*Neural Networks* 2020).
- **Implication:** crowding-in-DNNs is done at the **classification /
  representational** level. The **open** slice is *behavioral* crowding (and the
  **uncrowding** signature) in **chat VLMs**, on synthetic + real-UI stimuli.
  Cite this cluster; scope the novelty claim to behavior + VLMs + uncrowding.

### B2. Weber/log transducers in transformers — THE method neighbor (critical)
- **arXiv:2603.20642** — "Weber's Law in Transformer Magnitude Representations"
  (Cacioli, **March 2026**). Four converging paradigms — **RSA, behavioural
  discrimination, precision gradients, causal intervention** — across three
  magnitude domains (**numerical, temporal, spatial**) in three 7–9B
  instruction-tuned open models (Llama/Mistral/Qwen). Log-compressive geometry
  (RSA vs. Weber DM .68–.96); behaviour/geometry and layer-wise causal
  dissociations. **This is nearly the exact multi-surface + causal template
  proposed for our transducer battery — but in symbolic/token magnitude, NOT
  vision/contrast.** Our differentiation is now narrow and must be explicit:
  *sensory visual transducer (contrast/luminance through a vision encoder) vs.
  numeric magnitude in the token stream; and the Dekel-2017 lineage.*
- **arXiv:2604.04469** — "Same Geometry, Opposite Noise: Transformer Magnitude
  Representations Lack Scalar Variability" (follow-up; scalar-variability angle).
- LLM number-sense cluster: 2502.16147 (Number Representations in LLMs),
  2502.01540 (What is a Number…), 2402.03328 (Generative AI lacks visual number
  sense).
- **Human/ML grounding (log contrast is textbook, not a discovery):**
  "near-miss to Weber" for suprathreshold contrast discrimination; divisive
  normalization. Kausik
  "Psychophysical Machine Learning" (2208.11236) / "Accelerating ML via
  Weber-Fechner" (2204.11834) impose Weber-Fechner in training (not measure
  emergence).

### C. Causal fine-tuning of visual behavior (neighbor to the LoRA component)
- **arXiv:2502.15678** — "Testing the Limits of Fine-Tuning for Improving Visual
  Cognition in VLMs" (2025). Fine-tuning improves alignment on
  intuitive-physics / causal-reasoning tasks but is **brittle**. *Our
  distinction: we move a low-level, parametric **psychophysical threshold /
  transducer slope**, not high-level cognition — a sharper, more falsifiable
  causal claim; their brittleness finding is a reason to measure at the
  psychophysical level.*
- **arXiv:2602.12498** — layer-specific (causal) fine-tuning for negation in
  medical VLMs. Precedent for targeting causally-responsible layers.

### D. Foundational / framing (required by positioning rules)
- **Wichmann & Geirhos 2023** — "Are DNNs Adequate Behavioral Models of Human
  Visual Perception?" (*Annual Review of Vision Science*; arXiv:2305.17023).
  Anchor review; "statistical tool vs computational model" framing.
- **Geirhos et al.** — texture bias / robustness / human-machine comparison.
- **Zhang et al. 2018 (LPIPS)** and the perceptual-loss lineage (Johnson et al.
  2016, arXiv:1603.08155; task-specific reconstruction loss, arXiv:2103.14616 —
  this lineage cites Dekel 2017).
- **Kubilius & Schrimpf (Brain-Score)**; **PsyPhy**.
- "Getting aligned on representational alignment" (arXiv:2310.13018).

### E. Perceptual theory grounding (for the transducer stats)
- **PNAS 2024** — "A unified framework for perceived magnitude and
  discriminability of sensory stimuli" (2312293121). Weber / Fechner / Stevens
  reconciled: discriminability vs. perceived magnitude access different aspects
  of one representation. Directly relevant to how the log-transducer is fit and
  interpreted.
- NYU CNS Zhou et al. 2024 (contrast/transducer modeling) — check for a
  nonlinear-transducer / contrast-constancy formulation to align notation.

### F. Illusion benchmarks (killed direction — cite to show awareness)
- IllusionBench+ (2501.00848), VIA-Bench / "Seeing Is Believing?" (2602.01816),
  HallusionBench, Illusory VQA (2412.08169), "Grounding Visual Illusions in
  Language" (2311.00047), "Illusions in Humans and AI" (2508.12422).

### G. Adjacent (human-like biases / deficits — watch, don't build)
- "ViTs Exhibit Human-Like Biases…" (arXiv:2504.09393).
- Human cognitive benchmarks / MLLM visual gaps (2502.16435);
  neuropsychological deficits (2504.10786).

## Red-team: anticipated critiques & rebuttals

1. **"This is just 2508.10367."**
   → No. That is detection-threshold CSF; this is the **suprathreshold
   transducer** (log-response / discrimination scaling), measured in **behavior
   *and* the vision encoder**, and **causally manipulated** via LoRA. Different
   regime, extra surface, extra stance.

2. **"You already did this in 2017."**
   → 2017 was **representational** (RDM / encoder correlates), not behavioral
   psychophysics. This work adds behavioral psychometrics **and** a causal
   intervention. Framed as independent early work; no priority claim.

3. **"Fine-tuning changing behavior is known and brittle (2502.15678)."**
   → Those target high-level visual cognition. Moving a **parametric
   psychophysical threshold** is a sharper claim; brittleness at the cognition
   level is precisely why a low-level, controlled measure is informative. Pre-
   register the effect-size bar (see [05-open-questions.md](05-open-questions.md)).

3b. **"Cacioli (2603.20642) already measured Weber's law in transformers with
    RSA + behavior + causal intervention."**
   → In **symbolic magnitude** (number/time/space) through the **token stream**.
   Ours is a **sensory visual transducer** — contrast/luminance through the
   **vision encoder** — the behavioural + causal extension of Dekel-2017's CNN
   contrast log-law to frontier VLMs. Different modality, different pathway,
   different lineage. **Honest caveat:** the *method template* is no longer
   novel; the novelty is the **visual-contrast domain** + the **2017 lineage**,
   and (more cleanly) the **crowding/uncrowding** battery, which Cacioli does not
   touch at all. This is why the design leans toward including crowding rather
   than resting on the transducer alone.

4. **"Prompt-sensitivity makes VLM psychophysics unreliable."**
   → Acknowledged (2508.10367 shows it). Mitigate with prompt-robustness
   controls (multiple paraphrases, report variance), and lean on the
   **representational** surface for open models, which is prompt-free.

5. **"Better AI ≠ better model of biology"** (arXiv:2504.16940).
   → Agreed, and not the claim. The claim is **measurement methodology** + a
   **causal probe**, not "VLMs are good models of humans." Human alignment is
   reported as a comparison axis, never as the thesis.

6. **"N is tiny vs. ML benchmarks."**
   → That's the point of psychophysics — *few stimuli, many trials, thresholds*.
   Scaling answer below addresses volume.

## Draft answers to the two required interview questions

**Q1 — "What does this measure that BLINK / CV-Bench / VSR / Winoground don't?"**
Those score **accuracy on categories**. This measures the **shape of the
perceptual transducer** — a psychometric function with a **threshold and slope**
recovered by adaptive staircases under parametric stimulus control. It answers
*"how does the model's response scale with the physical variable, and where is
its just-noticeable difference,"* not *"what fraction did it get right."* Plus a
**causal** result: a targeted LoRA moves the measured threshold.

**Q2 — "How does it scale beyond psychophysics-lab N to millions of samples?"**
Own both halves; don't imply seamless scale. (a) **Generation scales trivially:**
stimuli are procedural and staircases automated, so the suite can emit millions
of labelled, parametrically-controlled trials — a data engine for training/eval —
and the representational surface is fully offline/batchable to arbitrary N (cf.
PsyPhy's millions of procedurally rendered scenes). (b) **But the scientific unit
is a threshold, not a sample:** thresholds need many trials *per condition*, not
many conditions, so "millions of samples" is the wrong axis for the *measurement*
and the right axis for the *stimulus/data-generation* use. The method is a
controlled-stimulus generator whose value is **precision** (thresholds, slopes,
CIs); volume is a separate compute dial. Say which you mean — conflating them is
the trap in this question.
