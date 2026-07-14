# 03 — Literature Sweep & The Unoccupied Niche

Focused sweep of 2024–2026 VLM-psychophysics / VLM-perception work, run to find
what is already taken. (Snippet-level; PDFs blocked by network policy — verify
before citing in the preprint.)

## The one dangerous collision

**arXiv:2508.10367 — "Contrast Sensitivity in Multimodal Large Language Models:
A Psychophysics-Inspired Evaluation"** (Hernández-Cámara, Gomez-Villa,
Jaén-Lorites, Vila-Tomás, Laparra, Malo — Valencia vision-science group,
Aug 2025).

What they do:
- Present bandpass-filtered noise at varying spatial frequency and contrast.
- Collect yes/no verbal responses → build **psychometric functions** →
  extract **contrast *detection* thresholds** → CSF = 1 / threshold.
- Models: GPT, Gemini (API chat-based).
- Findings: no model matches human CSF in both shape and scale; CSF is highly
  **prompt-sensitive**; CSF predicts behavior under frequency-filtered /
  adversarial inputs.

This is the reviewer's "already done" if the pitch were simply "behavioral
contrast psychophysics on VLMs." It must be positioned against surgically.

### Why it does not close our niche

| Axis | 2508.10367 (Malo group) | Our open ground |
|---|---|---|
| Regime | **Detection** (near-threshold: can it *see* the pattern?) | **Suprathreshold transducer** (how response scales with contrast — the log law) |
| Measure | Threshold contrast vs. spatial frequency | Discrimination scaling / Weber–Fechner slope (the 2017 R²≈98% log response) |
| Surface | Behavior only | Behavior **+ vision-encoder representation** (the actual 2017 measurement) |
| Stance | Descriptive | **Causal** (LoRA moves the transducer) |
| Models | API only | API **+ open-weights** |

Detection CSF and the suprathreshold log-transducer are **different
psychophysics**. They took the threshold; the 2017 result is the transducer
*shape* across the suprathreshold range. Their prompt-sensitivity finding is a
direct methodological warning for our behavioral protocol (robustness controls
required).

## What the sweep settled about the seed directions

- **(c) Illusion susceptibility → KILL.** Saturated. IllusionBench+
  (2501.00848), VIA-Bench / "Seeing Is Believing?" (2602.01816), HallusionBench,
  Illusory VQA (2412.08169), "Grounding Visual Illusions in Language"
  (2311.00047), "Illusions in Humans and AI" (2508.12422). Do not build here.
- **Deficit-cataloguing → avoid the framing.** "VLMs fail human visual tests"
  is taken: "Human Cognitive Benchmarks Reveal Foundational Visual Gaps in
  MLLMs" (2502.16435); "Visual Language Models show widespread visual deficits
  on neuropsychological tests" (2504.10786). Don't pitch a catalogue of
  failures; pitch a **transducer + causal manipulation**.
- **Weber's law** appears only for **length** judgment (MindSet toolbox), not
  contrast discrimination. Weber-of-contrast-discrimination is **open**.
- **(b) Crowding in VLMs → OPEN.** No behavioral crowding psychophysics for
  VLMs found (COREVQA, 2507.13405, is crowd-*scene* QA — unrelated). This is the
  most role-relevant battery: crowding is the limit on reading dense
  dashboards / screenshots, i.e. agentic computer-use.

## The unoccupied niche (one sentence)

> Parametric **transducer** psychophysics on frontier VLMs — the suprathreshold
> contrast log-response law (and, as an option, crowding limits) — measured
> **behaviorally and in the vision encoder**, with a **causal LoRA** that moves
> a measured threshold.

It separates from three neighbors at once:
- **BLINK / CV-Bench / VSR / Winoground:** accuracy-on-categories, no thresholds
  or transducer.
- **Illusion benchmarks:** categorical and descriptive.
- **2508.10367:** detection-only, behavior-only, descriptive-only.

## Citation seeds (verify PDFs before use)

- arXiv:2508.10367 — VLM contrast sensitivity (the direct competitor).
- arXiv:2502.16435 — human cognitive benchmarks / visual gaps in MLLMs.
- arXiv:2504.10786 — VLM visual deficits on neuropsychological tests.
- Illusion cluster: 2501.00848, 2602.01816, 2412.08169, 2311.00047, 2508.12422.
- Foundational credit (from positioning rules): Geirhos et al.; Zhang et al.
  2018 (LPIPS); Kubilius & Schrimpf (Brain-Score); PsyPhy.
