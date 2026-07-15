# Project Wiki — Psychophysics on VLMs

Working notes for a 4–6 week research artifact: applying rigorous visual
psychophysics to current frontier vision-language models (VLMs), released as
**code (GitHub) + arXiv preprint**, with a small **LoRA fine-tune of an open
VLM** so the work is causal, not only descriptive.

Status as of 2026-07-15: **planning / no code yet.** Design committed
(see below). Literature niche identified.

## Contents

| File | What's in it |
|---|---|
| [01-mission.md](01-mission.md) | Why this project, who's building it, hard constraints, fairness/positioning rules |
| [02-prior-work.md](02-prior-work.md) | The 2017 & 2019 papers, **the log-response result**, the RonInfra repo |
| [03-literature-niche.md](03-literature-niche.md) | 2024–2026 sweep, the collisions, the unoccupied niche, citations |
| [04-design.md](04-design.md) | Three candidate designs, scoring, the committed design |
| [05-open-questions.md](05-open-questions.md) | Decisions made, open decisions, immediate next steps |
| [06-related-work.md](06-related-work.md) | Clustered citations, red-team critiques + rebuttals, draft interview answers |
| [07-log-response-as-instrument.md](07-log-response-as-instrument.md) | The log response as a **readout instrument** — what to vary, what to show, what it is (and isn't) a proxy for |
| [08-adaptation-and-learning.md](08-adaptation-and-learning.md) | **Adaptation vs. learning** as two routes to the transducer — red-teams the memo ideas; promotes the cheap **in-context adaptation** experiment, parks the rest with prior art |
| [papers/](papers/README.md) | **One `.md` per paper** — exhaustive per-work summaries + differentiation, grouped A–I |

## One-line summary of the artifact

> Parametric **transducer** psychophysics on frontier VLMs — the suprathreshold
> contrast log-response law — measured **behaviorally and in the vision
> encoder**, with a **causal LoRA** that moves a measured threshold.

## What separates it (the interview question)

Three things at once that existing work does not do together:

- **vs. BLINK / CV-Bench / VSR / Winoground:** those measure accuracy on
  categories. This measures **thresholds and transducer shape** — a
  psychometric function, not a score.
- **vs. the illusion benchmarks (IllusionBench+, VIA-Bench, HallusionBench …):**
  those are categorical and descriptive. This is **parametric and causal**.
- **vs. the one direct competitor (contrast-sensitivity in MLLMs, 2508.10367):**
  that paper measures **detection thresholds** (behavior only, descriptive).
  This measures the **suprathreshold transducer**, in **behavior *and*
  representation**, and **manipulates it** via fine-tuning.

## Positioning discipline (non-negotiable)

This work is framed as **independent early work** (2017/2019), dates speaking
for themselves — **never** as priority over or ownership of the field. Related
work credits Geirhos et al., LPIPS (Zhang 2018), Brain-Score / Kubilius &
Schrimpf, PsyPhy, the Malo group's VLM-CSF work, and the 2024–2026
VLM-perception wave. See [01-mission.md](01-mission.md).
