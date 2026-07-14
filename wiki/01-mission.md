# 01 — Mission, Builder, Constraints

## Mission

Design and build a public, recent, hands-on **VLM perceptual-evaluation
artifact**: apply rigorous visual-psychophysics methodology to current frontier
vision-language models, release **code + arXiv preprint**, and include a small
**LoRA fine-tune of an open VLM** so the work is causal, not only descriptive.

The artifact converts a 2017 psychophysics-of-vision-models result from
"historical trivia" into a live, current-model demonstration — and makes
"API-only" no longer true of the builder (a real, if small, open-VLM training
run).

Secondary uses: an interpretability portfolio piece; relevant to other labs;
useful for a future reapplication.

## Builder (facts)

- **Ron Dekel.** Ph.D. + postdoc in visual psychophysics & computational
  modeling of perception, Weizmann Institute (advisor Dov Sagi, 2011–2020).
  ~20 years computer vision (military R&D → academia → industry). Currently
  Team Leader, Applied Research at Muze AI (semiconductor optical metrology;
  2-person team).
- **Strengths:** Python/PyTorch, C++, classical CV, statistical modeling,
  information theory, experimental design.
- **Honest gaps (state plainly, never paper over):** no VLM/LLM *training*
  experience (API/prompting only); largest past runs ~4 GPUs × days; no
  distributed training; no eval-harness infrastructure built before.

These gaps shape the design: the LoRA run must fit 1–2 rented GPUs on a ~7B
open VLM; nothing requires frontier-lab compute.

## Hard constraints

1. **Time:** 4–6 weeks of evenings/weekends. **Budget:** a few hundred dollars
   total (API eval calls + short GPU rentals).
2. **Compute ceiling:** LoRA on a ~7B open VLM must fit 1–2 rented GPUs.
3. **Fairness (absolute):** no overclaiming; every paper claim must survive
   three-levels-deep interview probing; use relative numbers where absolute
   ones aren't defensible; nothing that can't be personally defended.
4. **Novelty bar:** the suite must measure something **BLINK, CV-Bench, VSR,
   and Winoground do not.** The differentiator is *method* — real psychophysics
   (adaptive staircases, thresholds, psychometric functions, controlled
   stimulus manipulation), not accuracy-on-a-benchmark.

## Positioning rules (absolute)

- **Never** claim priority over, or ownership of, the field.
- The 2017/2019 work is framed as **independent early work**; the dates speak
  for themselves.
- Related work must credit: Geirhos et al.; LPIPS (Zhang et al. 2018);
  Brain-Score / Kubilius & Schrimpf; PsyPhy; the Malo-group VLM contrast-
  sensitivity work (arXiv:2508.10367); and the broader 2024–2026 VLM-perception
  wave.
- Releasing the original 2017 code is allowed **only** as a living part of this
  repo — not as standalone résumé decoration.

## Deliverables

- GitHub repo: a runnable psychophysics suite with adapters for API models and
  open VLMs.
- arXiv preprint.
- A one-sentence CV line and a cover-letter sentence.
- Interview-grade answers to:
  1. *"What does this measure that existing benchmarks don't?"*
  2. *"How does it scale beyond psychophysics-lab N to millions of samples?"*

## Working style

Dense, terse, critical. Challenge weak ideas hard. Red-team the design as a
skeptical staff interviewer before committing. Ask batteries of numbered
questions when facts are needed.
