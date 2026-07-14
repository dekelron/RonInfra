# 03 — Literature Sweep & The Unoccupied Niche

Focused sweep of 2024–2026 VLM-psychophysics / VLM-perception work, run to find
what is already taken. (Snippet-level; PDFs blocked by network policy — verify
before citing in the preprint.)

## Two dangerous collisions (not one)

A hard second-pass search on "log / Weber-Fechner transducer" (not just
"VLM psychophysics") surfaced a second, closer neighbor. **The log response is
also not itself novel** — it is textbook human vision (near-miss to Weber,
Naka-Rushton, divisive normalization) and Dekel-2017 already showed it in CNN
*representations*. So the novelty was never the phenomenon; it is the
*behavioural + causal instantiation in the visual/contrast domain in frontier
VLMs*. Read both collisions below before treating the transducer as a safe spine.

## Collision 1

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

## Collision 2 — the method neighbor (found on the second pass)

**arXiv:2603.20642 — "Weber's Law in Transformer Magnitude Representations"**
(Cacioli, **March 2026**). Four converging paradigms — **RSA + behavioural
discrimination + precision gradients + causal intervention** — in three 7–9B
open models. This is nearly the exact multi-surface + causal template proposed
for the transducer battery.

- **Why it does not fully close our niche:** its domains are **numerical,
  temporal, spatial** magnitude in the **token stream** — *not* a sensory visual
  transducer. A second search for suprathreshold contrast transducers,
  behavioural, in VLMs returned **nothing** — that cell is empty.
- **Why it still hurts:** the *method template* (measure Weber's law 4 ways incl.
  causal) is now published and 4 weeks old. The transducer battery is defensible
  only on **domain** (visual/contrast via the vision encoder) + the **Dekel-2017
  lineage** — an interviewer who knows Cacioli will say "you changed the stimulus
  to gratings." Survivable, but it is a real punch.
- **Design consequence:** the transducer spine now has **two** strong neighbors
  (Collision 1 = detection-CSF; Collision 2 = causal-Weber method). **Crowding /
  uncrowding has none.** This argues for the two-battery suite or a
  crowding-forward design rather than resting on the transducer alone. See
  [04-design.md](04-design.md) (recommendation reopened).

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
- **(b) Crowding in VLMs → PARTIALLY open (scope carefully).** Crowding in
  neural nets is **already studied representationally**: Doerig et al. 2020
  (arXiv:2004.12676) and Volokitin et al. 2017 showed feedforward CNNs reproduce
  crowding but not **uncrowding**. What remains **open** is *behavioral* crowding
  psychophysics — critical-spacing thresholds and the **uncrowding** signature —
  in **chat VLMs**, on synthetic + real-UI stimuli. That slice is unclaimed and
  is the most role-relevant battery (crowding is the limit on reading dense
  dashboards / screenshots, i.e. agentic computer-use). Cite the CNN-crowding
  cluster; scope the claim to behavior + VLMs + uncrowding. (COREVQA, 2507.13405,
  is crowd-*scene* QA — unrelated.)

## The unoccupied niche (one sentence)

> Parametric **transducer** psychophysics on frontier VLMs — the suprathreshold
> contrast log-response law (and, as an option, crowding limits) — measured
> **behaviorally and in the vision encoder**, with a **causal LoRA** that moves
> a measured threshold.

It separates from five neighbors at once:
- **BLINK / CV-Bench / VSR / Winoground:** accuracy-on-categories, no thresholds
  or transducer.
- **Illusion benchmarks:** categorical and descriptive.
- **2508.10367 (VLM-CSF):** detection-only, behavior-only, descriptive-only.
- **2603.20642 (transformer Weber's law):** symbolic magnitude in the token
  stream, not a sensory visual transducer.
- **2502.15678 (fine-tuning visual cognition):** high-level cognition tasks,
  found brittle — vs. our low-level parametric threshold moved causally.

**Caveat, stated plainly:** for the *transducer* battery the separation from
2508.10367 + 2603.20642 is narrow (domain + lineage). For the *crowding/
uncrowding* battery there is no comparably close neighbor — which is why the
committed design is being reconsidered in favor of including crowding.

See [06-related-work.md](06-related-work.md) for the full citation map,
red-team critiques + rebuttals, and draft interview answers.

## Citation seeds

Full, clustered citation map now lives in
[06-related-work.md](06-related-work.md) (competitor, crowding prior art, causal
fine-tuning, foundational/framing, transducer theory, illusion cluster,
adjacent). Verify all PDFs before use.
