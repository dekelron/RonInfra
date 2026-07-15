# 04 — Designs & The Committed Plan

Three candidate designs were scored on Novelty / Defensibility / Effort /
Role-relevance (Role = relevance to "Visual Knowledge Work" / agentic
computer-use). **Design 1 is committed**, structured so Design 2 can attach if a
week-3 gate is green.

## Design 1 — "The Transducer" (log-response as spine) ✅ COMMITTED

Behavioral suprathreshold contrast-discrimination psychometrics (JND vs.
pedestal contrast) across API + open VLMs; fit the Weber–Fechner log slope;
replicate in open-encoder embeddings (direct 2017 replication); LoRA on one ~7B
open VLM targeting the transducer, then re-measure slope / threshold.

- Novelty 4/5 · Defensibility 5/5 · Effort 3/5 · Role-relevance 3/5
- Deepest and most *Ron's own*; cleanest causal story.
- Risk: narrow; adjacent to 2508.10367, so positioning must be surgical
  (suprathreshold transducer ≠ detection CSF — see
  [03-literature-niche.md](03-literature-niche.md)).

## Design 2 — "Crowding under Computer-Use" (role-first)

Behavioral crowding psychophysics — target recognition vs. flanker
spacing / clutter, psychometric threshold for critical spacing, plus the
**uncrowding** signature — on synthetic stimuli *and* real UI screenshots;
across API + open; LoRA to move critical spacing.

- Novelty 4.5/5 · Defensibility 4/5 · Effort 4/5 · Role-relevance 5/5
- Bullseye for the target role. Novelty is real but **must be scoped**:
  crowding in CNNs is already studied representationally (Doerig et al. 2020,
  arXiv:2004.12676; Volokitin et al. 2017). The open slice is **behavioral**
  crowding + **uncrowding** in **chat VLMs** — cite the CNN-crowding cluster and
  claim only that slice. Uncrowding is the strongest differentiator (feedforward
  nets fail it).
- Risk: further from the 2017 anchor; more stimulus engineering.

## Design 3 — "Two-Battery Suite" (transducer + crowding)

Both of the above as one suite; shared adapter / staircase infra; LoRA on
whichever deficit is largest.

- Novelty 5/5 · Defensibility 5/5 · Effort 5/5 · Role-relevance 4/5
- Strongest artifact; measures a *space*, not a point.
- Risk: heaviest for a 4–6 week evenings/weekends budget.

## Committed decision (2026-07-14, confirmed after second literature pass)

**Design 1 — transducer-led.** The choice was reopened when the second search
found a second strong neighbor (**arXiv:2603.20642**, Cacioli Mar 2026: RSA +
behaviour + causal Weber's law in 7–9B transformers, symbolic magnitude) on top
of the detection-CSF neighbor (2508.10367). Ron's call: **hold transducer-led**
— the visual-contrast domain + the Dekel-2017 lineage are the genuine edge, and
crowding is both further from that core and heavier on stimulus engineering.

This is defensible **only if the design is built to beat the two neighbors.**
The following are **requirements, not options** (ordered by what actually carries
the novelty):

1. **Primary novelty = mechanism disambiguation.** No neighbor resolves whether a
   log/Weber signature reflects genuine **log-encoding + constant noise** or
   **linear-encoding + scalar noise** (2604.04469 shows geometry and noise
   dissociate). The dual behaviour+RDM surface + causal LoRA can *decide* it in
   the **sensory/contrast** domain. **Headline this** — it is the one claim no
   neighbor makes. See [papers/theory-efficient-coding-log.md](papers/theory-efficient-coding-log.md).
2. **Sensory transducer via the vision encoder**, explicitly distinguished from
   Cacioli's symbolic/token magnitude (different modality *and* pathway). Cite
   him first, distinguish in one line.
3. **Causal LoRA must land a pre-registered effect** — separates from the
   descriptive CSF paper and puts causal intervention in the *sensory* domain
   where it has not been done. Fix the effect-size bar before running.
4. **Dual surface — but know where it applies.** Behaviour ↔ encoder-RDM only
   works on **open-weights** models; API models (Claude/GPT/Gemini) give
   behaviour ONLY. So the frontier-model story is behaviour-only (same surface as
   2508.10367), and the novel dual-surface + causal work lives on **7–9B open
   models — exactly Cacioli's regime.** State this openly; the differentiation
   there is modality/pathway + mechanism, not experimental setup.
5. **Not a single point.** Measure the transducer **across spatial frequency**
   (the 2017 contrast-constancy / "deblurring" result) and/or across model scale
   — a *space* within the contrast domain, so the artifact is a suite, not one
   slope.
6. **Lineage is framing, not novelty.** The Dekel-2017 tie-in is a credibility /
   narrative asset (independent early work) — **not** a scientific-novelty claim.
   Novelty must stand on 1–3 alone; do not let the argument become circular
   ("it's novel because I did the earlier version").

Crowding/uncrowding is **demoted to an optional week-3-gated add-on** (its lack
of a close neighbor is noted, but it is no longer the recommended lead).

Rationale (why the transducer works as the spine when built to the above):
- The **log-response transducer** is the result Ron can defend three levels
  deep — it is the defensible spine.
- **Crowding** is what makes the suite *role-relevant* — held as an add-on, not
  a schedule bet.
- Build the staircase / adapter / psychometric-fitting infrastructure
  **generically** from day one so a second battery is a plug-in, not a rewrite.
- This protects the 4–6 week timeline while leaving a path to the full suite.

## Architecture implications (for when code starts — not yet)

- **Generic core:** stimulus generator → adapter (model under test) → response
  collector → staircase controller → psychometric fit → threshold + CI.
- **Two adapter types:** API models (behavioral surface only) and open-weights
  VLMs (behavioral **+** vision-encoder representational surface).
- **Batteries are plug-ins** to the core (transducer first; crowding second).
- **Causal module:** LoRA fine-tune on a ~7B open VLM → re-run the same
  psychometric measurement → compare pre/post threshold & slope. Position
  against arXiv:2502.15678 (fine-tuning helps high-level visual cognition but is
  brittle): our claim is sharper — moving a **low-level parametric threshold**,
  with a pre-registered effect-size bar.

## Still to produce (session-plan step 3)

Week-by-week plan with go/no-go gates; final model list (Claude + GPT-4V-class +
Gemini-class via API; Qwen-VL / LLaVA / InternVL-class open); stimulus-generation
plan; statistics plan (staircase design, threshold CIs, power); budget.
See [05-open-questions.md](05-open-questions.md).
