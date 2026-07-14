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

## ⚠️ Decision reopened (2026-07-14, after second literature pass)

A hard second search found **arXiv:2603.20642** (Cacioli, Mar 2026): RSA +
behaviour + causal intervention for Weber's law in 7–9B transformers — nearly
the transducer battery's method template, but in symbolic magnitude, not vision.
Combined with the detection-CSF neighbor (2508.10367), the **transducer spine
now has two strong neighbors**, while **crowding/uncrowding has none**.

This reopens the design choice. Leading options:
- **Elevate to Design 3 (two-battery suite)** — transducer + crowding as
  co-equal, so the artifact never rests on the contested ground alone.
- **Or Design 2 (crowding-forward)** — lead with the clearly-open battery, keep
  the transducer as the 2017-lineage supporting result.

Recommendation now: **Design 3 if the 4–6 week budget can absorb it; otherwise
Design 2-forward** (crowding as headline, transducer as the second battery and
the 2017 tie-in). **Ron to decide.** The prior "Design 1 + add-on" framing below
is retained for context but is no longer the default.

## (Superseded) Committed decision (2026-07-14)

~~Design 1 as the committed core, built so crowding (Design 2) drops in as an
add-on if the week-3 go/no-go is green.~~ Superseded by the reopened decision
above.

Rationale (still valid for why the transducer is *a* battery, not *the* battery):
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
