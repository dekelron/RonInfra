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
spacing / clutter, psychometric threshold for critical spacing — on synthetic
stimuli *and* real UI screenshots; across API + open; LoRA to move critical
spacing.

- Novelty 5/5 · Defensibility 4/5 · Effort 4/5 · Role-relevance 5/5
- Most defensibly novel (unoccupied) and bullseye for the target role.
- Risk: further from the 2017 anchor; more stimulus engineering.

## Design 3 — "Two-Battery Suite" (transducer + crowding)

Both of the above as one suite; shared adapter / staircase infra; LoRA on
whichever deficit is largest.

- Novelty 5/5 · Defensibility 5/5 · Effort 5/5 · Role-relevance 4/5
- Strongest artifact; measures a *space*, not a point.
- Risk: heaviest for a 4–6 week evenings/weekends budget.

## Committed decision (2026-07-14)

**Design 1 as the committed core, built so crowding (Design 2) drops in as an
add-on if the week-3 go/no-go is green.**

Rationale:
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
  psychometric measurement → compare pre/post threshold & slope.

## Still to produce (session-plan step 3)

Week-by-week plan with go/no-go gates; final model list (Claude + GPT-4V-class +
Gemini-class via API; Qwen-VL / LLaVA / InternVL-class open); stimulus-generation
plan; statistics plan (staircase design, threshold CIs, power); budget.
See [05-open-questions.md](05-open-questions.md).
