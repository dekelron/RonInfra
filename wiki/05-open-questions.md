# 05 — Decisions, Open Questions, Next Steps

## Decisions made

- **Design CONFIRMED: Design 1, transducer-led** (2026-07-14). Reopened after
  the Cacioli 2026 neighbor was found; Ron's call is to hold transducer-led on
  the strength of the visual-contrast domain + the Dekel-2017 lineage. Survival
  depends on five **mandatory** defense conditions (2017-lineage lead; sensory-
  encoder framing vs. symbolic magnitude; pre-registered causal LoRA effect;
  dual behaviour+RDM surface; transducer measured across spatial frequency /
  scale so it is a space, not a point). See [04-design.md](04-design.md).
  Crowding demoted to optional week-3-gated add-on.
- **Illusion direction killed** (saturated literature).
- **2017 code:** Ron will supply it; priority is the **log-response / contrast**
  material.
- **Sweep depth:** focused (done); broadened pass added crowding prior art,
  the causal-fine-tuning neighbor, and the related-work backbone —
  see [06-related-work.md](06-related-work.md).
- **Crowding novelty rescoped:** representational crowding-in-CNNs is prior art
  (Doerig 2020, Volokitin 2017); the open slice is behavioral crowding +
  uncrowding in VLMs.

## Open questions (need Ron's facts before step 3 is precise)

Numbered for easy reply.

0. **Which design?** RESOLVED — Design 1, transducer-led (see Decisions above).

1. **The log-response measurement — behavior vs. representation as the headline.**
   The 2017 result is representational (encoder embeddings). Behaviorally, a
   chat VLM can't emit an L1 distance. Options for the *behavioral* transducer:
   (a) 2-AFC contrast **discrimination** (JND vs. pedestal) → fit slope; or
   (b) suprathreshold **contrast matching** (adjust until two patches match).
   Which maps best to what you want to claim? Preference?

2. **Model list — confirm the API set.** Claude (which tier?), a GPT-4V-class,
   a Gemini-class. Budget-bounded — how many API models is "enough" for the
   claim vs. cost?

3. **Open-weights model for the representational surface + LoRA target.**
   Qwen-VL, LLaVA, or InternVL class, ~7B. Any preference / prior familiarity?
   (Drives the 1–2 GPU rental plan.)

4. **Stimuli.** Reuse 2017 stimulus generation (Gabors / bandpass patterns at
   controlled contrast & spatial frequency) ported to Python? Or clean-room
   rebuild from the paper? (You're supplying code — which parts are usable?)

5. **The causal claim's bar.** What counts as success for the LoRA experiment —
   a *statistically significant* shift in the transducer slope/threshold, or a
   specific effect size? Define the go/no-go before running (pre-registration
   hygiene, and it's an interview-defense point).

6. **Scaling answer.** Interview Q2 is "how does it scale beyond lab-N to
   millions of samples?" Draft answer now recorded in
   [06-related-work.md](06-related-work.md). Confirm it matches how you'd say it.

7. **Crowding add-on — include the uncrowding test?** Prior CNN work (Doerig
   2020) shows feedforward nets fail *uncrowding*; testing it in VLMs is the
   strongest crowding differentiator but adds stimulus-design effort. In scope
   for the gated add-on, or drop to keep the timeline?

## Immediate next step

Once Ron answers the above, produce **session-plan step 3**: the week-by-week
plan with go/no-go gates, final model list, stimulus plan, statistics plan
(staircase, threshold CIs, power analysis), and budget. **No code until the
plan is agreed.**

## Housekeeping / risks to keep visible

- **HIGHEST-PRIORITY VERIFICATION:** the entire two-collision framing pivots on
  **arXiv:2603.20642 (Cacioli)** and **2604.04469**, both **snippet-sourced
  only**. Confirm their real method / scope / author / IDs against the PDFs
  before treating the positioning as settled — if either differs materially,
  [03-literature-niche.md](03-literature-niche.md) and
  [04-design.md](04-design.md) change.
- Network policy blocks arxiv / Semantic Scholar / PDF fetch — verify all cited
  numbers against real PDFs before the preprint.
- 2019 paper is (apparently) JOV-published — keep the "peer-review" framing
  accurate.
- Positioning discipline (independent early work; credit the field) applies to
  every artifact, including this wiki and the eventual README.
