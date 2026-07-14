# 05 — Decisions, Open Questions, Next Steps

## Decisions made

- **Design REOPENED** (was Design 1 + add-on). Second literature pass found a
  second strong neighbor to the transducer (Cacioli 2026, arXiv:2603.20642 —
  causal Weber's-law method in transformers). Transducer now has two neighbors;
  crowding has none. Leaning toward Design 3 (two-battery) or Design 2-forward.
  **Ron to decide** — see [04-design.md](04-design.md). This is now open
  question #0 (below).
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

0. **Which design?** (Reopened — see above.) Design 3 (two-battery: transducer +
   crowding), Design 2-forward (crowding headline, transducer as 2017-lineage
   second battery), or hold to transducer-led despite the two neighbors? This
   gates everything in step 3.

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

- Network policy blocks arxiv / Semantic Scholar / PDF fetch — verify all cited
  numbers against real PDFs before the preprint.
- 2019 paper is (apparently) JOV-published — keep the "peer-review" framing
  accurate.
- Positioning discipline (independent early work; credit the field) applies to
  every artifact, including this wiki and the eventual README.
