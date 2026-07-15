# 07 — The log-linear response as a measurement instrument

Status: **thinking note** (2026-07-15). Captures Ron's "what can we experiment
against / what to show / what is it a proxy for" questions and turns them into a
usable frame. Not a design change — Design 1 is still committed
([04-design.md](04-design.md)). This page is the *conceptual scaffold under* the
transducer work: it says what the log response can and cannot be used to measure,
so the eventual claims stay inside the fairness bar ([01-mission.md](01-mission.md)).

## The reframe (the one idea)

Stop treating the log-linear response as a **phenomenon we describe**. Treat it as
an **instrument we read out**. A phenomenon is a result ("the model has a log
contrast response"). An instrument is a *dial* whose reading changes when you vary
something — and a dial is only useful if theory predicts *which way it should move*.

Efficient coding gives us that prediction (see
[papers/theory-efficient-coding-log.md](papers/theory-efficient-coding-log.md)):
a bounded-output channel encoding a scale-invariant (~1/x) prior is optimal when
its transfer function is logarithmic. So the log slope is not decoration — it is a
**readout of how well a representation has matched its input prior under a
dynamic-range constraint.** That, and only that, is what the dial measures. Every
question below is "what do I turn, and what do I read."

## Q1 — What can we experiment against? (the axes you turn)

The response is the *readout*; these are the independent variables you vary while
reading it. Ranked by whether theory gives a **directional prediction** (which is
what separates a real experiment from a fishing trip) and by defensibility.

| Axis | What you vary | Does theory predict a direction? | Verdict |
|---|---|---|---|
| **Input statistics** | train/adapt on natural (~1/x, heavy-tailed) vs whitened/uniform contrast | **Yes** — optimal transfer = CDF of the prior; change the prior, the slope should track it | **Strongest causal lever.** This is efficient coding's one falsifiable knob. |
| **Causal LoRA** | fine-tune the ~7B open VLM toward/away the transducer | Yes (pre-registered effect) | Already in the design; the headline causal result. |
| **Layers / depth** | read the slope at each vision-encoder layer | Partial — expect emergence early→mid, stabilize late | **Localization**, not ranking. Good as a *space*, not a scalar. |
| **Training checkpoints** | slope vs training step on an open model with public checkpoints | Yes — should *emerge* as the prior is absorbed | Strong if checkpoints exist; a genuine "did learning work" readout. |
| **Model scale** | slope across a size family (same arch/data) | Weakly — cleaner transducer with scale, plausibly | Descriptive; fine as one axis of the "space," weak alone. |
| **Architecture** | CNN vs ViT vs hybrid; divisive-norm vs LayerNorm | Weakly — normalization enables compression | Confounded (data/objective differ too). Relative only. |
| **Objective / loss** | contrastive vs generative vs classification; contrast-invariance term | Partial — invariance objective pushes compression | Interesting, heavily confounded across real models. |
| **Optimizer / LR / schedule** | Adam vs SGD, etc. | **No** | **Do not pitch this.** No theory prediction → any result is a coincidence, not a measurement. |

The load-bearing distinction: axes with a **directional prediction** (input
statistics, checkpoints, LoRA) are *experiments* — the slope moving the predicted
way is evidence. Axes without one (optimizer, and mostly architecture/loss across
uncontrolled models) are *correlations* — reading the dial and hoping. Ron's edge
is the causal, predicted axes; the artifact should lead with those.

## Q2 — What to show on it? (what you read off the dial)

For any fixed setting of an axis above, these are the readouts, weakest→strongest
as evidence:

1. **The transducer curve** — response magnitude vs **log** contrast. The raw
   picture.
2. **Log slope** in the suprathreshold region — the Weber–Fechner number. *The*
   scalar, but see the caveat: a single slope is under-determined.
3. **Model-comparison of the log fit** — how *log* is it really? Fit log vs
   linear vs Stevens power vs Naka-Rushton and report which wins + by how much.
   "It's log-shaped" is a claim you must *earn* against alternatives, not assume.
4. **Threshold / JND and the dipper** — behavioral surface; real discrimination is
   a **near-miss to Weber** (dipper function), not a clean line. Showing the dipper
   is more honest and more diagnostic than a bare slope.
5. **Across spatial frequency** — the 2017 contrast-constancy / "deblurring"
   result. Turns one slope into a *surface*; this is design requirement #5.
6. **Dual surface: behavior slope vs encoder-RDM slope, same model** — do the two
   instruments agree? This is the core differentiator (nobody runs it for
   contrast) and the only way to attack the identifiability problem.
7. **Noise / variability, not just mean shape** — magnitude-dependent vs constant
   noise (Cacioli follow-up 2604.04469,
   [papers/2604.04469-same-geometry-opposite-noise.md](papers/2604.04469-same-geometry-opposite-noise.md)).
   The geometry can match while the noise does not. Reading *only* the mean
   transducer is exactly the trap that paper exposes.

## Q3 — Is it a proxy for…? (answering the real question)

Ron's list of "is it a proxy for X" — each gets an honest valid/overclaim verdict.
The pattern: the log response is a **specific probe of prior-matching under a
range constraint**, and it is a *valid* proxy only for things that reduce to that.

- **"…which layers are useful?"** — **Partial / valid-with-scope.** It localizes
  *where the compressive, prior-matched code forms* across depth. That is not
  general "usefulness" (not saliency, not ablation importance). Sell it as
  *localization of the transducer*, never as a layer-importance ranking.

- **"…whether learning was good?"** — **Valid, in one specific sense.** Efficient
  coding says a well-trained encoder of natural statistics *should* grow a
  transducer matching the input CDF; failure to = the prior wasn't absorbed. So a
  missing/flat log response is a real red flag. **But** the converse fails: a clean
  log slope does **not** certify good learning, because it can be Weber-*mimicking*
  noise (linear encoding + scalar noise) rather than genuine log-encoding (the
  under-determination caveat, [theory doc](papers/theory-efficient-coding-log.md)
  + 2604.04469). Necessary-ish, never sufficient.

- **"…which architecture / loss / optimizer is best?"** — **Mostly overclaim.**
  There is no theory that maps log slope → model quality, so this cannot be a
  scalar "best model" score. It is *relative and domain-specific*: it ranks systems
  only on *dynamic-range compression / prior-matching*, not on capability. For
  optimizer specifically: **no directional prediction → not a measurement.** State
  this limit out loud; it is itself an interview-defense point.

- **"…one purpose that a second purpose conflicts with unless we do good?"** —
  **Yes, and this is the most interesting thread.** The log transducer is optimal
  for *one* objective: maximize information transmitted about a ~1/x prior through a
  bounded channel. Impose a *second* objective that needs **veridical / linear
  magnitude** (counting, precise comparison, calibrated measurement — Ron's own
  metrology world), and the two are in tension: compression that is optimal for
  transmission is *lossy for faithful magnitude*. A good system must *allocate* —
  keep the log code where the prior dominates and recover linearity where the task
  demands it. **Measuring where and how a VLM trades these off is itself a probe**
  of representational allocation. This is a real second-order design idea, not just
  the transducer; park it as a candidate extension, not a week-1 commitment.

## The one caveat that governs all of the above

**The log-linear response is under-determined as a metric.** A constant Weber
fraction / log geometry is consistent with (i) log-encoding + constant noise OR
(ii) linear encoding + scalar noise (2604.04469). Therefore:

- It is **not** a general model-quality score, a layer-usefulness ranking, or an
  optimizer benchmark. Pitching it as any of those fails the fairness bar.
- Its **real** power: a *specific, theory-grounded probe of dynamic-range
  allocation / prior-matching*, made into an **experiment** by turning the one knob
  theory predicts (input statistics, checkpoints, LoRA), and made **rigorous** by
  the dual surface + across-frequency space + noise axis.

That is the defensible use, and it is exactly what Design 1 already commits to —
this page just names *why* those five requirements are the right ones.

## What this implies for next steps

- Nothing here reopens the committed design; it sharpens the *claims* the design is
  allowed to make.
- The "input-statistics" axis is the cheapest strong causal experiment after the
  LoRA and is worth a line in the eventual statistics plan (05, step 3) — it is the
  purest test of the efficient-coding prediction.
- The "conflicting second purpose" (log vs veridical magnitude) is logged as a
  candidate extension, gated like crowding — not on the critical path.
- The **adaptation-vs-learning** reading of this same "input-statistics" axis, and
  a cheap **in-context adaptation** experiment that extends it to frontier APIs,
  are worked out in [08-adaptation-and-learning.md](08-adaptation-and-learning.md).
