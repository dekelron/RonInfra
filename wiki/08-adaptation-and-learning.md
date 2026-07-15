# 08 — Adaptation vs. learning: two routes to the log transducer

Status: **thinking / directions note** (2026-07-15). Organizes and red-teams a
batch of ideas on **fast adaptation vs. slow learning** as competing/complementary
origins of the log-contrast transducer, and the spin-offs ("adaptation as a
pre-fine-tune stage", "a normalization that matches more than the mean", "log
response as a learning-quality check"). **Not a design change** — Design 1 is
still committed ([04-design.md](04-design.md)). The point is to extract the *one
thing that is cheap, fast, in-scope, and genuinely new* — the stated goal is to
**show value quickly and strongly**, not to open a second research program.

## 0. The one-paragraph version

The log transducer can arise **two ways**. (a) **Learning:** weights slowly
absorb the input distribution's statistics until the encoder's transfer function
matches the prior's CDF (efficient coding,
[theory doc](papers/theory-efficient-coding-log.md)). (b) **Adaptation:** a fast,
input-history-dependent re-centering of the operating point on recent input, with
**no weight change** — the test-time analog. A trained network plainly has (a);
the question is whether it has anything like (b), or must reach the same place by
(a) alone. Most of the downstream ideas below are interesting but **out of the
4–6 week lane**; one is not.

## 1. Sharpening the core distinction (and correcting it)

The framing: a network has no built-in adaptation mechanism, so it can only reach
the log response by **learning it explicitly**; the normalization in networks
isn't the same thing — it just divisively normalizes the mean/gain per unit,
which is *not* the same as encoding the log transfer.

This is **mostly right, but the strong version ("networks have no such
mechanism") is false and worth fixing before it becomes a claim:**

| | Fast adaptation (gain control) | Slow per-unit re-centering | Task learning |
|---|---|---|---|
| **Timescale** | per-input (test time) | gradual, persistent | training run |
| **Substrate** | activity-dependent divisive gain (LayerNorm / divisive norm) | none standard | weight updates on a loss |
| **In a trained net?** | **yes** — normalization *is* input-dependent gain control | **no** — nothing keeps a unit's dynamic range populated independent of the loss | yes (this is the whole thing) |
| **What it sets** | the gain / operating point | the operating range over time | **the transfer shape (the log)** |

So the honest split is: networks **do** have the *fast, input-dependent gain
control* component (normalization is exactly this), and they **lack** any *slow,
persistent per-unit re-centering* that keeps a unit's dynamic range populated
regardless of the loss. The real point survives in a cleaner form:

> **Divisive normalization matches the mean/variance (gain); it does not, by
> itself, install the *shape* of the log transfer.** The network still has to
> *learn* the transfer that matches the prior. Normalization enables the
> compression; learning chooses it.

One caveat that matters below: normalization **can** be trained to match more
than the mean — **GDN (Ballé, Laparra & Simoncelli 2016)** is a generalized
divisive normalization *optimized to Gaussianize the joint statistics of natural
images*, i.e. it deliberately matches higher-order structure for optimal coding.
So "normalization only fixes the mean" is true of LayerNorm, **not** of the whole
family. This directly pre-empts idea B2 below.

## 2. Direction A — "maybe learning *is* the second account of adaptation"

The idea: adaptation is usually cast as *matching the statistics of the input
distribution*; maybe if a network's features are roughly **orthogonal to the
tasks**, then plain task-learning in a given environment reproduces the same
statistics-matching — so "adaptation" and "learning" are two names for one thing.
Stated that way it *"sounds trivial"* — is it?

**Critical assessment — half real, half a category error:**

- **The decisive split is timescale.** A fast, test-time re-centering (change the
  input distribution for a few samples, the response re-centers, and it reverts)
  is **reversible and input-history-specific**; slow weight-learning cannot
  produce a re-tunable-in-a-few-samples, reversible effect. So the two are
  **answers to different questions**:
  - *Why is the steady-state transducer log-shaped at all?* → **learning /
    efficient coding.** Here the idea is right: chronic exposure statistics,
    absorbed into weights, is a legitimate (and arguably the primary) account of
    the *baseline* curve.
  - *Why does the curve re-center after a biased run of inputs?* → **fast
    adaptation.** Learning has nothing to say here.
- Stated as "learning is *an alternative reason* for adaptation," the claim
  **overreaches**. Stated as "learning fixes the *baseline* transducer to the
  input statistics, and adaptation is a separate fast re-centering on top," it is
  correct and already the mainstream efficient-coding view — which is why it
  "sounds trivial." The non-trivial residue is the **"features orthogonal to
  tasks"** conjecture, and that is **under-specified**: what does orthogonality
  mean operationally, and what prediction does it make? Until that's a testable
  statement it's an intuition, not a result.
- **Lane check:** as a general claim about *learning dynamics* it is a much
  larger, separate bet that a 4–6 week measurement artifact **cannot** discharge
  and that the fairness bar ([01-mission.md](01-mission.md)) would flag. **Park it
  as an interpretation/discussion point, never as a deliverable claim.**

**The salvageable, in-scope version.** There is a way to run "adaptation" inside
the committed design *without training*: a chat VLM has **no weight plasticity at
inference**, so any adaptation-like re-centering it shows must come from
**context**. That gives a cheap, novel, API-only experiment — see §4.

## 3. Direction B — importing an adaptation-style mechanism *into* networks

Three concrete options; **all three are further from the committed spine than
they look.**

### B1 — Adaptation as a pre-fine-tune stage ("revive the flat units")

Before fine-tuning, walk the network, find units whose responses are **too flat**,
and **expand** them, opening up a representation that is otherwise partly dead.
Stated risk: *you may expand units that should stay dead.*

- **Not new as a method, and the named risk is the known failure mode.** This is
  the **dead / dormant-unit** literature: dying-ReLU, and specifically **ReDo —
  "The Dormant Neuron Phenomenon in Deep RL" (Sokar, Agarwal, Castro & Evci,
  ICML 2023, arXiv:2302.12902)**, which periodically detects τ-dormant units and
  *recycles* them by re-initializing incoming weights and zeroing outgoing ones.
  "Expand flat units before adapting" is a variant of exactly this.
- The **do-it-as-a-deliberate-pre-fine-tune-pass** framing is a fresh spin, but
  the mechanism and its pitfall are charted territory.
- **Verdict:** a **training-methods** contribution needing a benchmark to show
  value; it **dilutes** the tight measurement story and needs infra we don't have
  (no from-scratch training; see [01-mission.md](01-mission.md) gaps). **Off
  critical path.** Cite ReDo if it ever surfaces in the paper's discussion.

### B2 — Replace divisive normalization with a "match-more-statistics" layer

A normalization that doesn't just match mean/std but matches **most statistics**,
to *encourage/enforce* a representation that satisfies the reason for the log
response — matching the input's power-law and, for a ~1/x prior, landing on the
log-optimal transfer.

- **Intellectually the best of the three — and partly already built.** This is
  essentially **GDN (Ballé et al. 2016 / end-to-end optimized image compression,
  ICLR 2017, arXiv:1611.01704)**: a learned divisive normalization that
  Gaussianizes natural-image joint statistics for optimal coding — a
  normalization primitive whose *job* is prior-matching, not just mean/gain. So
  the idea is directionally validated **and** partially pre-empted. The remaining
  delta (higher-order moments, or a training-time regularizer that explicitly
  pulls the transducer toward the efficient-coding log optimum) is real but
  incremental over GDN.
- **Scope reality:** designing, implementing, training, and benchmarking a new
  normalization layer is a **multi-month architecture project** requiring
  from-scratch or large-scale training — outside the time, budget, compute
  ceiling (LoRA on ~7B), and stated skill profile. **Not in the 4–6 week
  artifact.** Log it as a genuine **long-horizon** direction; cite GDN as the
  starting point, not a blank slate.

### B3 — Log response as a proxy that "the network learned well"

If learning went badly, then *for different kinds of features* the clean log
response would not appear.

- **Already answered — this is [07-log-response-as-instrument.md](07-log-response-as-instrument.md)
  Q3 "…whether learning was good?"** The verdict there governs: **necessary-ish,
  not sufficient.** A missing/flat log response is a real red flag (a well-trained
  encoder of the input statistics *should* grow the transducer); but a clean log
  slope does **not** certify good learning, because it can be **mimicked** by a
  linear encoding plus magnitude-dependent noise rather than genuine log-encoding
  — the under-determination caveat
  ([theory doc](papers/theory-efficient-coding-log.md), 2604.04469).
- The one **new** wrinkle — check the log response across **many feature
  dimensions**, not one — is a mild upgrade: a *log-response audit across
  features* is a cheap, richer diagnostic than a single contrast slope, and its
  **consistency** is more informative than any one fit. But it inherits the same
  sufficiency caveat and is still a **proxy**, not a certificate. Worth one line
  in the eventual diagnostics, no more.

## 4. What is actually actionable now (the fast-value item)

Filtering everything above through *cheap · fast · in-scope · genuinely new*, two
things survive, and they are the same lever seen twice:

1. **Input-statistics / learning axis (already flagged in
   [07](07-log-response-as-instrument.md) as the cheapest strong causal
   experiment after the LoRA).** The adaptation framing gives it a crisper story:
   the log transducer is **learned, not built-in** — prove it by controlling the
   training/adaptation statistics and watching the slope track the prior's CDF.
   This is the purest test of the efficient-coding prediction and it now has a
   one-line hook ("built or learned?").

2. **In-context ("test-time") adaptation in a chat VLM — new, API-only, days not
   weeks.** Because a chat VLM has *no inference-time weight plasticity*, any
   shift in its contrast transducer after a **context** of biased-contrast images
   is a pure **adaptation-by-context** effect, at **zero training cost**.
   Concretely: measure the behavioral transducer (Design 1's 2-AFC contrast
   discrimination) **cold**, then **after** priming the context with a run of
   high- (or low-) contrast images, and test for a **predicted, reversible**
   re-centering.
   - **Why it's worth it:** it costs only API calls (fits the budget), runs on the
     **frontier API models** where the dual-surface/LoRA legs *cannot* reach (see
     [04-design.md](04-design.md) req. 4), it is **novel** (nobody has run
     in-context contrast adaptation in a VLM), and it is a **direct instantiation
     of the adaptation idea** — the one thread that lands inside the committed
     design instead of outside it.
   - **Discipline:** pre-register the direction and effect-size bar (same hygiene
     as the LoRA, [05-open-questions.md](05-open-questions.md) Q5). Frame it as a
     **candidate leg tied to Design 1**, gated like crowding — not a week-1
     commitment, and explicitly **behavioral-only** (no mechanism claim from a
     black-box API).

## 5. Bottom line / verdict table

| Idea | Novel? | In 4–6 wk scope? | Disposition |
|---|---|---|---|
| Adaptation vs. learning as **two routes** to the transducer | Framing, not a result | Yes (as narrative) | **Keep** — sharpens "built or learned?"; correct the "no mechanism" overstatement |
| A1 — learning is an **alt account of adaptation** | Overreaches (timescale) | No (general dynamics claim) | **Park** as interpretation only; not a deliverable |
| A2 — **in-context adaptation** in a VLM | **Yes** | **Yes (API-only, cheap)** | **Promote** to candidate Design-1 leg (§4) |
| B1 — adaptation **pre-fine-tune** revival | No (≈ ReDo / dormant units) | No (methods + infra) | Off critical path; cite ReDo if discussed |
| B2 — **better-than-DN normalization** (match higher stats) | Partly (≈ GDN) | No (multi-month arch) | Long-horizon; cite Ballé/GDN |
| B3 — log response as **learning-quality proxy** | No (already 07 Q3) | Yes (as diagnostic) | Necessary-ish, not sufficient; add cross-feature audit as one line |

**Nothing here reopens the committed design.** The net effect: one correction
(nets *do* have fast normalization-based gain control; they lack slow persistent
re-centering), one promoted cheap experiment (in-context contrast adaptation on
frontier APIs), and three ideas explicitly **parked with their prior art named**
so they don't quietly re-enter as if novel.

## Sourcing note

New citations on this page — **Ballé, Laparra & Simoncelli 2016** (GDN / "Density
Modeling of Images using a Generalized Normalization Transformation"; end-to-end
optimized image compression, arXiv:1611.01704) and **Sokar et al. 2023** (ReDo,
dormant units, arXiv:2302.12902) — were confirmed by **web-search snippets**
(2026-07-15), reliable for orientation. Verify exact refs against PDFs before any
preprint (arxiv/PDF fetch is blocked by session network policy — see
[02-prior-work.md](02-prior-work.md)).
