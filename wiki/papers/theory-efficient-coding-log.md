# Theory — why a log / Weber response emerges (efficient coding)

- **Cluster:** I — theory of the log response (the "why")
- **Sourcing:** `verified` (classical results); specific modern refs `snippet`

The theoretical spine of Design 1. Answers the interview question "why would a
network have a log contrast response at all?"

## The core argument (efficient coding)

- A channel with **bounded output range** encoding a stimulus drawn from prior
  `p(x)` maximizes transmitted information when its transfer function is the
  **CDF of `p(x)`** (histogram equalization).
- **Laughlin 1981** (*Z. Naturforsch.*) showed the fly LMC contrast-response
  curve **is** the CDF of natural-scene contrast — the empirical proof of
  concept.
- When `p(x)` is approximately **scale-invariant / power-law (~1/x)** — true of
  natural luminance, contrast, and symbolic magnitudes — the optimal transfer
  function is **logarithmic**. This is Fechner's intuition made rigorous.
- **Fisher-information / Bayesian** restatements (e.g. the least-error account,
  Royal Society 2023) reach the same conclusion: allocate representational
  resolution to match the prior → log for ~1/x priors.

## Log ⟺ Weber (two instruments, one claim)

- **Log encoding + constant internal noise ⇒ Weber's law** (ΔI/I = const). So a
  behavioural Weber fraction and a log encoder geometry are, in the standard
  story, the same phenomenon seen behaviourally vs. representationally.

## Mechanistic enablers in networks

- **Divisive normalization** (Heeger 1992) — the standard model of the
  **Naka-Rushton** contrast-response curve; compresses dynamic range ≈ log.
  LayerNorm / normalization layers play an analogous role in transformers.
- **Heavy-tailed training statistics** + a **contrast-invariance** objective push
  a recognition network toward a compressive transducer.
- Cacioli 2603.20642 checks the **power-law precondition** (integer frequencies)
  explicitly — same logic, symbolic domain.

## The critical caveat (our leverage)

- **Under-determination:** a constant Weber fraction is consistent with (i) **log
  encoding + constant noise** OR (ii) **linear encoding + scalar (Poisson-like)
  noise**. Behaviour alone cannot decide; geometry alone cannot decide the noise
  (see 2604.04469). "Log-linear response" is often asserted beyond what one
  surface licenses.
- Real contrast discrimination is a **near-miss to Weber** (dipper function), not
  a clean log line — method and range effects contaminate Stevens/Fechner fits.

## The noise axis (scalar variability) — the concrete test

The under-determination has a **measurable** discriminator: **scalar
variability**. Biological magnitude systems have representational noise that grows
with magnitude → **constant coefficient of variation (CV)**; that is what turns a
log transducer into Weber behaviour. Cacioli 2604.04469 measured this for
*numeric* magnitude in 7–8B transformers and found the **opposite**: CV
*decreases* with magnitude (α ≈ −0.19; 0/16 layers positive) — log geometry,
**non-biological noise**.

**No one has measured the noise signature of a *sensory* (contrast) transducer in
a VLM.** So our sharpest, falsifiable question is:

> Does the VLM contrast transducer carry biological **constant-CV** noise
> (scalar variability) or the transformer-like **decreasing-CV** signature — and
> does a **causal LoRA** move it?

## Why this hands us the contribution

Neighbors measure **one surface**. Our **dual behaviour+RDM surface in the same
model** + **causal LoRA** can test whether the log response is genuine
log-encoding or Weber-mimicking noise — mechanism disambiguation in the **sensory**
domain that no one has run. The scalar-variability test above makes it concrete
and pre-registrable. This is the strongest defense of Design 1.

## Key refs

Laughlin 1981; Fechner (1860); Heeger 1992 (divisive normalization); Wilson 1980
("A transducer function for threshold and suprathreshold human vision",
Biol. Cybern.); PNAS 2024 unified framework (see own file); Royal Society 2023
(Weber as least-error). Verify exact citations before the preprint.
