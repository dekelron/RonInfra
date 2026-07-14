# 02 — Prior Work & The Log-Response Result

## arXiv:1701.04674 — "Human perception in computer vision" (Dekel, 2017, solo)

Psychophysical evaluation of deep vision models (VGG-19, CaffeNet, GoogLeNet,
ResNet). Human perceptual sensitivities are mapped onto layer-wise computation;
DNN features are quantified as estimators of **perceptual loss**.

Abstract findings, mapped to layer depth:
- Sensitivity to **image changes** → *mid*-computation correlates.
- **Segmentation, crowding, shape** → *end*-computation correlates.
- Conclusion: properties of human perception are consistent with being a
  consequence of **architecture-independent visual learning**.

History: ICLR-2017 reject; never peer-reviewed; code exists but was never
released.

### ⭐ The log-response result (the priority result for this project)

The single result the artifact is built around:

- **Logarithmically-spaced contrast inputs become *linearly* spaced at the
  end-computation layers.**
- The average change in DNN representation scales with the **log** of the
  input-change magnitude.
- Output L1 distance vs. **log-contrast** fit at **R² ≈ 98%** (averaged across
  spatial frequencies).

In psychophysics terms: a **Weber–Fechner logarithmic transducer emerges** in a
network trained only for object recognition — never trained to produce it. This
is the *suprathreshold transducer shape* (how response magnitude scales with
contrast across the full range), i.e. contrast-constancy / Weber-law-of-
discrimination territory. It is **not** a detection threshold.

Why it anchors the artifact:
- **Parametric & quantitative** — one stimulus knob, a continuous law, a single
  fit statistic. Exactly the shape a psychometric function wants.
- **Ports to behavior cleanly** — forced-choice discrimination at parametrically
  varied contrast → fit the psychometric function → read the behavioral
  transducer slope and threshold.
- **Dual surface** — on open-weights VLMs it can be measured both behaviorally
  and in the vision-encoder embeddings (a direct 2017 replication), giving the
  LoRA experiment a clean causal target: *does fine-tuning move the transducer's
  slope / threshold?*

## arXiv:1907.09019 — Scintillating Grid in a DNN (Sun & Dekel, 2019)

Characterized VGG-19's **representational dissimilarity** as a function of
dot-whiteness in the Scintillating Grid illusion. Found a **non-monotonic**
response — dissimilarity rises then *falls* as whiteness increases — mirroring
the illusion, deviating from the expected monotonic relation.

First author Eric D. Sun was Ron's high-school mentee (now MIT faculty).

**Correction to a working assumption:** a version of this appears **published in
JOV (Journal of Vision)** — "ImageNet-trained deep neural networks exhibit
illusion-like response to the Scintillating grid." So the "never peer-reviewed"
description applies to the **2017** paper, not cleanly to 2019. This is an
interview-probe surface; state it accurately.

## Method distinction that governs the whole project

Both 2017 and 2019 measure **DNN internal representations** — representational
dissimilarity, layer-wise correlates. Neither runs **behavioral psychophysics**:
no adaptive staircases, no psychometric functions fit to a *subject's
responses*, no thresholds derived from behavior. The "subject" was the feature
vector; the tool was RDM correlation.

Therefore the new artifact is a **genuine methodological step beyond** the 2017
work, not a port of it. Framing that is both honest and stronger:

> 2017 mapped human perceptual phenomena onto DNN internal computation; this
> work closes the loop by measuring the model's own **behavior** with the rigor
> a human lab would use — and then manipulates it.

Do **not** say "I already did psychophysics on vision models in 2017" without
this qualifier — an interviewer will catch that 2017 was representation
analysis, not behavioral psychometrics.

## The RonInfra repo (this repo)

Small; **not** the 2017 experiment code. Contents:
- `Images/` — `resize_image.m` (AlexNet-style aspect-preserving resize),
  `ron_format_conversion.m`, `common_dir.m`.
- `Misc/` — `memoizeHDD.m`, `ron_cartesian_arrayfun.m`,
  `ron_cartesian_cellfun.m`.
- MATLAB utilities only. README: "My MATLAB code that others might find useful."

Implication for the "living repo" constraint: the 2017 experiment code lives
elsewhere. **Ron will supply it** (priority: the log-response / contrast
material). It will be ported/adapted into the new suite, satisfying the
constraint that the 2017 code ships only as a living part of this repo.

## Sourcing note

The session's network policy **blocks arxiv.org, Semantic Scholar, and direct
PDF fetch** (403 at the proxy). All paper details above are reconstructed from
web-search snippets — reliable for orientation, but exact figures/numbers should
be verified against the PDFs once Ron supplies them.
