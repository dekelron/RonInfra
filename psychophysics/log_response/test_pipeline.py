"""Offline self-tests for the log-response pipeline (no downloaded weights).

Verifies:
1. Stimulus generator: gray mean preserved, Michelson contrast correct, uniform
   reference.
2. Fitting: a synthetic log law is recovered (R^2 ~ 1); a linear-in-contrast
   signal is not; log-spaced contrasts land evenly under a true log law.
3. Metric plumbing: the experiment computes the Eq. 4 distance-of-means and the
   synthetic band-pass + compressive front-end yields a high-R^2 log-contrast
   response at its compressive "output" stage but not at the linear "energy"
   stage.

Run:  python -m pytest psychophysics/log_response/test_pipeline.py -q
  or: python psychophysics/log_response/test_pipeline.py
"""

from __future__ import annotations

import numpy as np

from .gratings import (
    GratingConfig,
    make_grating,
    make_reference,
    sample_gratings,
    PAPER_CONTRASTS,
)
from .fit import fit_log_linear, linear_spacing_uniformity
from .features import FeatureModel, SyntheticFrontEnd, l1_distance
from .experiment import run_experiment


def test_grating_mean_and_contrast():
    g = make_grating(contrast=0.5, frequency_cpi=8, size=128)
    # Mean luminance preserved at 0.5 (same as the gray reference).
    assert abs(g.mean() - 0.5) < 1e-3
    # Michelson contrast of an unclipped grating equals the requested contrast.
    lmax, lmin = g.max(), g.min()
    michelson = (lmax - lmin) / (lmax + lmin)
    assert abs(michelson - 0.5) < 1e-2


def test_reference_uniform():
    ref = make_reference(64)
    assert np.allclose(ref, 0.5)
    assert ref.std() == 0.0


def test_paper_contrast_grid():
    c = np.asarray(PAPER_CONTRASTS)
    assert len(c) == 14
    assert abs(c[0] - 1 / 128) < 1e-9
    assert abs(c[-1] - 1.0) < 1e-9
    # log-spacing: consecutive log-gaps are roughly constant above the low end
    gaps = np.diff(np.log10(c))[3:]
    assert gaps.std() / gaps.mean() < 0.25


def test_fit_recovers_log_law():
    contrasts = np.logspace(-2, 0, 14)
    y = 3.0 * np.log10(contrasts) + 7.0
    fit = fit_log_linear(contrasts, y)
    assert fit.r2 > 0.999
    assert abs(fit.slope - 3.0) < 1e-6


def test_linear_in_contrast_is_not_log_linear():
    contrasts = np.logspace(-2, 0, 14)
    fit = fit_log_linear(contrasts, contrasts.copy())
    assert fit.r2 < 0.95


def test_log_spaced_becomes_evenly_spaced():
    # A perfect log law has constant local log-slope even on a non-uniform
    # contrast grid, once gaps are normalised by the log-contrast spacing.
    contrasts = np.asarray(PAPER_CONTRASTS)
    logc = np.log10(contrasts)
    y = 3.0 * logc + 7.0
    assert linear_spacing_uniformity(y, logc) < 1e-9
    # Without normalisation, the paper's non-uniform grid looks uneven.
    assert linear_spacing_uniformity(y) > 0.1


def test_synthetic_frontend_shows_log_response():
    # small grid; a few more reps than 1 to average the phase-invariant energy.
    cfg = GratingConfig(size=96, frequencies_cpi=(3.5, 7, 14, 28))
    result = run_experiment(SyntheticFrontEnd(), cfg, repetitions=8, verbose=False)
    out = result.results["output"]
    energy = result.results["energy"]
    assert out.mean_r2 > 0.9  # compressive stage is log-linear in contrast
    assert out.mean_r2 > energy.mean_r2  # ... more so than the linear stage


class _PhaseSignedModel(FeatureModel):
    """A model whose single unit reads the center pixel (minus gray).

    For a full-field sinusoid this is a *signed* quantity uniform in the phase,
    so it cancels when activations are averaged over random phases first (Eq. 4)
    but not when per-stimulus absolute values are averaged. Orientation-agnostic.
    """

    def __init__(self):
        self.layers = ["signed"]

    def represent(self, image):
        h, w = image.shape[0] // 2, image.shape[1] // 2
        return {"signed": np.array([[float(image[h, w, 0]) - 0.5]])}


def test_experiment_uses_distance_of_means_not_mean_of_distances():
    cfg = GratingConfig(size=96, frequencies_cpi=(7.0,))
    reps = 60
    model = _PhaseSignedModel()
    result = run_experiment(model, cfg, repetitions=reps, seed=1, verbose=False)
    # Eq. 4 (mean first): the signed phase-dependent activity cancels -> small D.
    d_mean_first = result.surfaces["signed"][0, -1]  # highest contrast

    # Compare against the *other* ordering (mean of per-stimulus |.|), which does
    # not cancel and is therefore substantially larger.
    rng = np.random.default_rng(1)
    ref = model.represent(np.full((cfg.size, cfg.size, 3), 0.5))["signed"]
    per_stim = [
        l1_distance(model.represent(img)["signed"], ref)
        for img in sample_gratings(1.0, 7.0, reps, rng, size=cfg.size)
    ]
    d_mean_of_dists = float(np.mean(per_stim))

    assert d_mean_of_dists > 0.05  # the per-stimulus signal is real
    assert d_mean_first < 0.4 * d_mean_of_dists  # ... but it cancels under Eq. 4


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
