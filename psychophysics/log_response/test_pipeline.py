"""Offline self-tests for the log-response pipeline.

These run without any downloaded model weights. They verify:

1. Stimulus generator: gray mean is preserved, Michelson contrast is correct,
   the reference is uniform.
2. Fitting: a synthetically generated log law is recovered with R^2 ~ 1, and a
   deliberately non-log (linear-in-contrast) signal is NOT.
3. End to end: the synthetic band-pass + compressive front-end reproduces the
   qualitative 2017 phenomenology -- a high-R^2 log-contrast response at its
   compressive "output" stage.

Run:  python -m pytest psychophysics/log_response/test_pipeline.py -q
  or: python psychophysics/log_response/test_pipeline.py
"""

from __future__ import annotations

import numpy as np

from .gratings import GratingConfig, make_grating, make_reference
from .fit import fit_log_linear, linear_spacing_uniformity
from .features import SyntheticFrontEnd
from .experiment import run_experiment


def test_grating_mean_and_contrast():
    g = make_grating(frequency_cpi=8, contrast=0.5, size=128)
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


def test_fit_recovers_log_law():
    contrasts = np.logspace(-2, 0, 12)
    # response is exactly linear in log10(contrast)
    y = 3.0 * np.log10(contrasts) + 7.0
    fit = fit_log_linear(contrasts, y)
    assert fit.r2 > 0.999
    assert abs(fit.slope - 3.0) < 1e-6


def test_linear_in_contrast_is_not_log_linear():
    contrasts = np.logspace(-2, 0, 12)
    # A signal linear in contrast (not log) should fit the log law poorly.
    y = contrasts.copy()
    fit = fit_log_linear(contrasts, y)
    assert fit.r2 < 0.95


def test_log_spaced_becomes_evenly_spaced():
    # Under a perfect log law, log-spaced contrasts give evenly spaced responses.
    contrasts = np.logspace(-2, 0, 12)
    y = 3.0 * np.log10(contrasts) + 7.0
    cv = linear_spacing_uniformity(y)
    assert cv < 1e-6


def test_synthetic_frontend_shows_log_response():
    cfg = GratingConfig(size=96, frequencies_cpi=(4, 8, 16, 32))
    result = run_experiment(SyntheticFrontEnd(), cfg, verbose=False)
    # The compressive "output" stage should be strongly log-linear in contrast.
    out = result.results["output"]
    assert out.mean_r2 > 0.9
    # ... and clearly more log-linear than the pre-compression linear "energy".
    energy = result.results["energy"]
    assert out.mean_r2 > energy.mean_r2


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)} tests passed")


if __name__ == "__main__":
    _run_all()
