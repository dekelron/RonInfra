"""Fitting the log-contrast response law.

The 2017 result, stated operationally: the mean absolute change in an
end-computation DNN representation (L1 distance from the gray reference) is a
*linear* function of the *log* of input contrast. Equivalently, log-spaced
contrasts land at (near) equal spacing in representation space. The reported
quality of that linear fit is R^2 ~= 0.98 at the final ("prob") layer, averaged
across spatial frequencies.

This module fits ``L1 = a * log10(contrast) + b`` and reports R^2, both per
spatial frequency and pooled across frequencies.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class LinearLogFit:
    slope: float
    intercept: float
    r2: float
    n: int

    def predict(self, contrast: np.ndarray) -> np.ndarray:
        return self.slope * np.log10(contrast) + self.intercept


def fit_log_linear(contrast: np.ndarray, response: np.ndarray) -> LinearLogFit:
    """Least-squares fit of ``response = slope * log10(contrast) + intercept``.

    Zero/negative contrasts are dropped (log undefined). R^2 is the ordinary
    coefficient of determination.
    """
    contrast = np.asarray(contrast, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    mask = contrast > 0
    x = np.log10(contrast[mask])
    y = response[mask]
    if x.size < 2:
        raise ValueError("need at least two positive-contrast points to fit")
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return LinearLogFit(slope=float(slope), intercept=float(intercept), r2=r2, n=int(x.size))


@dataclass
class LayerLogResult:
    """Log-response summary for one layer.

    ``response`` is the L1 distance surface indexed [freq_idx, contrast_idx].
    """

    layer: str
    contrasts: np.ndarray
    frequencies: np.ndarray
    response: np.ndarray  # (n_freq, n_contrast)
    per_frequency: list[LinearLogFit]
    pooled: LinearLogFit

    @property
    def mean_r2(self) -> float:
        """Mean per-frequency R^2 -- the statistic the paper reports."""
        return float(np.mean([f.r2 for f in self.per_frequency]))


def summarise_layer(
    layer: str,
    contrasts: np.ndarray,
    frequencies: np.ndarray,
    response: np.ndarray,
) -> LayerLogResult:
    """Fit the log law per frequency and pooled for one layer's L1 surface."""
    contrasts = np.asarray(contrasts, dtype=np.float64)
    frequencies = np.asarray(frequencies, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    per_freq = [fit_log_linear(contrasts, response[fi]) for fi in range(response.shape[0])]

    # Pooled fit: to remove the per-frequency gain offset before pooling, we fit
    # each frequency's response after subtracting its own mean (a within-frequency
    # centring), which mirrors "averaged across spatial frequencies".
    centred = response - response.mean(axis=1, keepdims=True)
    pooled_contrast = np.tile(contrasts, response.shape[0])
    pooled_response = centred.reshape(-1)
    pooled = fit_log_linear(pooled_contrast, pooled_response)

    return LayerLogResult(
        layer=layer,
        contrasts=contrasts,
        frequencies=frequencies,
        response=response,
        per_frequency=per_freq,
        pooled=pooled,
    )


def linear_spacing_uniformity(
    response_row: np.ndarray, log_contrast: np.ndarray | None = None
) -> float:
    """How constant is the local log-contrast slope of the response?

    Under a perfect log law ``D = a + b*log c`` the per-interval slope
    ``ΔD / Δlog c`` is constant, so its coefficient of variation is 0 (lower is
    more "log-linear"). Pass ``log_contrast`` so the D-gaps are normalised by the
    actual log-contrast spacing -- important because the paper's 14 contrasts are
    only *approximately* even in log, so raw D-gaps would look uneven even for a
    perfect log law. If ``log_contrast`` is omitted, even spacing is assumed.
    """
    response_row = np.asarray(response_row, dtype=np.float64)
    diffs = np.diff(response_row)
    if log_contrast is not None:
        dlog = np.diff(np.asarray(log_contrast, dtype=np.float64))
        diffs = diffs / dlog  # per-unit-log-contrast slope
    m = np.mean(diffs)
    if m == 0:
        return float("nan")
    return float(np.std(diffs) / abs(m))
