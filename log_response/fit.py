"""Fitting the log-contrast response law.

Stated operationally: the mean absolute change in an end-computation DNN
representation (L1 distance from the gray reference) is a *linear* function of
the *log* of input contrast. Equivalently, log-spaced contrasts land at (near)
equal spacing in representation space. The quality of that linear fit reaches
R^2 ~= 0.98 at the final ("prob") layer, averaged across spatial frequencies.

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


def fit_linear_in_contrast(contrast: np.ndarray, response: np.ndarray) -> LinearLogFit:
    """Least-squares fit of ``response = slope * contrast + intercept``.

    The null the log law is judged against: same data, same parameter count,
    raw contrast instead of its logarithm. ``slope``/``intercept`` are in
    contrast units here, so only ``r2`` is comparable with ``fit_log_linear``.
    """
    contrast = np.asarray(contrast, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    if contrast.size < 2:
        raise ValueError("need at least two points to fit")
    slope, intercept = np.polyfit(contrast, response, 1)
    pred = slope * contrast + intercept
    ss_res = float(np.sum((response - pred) ** 2))
    ss_tot = float(np.sum((response - np.mean(response)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return LinearLogFit(
        slope=float(slope), intercept=float(intercept), r2=r2, n=int(contrast.size)
    )


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
        """Mean per-frequency R^2 -- the headline log-linearity statistic.

        NaN R^2 (a frequency with constant response) is excluded rather than
        propagated, matching the spacing-CV column in the experiment report.
        """
        return float(np.nanmean([f.r2 for f in self.per_frequency]))

    @property
    def logness(self) -> float:
        """Does the response follow log contrast or raw contrast? In [-1, +1].

        ``R2_log - R2_linear``, averaged over frequencies: -1 is a perfectly
        linear-in-contrast response, +1 perfectly linear in log contrast, 0
        either a tie or noise.

        R^2 alone cannot answer this. Any monotone rising response scores high
        against *both* regressors -- conv1_1 reaches 0.94 against raw contrast
        while scoring 0.55 against log contrast, so its low log-R^2 reads as a
        weak response when the layer is in fact near-perfectly linear, which is
        what a linear filter must do. Differencing cancels the shared "it rises"
        variance and leaves only the shape.

        Normalised by total variance rather than by the residuals: the ratio
        form ``(RSS_lin - RSS_log)/(RSS_lin + RSS_log)`` is the same comparison
        and equals ``-tanh(dAIC/2n)``, but its denominator shrinks as both fits
        improve, which measured 2.4x noisier across scramble seeds.

        Both models carry two parameters on identical data, so AIC and BIC
        differences reduce to the plain likelihood ratio -- no complexity
        penalty enters the comparison.
        """
        return float(np.nanmean([
            f.r2 - lin.r2 for f, lin in zip(self.per_frequency, self._linear_fits)
        ]))

    @property
    def fit_quality(self) -> float:
        """Best R^2 of either law, in [0, 1] -- the companion to ``logness``.

        ``logness`` is 0 both when the two laws fit equally well and when
        neither fits at all; one scalar cannot separate those. This says which:
        near 1 the response is well described and the laws are simply hard to
        tell apart, near 0 nothing describes it.
        """
        return float(np.nanmean([
            max(f.r2, lin.r2)
            for f, lin in zip(self.per_frequency, self._linear_fits)
        ]))

    @property
    def _linear_fits(self) -> list[LinearLogFit]:
        """The same OLS against raw contrast, for the ``logness`` comparison."""
        return [
            fit_linear_in_contrast(self.contrasts, self.response[fi])
            for fi in range(self.response.shape[0])
        ]


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
    actual log-contrast spacing -- important because the 14 contrasts are only
    *approximately* even in log, so raw D-gaps would look uneven even for a
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
