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



def inverse_contrast_error(contrast: np.ndarray, response: np.ndarray) -> tuple[float, float]:
    """Median relative error when each law is inverted to predict contrast.

    Returns ``(linear, log)``. Both laws are fitted **forward** -- least squares
    in the response axis -- and only then inverted:

        linear  D = a + b*c        ->  c_hat = (D - a) / b
        log     D = a + b*log10 c  ->  c_hat = 10**((D - a) / b)

    Forward matters. Contrast is set exactly by the experiment and only ``D``
    carries noise, so regressing contrast on ``D`` would put the noise in the
    regressor and attenuate the slope. Fitting forward and inverting the fitted
    curve is calibration, not reverse regression.

    Reported as a *median relative* error for a measured reason: the two
    inversions have structurally different noise scaling in contrast units. The
    log law's inverse is exponential in ``D``, so its absolute contrast error
    grows in proportion to ``c`` -- across this grid, by a factor of ~550 (0.0009
    at the lowest contrast, 0.50 at the highest). A sum of squared contrast
    residuals would therefore be decided almost entirely by the single largest
    contrast, and would favour the linear law for that reason alone. The median
    is robust to it.

    This is a *reporting* quantity, not an arbiter. Absolute error in contrast
    versus relative error in contrast is the same arbitrary choice as ``c``
    versus ``log c`` as the regressor -- inverting relocates the choice rather
    than removing it. Use it because "recovers contrast to within 16%" is
    readable in a way R^2 is not; decide with ``logness``, which measures better
    against seed noise.
    """
    contrast = np.asarray(contrast, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    out = []
    for x in (contrast, np.log10(contrast)):
        slope, intercept = np.polyfit(x, response, 1)
        if not np.isfinite(slope) or slope == 0:
            out.append(float("nan"))
            continue
        predicted = (response - intercept) / slope
        if x is not contrast:  # fitted against log10 c, so invert the log too
            predicted = np.power(10.0, np.clip(predicted, -30.0, 30.0))
        out.append(float(np.median(np.abs(contrast - predicted) / contrast)))
    return out[0], out[1]


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

        ``(RSS_lin - RSS_log) / (RSS_lin + RSS_log)``, averaged over
        frequencies: **-1** a perfectly linear-in-contrast response, **+1**
        perfectly linear in log contrast, **0** a tie -- which for an
        unstructured response means noise, since neither law explains it.

        R^2 alone cannot answer this. Any monotone rising response scores high
        against *both* regressors -- conv1_1 reaches 0.94 against raw contrast
        while scoring 0.55 against log contrast, so its low log-R^2 reads as a
        weak response when the layer is in fact near-perfectly linear, which is
        what a linear filter must do. Differencing cancels the shared "it rises"
        variance and leaves only the shape.

        Both fits run on the same data, so their total sum of squares cancels
        and this is computable from the two R^2 directly::

            (r2_log - r2_lin) / (2 - r2_log - r2_lin)

        **The denominator is the whole point.** Normalising that same numerator
        by *total* variance instead -- plain ``r2_log - r2_lin``, which this
        property used to return -- cannot reach either endpoint: ``c`` and
        ``log c`` are monotone transforms of one another and stay strongly
        correlated, so when ``r2_log`` hits 1.0 the linear fit still holds
        ``r2_lin = 0.736``. A perfect log response scored only **+0.264** on
        this grid, and that ceiling moved with the contrast grid (**0.294**
        linearly spaced), so values were not comparable across grids. It was
        also unbounded in practice, since curvature reads as "anti-log":
        ``D = c^2`` scored -1.589 and a scrambled net reached -0.381, both past
        a perfect straight line's -0.264. Dividing by the residual budget fixes
        all three -- endpoints exact, grid-free, and inside [-1, +1] by
        construction.

        Verified against ground truth in ``test_metric_calibration``: perfect
        log +1.000, perfect linear -1.000, pure noise 0.000 +/- 0.05, on both
        the log-spaced and the linear grid.

        The cost is variance -- the denominator shrinks as both fits improve, so
        this is noisier than the difference form on a clean response. That is
        the right trade: a calibrated statistic with an honest error bar beats a
        quiet one whose scale is wrong by 3.8x.

        Residuals are taken in the **response** axis, not the contrast axis.
        Inverting each law to predict contrast and differencing *there* is the
        more natural-sounding comparison, but the two inversions are not
        symmetric -- the log law's inverse exponentiates response noise while
        the linear law's divides it -- so their errors differ by orders of
        magnitude, the ratio pins to +/-1, and the zero stops meaning anything
        (pure noise scores -0.985). Same axis, same noise, symmetric comparison.
        ``inverse_contrast_error`` reports the contrast-axis view separately,
        where its units are the point.

        Both models carry two parameters on identical data, so AIC and BIC
        differences reduce to the plain likelihood ratio -- no complexity
        penalty enters the comparison.
        """
        out = []
        for f, lin in zip(self.per_frequency, self._linear_fits):
            denom = 2.0 - f.r2 - lin.r2
            out.append((f.r2 - lin.r2) / denom if denom > 0 else float("nan"))
        if not out or np.all(np.isnan(out)):
            return float("nan")
        return float(np.nanmean(out))

    @property
    def logness_r2diff(self) -> float:
        """The pre-2026-07-26 ``logness``: plain ``R2_log - R2_linear``.

        Kept because every run committed before that date quotes it, and the
        surfaces in ``result.npz`` re-fit into either statistic. Do not use it
        for new claims -- its endpoints are unreachable and its scale depends on
        the contrast grid. See ``logness`` for the measured numbers.
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
    def contrast_error(self) -> tuple[float, float]:
        """``(linear, log)`` median relative error of inverse contrast prediction.

        Averaged over frequencies. Physical units: 0.16 means the law recovers
        the stimulus contrast to within 16 %.
        """
        pairs = [
            inverse_contrast_error(self.contrasts, self.response[fi])
            for fi in range(self.response.shape[0])
        ]
        return (
            float(np.nanmean([p[0] for p in pairs])),
            float(np.nanmean([p[1] for p in pairs])),
        )

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
