"""Fitting the contrast response law.

Stated operationally: the mean absolute change in an end-computation DNN
representation (L1 distance from the gray reference) is a *linear* function of
the *log* of input contrast. Equivalently, log-spaced contrasts land at (near)
equal spacing in representation space.

Two things are fitted here, and only one of them is a claim.

``fit_log_linear`` fits ``L1 = a * log10(contrast) + b`` and reports R^2, per
spatial frequency and pooled -- the law as stated, taken at face value.

``fit_power_lambda`` asks the prior question: *what shape is the response?* It
fits the one-parameter family ``L1 = a + b * (c**lam - 1) / lam``, in which
``lam = 0`` is the log law and ``lam = 1`` is linear in raw contrast, and
returns the exponent with a confidence interval. This supersedes the
``logness`` statistic the module used to report (a race between the two
straight lines, scored by residual ratio), which was removed on 2026-07-26
because neither straight line describes the data: the trained net is convex in
``log c`` at 95 % of layer-frequency cells and the scrambled control is not
monotone at 41 % of them. The power family fits 0.92-0.998 everywhere the
straight lines did not. See ``wiki/Results.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import numpy as np

# Search bounds for the Box-Tidwell exponent. Wide enough to contain everything
# measured so far (a scrambled Caffe net reaches +2.75) with room either side;
# a fit that pins to a bound is reported through its confidence interval, which
# then spans the whole range and says so.
LAMBDA_LO, LAMBDA_HI = -3.0, 4.0


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

    The ``lam = 1`` corner of ``fit_power_lambda``, kept as a direct fit because
    "linear in raw contrast" is worth being able to ask for by name.
    ``slope``/``intercept`` are in contrast units here, so only ``r2`` is
    comparable with ``fit_log_linear``.
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
class PowerFit:
    """One layer-frequency fitted to ``D = a + b * (c**lam - 1) / lam``.

    ``lam`` is the whole answer: **0** the log law, **1** linear in raw
    contrast, **0.5** a square root, negative a saturating response. ``lo``/
    ``hi`` bracket it at 95 %; a fit that determines nothing returns the full
    search range, which is the honest report rather than a fabricated zero.
    """

    lam: float
    a: float
    b: float
    r2: float
    lo: float
    hi: float
    n: int

    def predict(self, contrast: np.ndarray) -> np.ndarray:
        return self.a + self.b * power_basis(np.asarray(contrast, dtype=np.float64), self.lam)


def power_basis(contrast: np.ndarray, lam: float) -> np.ndarray:
    """``(c**lam - 1) / lam``, continuous at ``lam = 0`` where it is ``ln c``.

    The Box-Cox/Box-Tidwell form rather than the bare ``c**lam``, and the
    difference matters: ``c**lam`` collapses to the constant 1 as ``lam -> 0``,
    so the log law would be unreachable. Subtracting 1 and dividing by ``lam``
    makes the limit ``ln c`` by L'Hopital, so a single parameter moves
    continuously from log through square root to linear and beyond.
    """
    log_c = np.log(contrast)
    if abs(lam) < 1e-9:
        return log_c
    return np.expm1(lam * log_c) / lam


def _profile_rss(basis: np.ndarray, response: np.ndarray) -> float:
    """Residual sum of squares after fitting ``a + b * basis`` by OLS.

    ``a`` and ``b`` enter linearly at any fixed ``lam``, so they are profiled
    out in closed form and the search over ``lam`` stays one-dimensional.
    """
    xm, ym = basis.mean(), response.mean()
    sxx = float(((basis - xm) ** 2).sum())
    if sxx <= 0.0:  # degenerate basis (lam pinned where c**lam is constant)
        return float(((response - ym) ** 2).sum())
    b = float(((basis - xm) * (response - ym)).sum()) / sxx
    resid = response - (ym + b * (basis - xm))
    return float((resid * resid).sum())


def _f_quantile_95(dof: int) -> float:
    """``F(1, dof, 0.95)`` = ``t(dof, 0.975)**2``, via Cornish-Fisher.

    scipy is not a dependency here. The expansion is within 0.02 % of the exact
    quantile by dof = 11 (the 14-point contrast grid), which is far finer than
    the interval it feeds.
    """
    if dof < 1:
        return float("inf")
    z = 1.959963984540054  # normal 0.975
    z3, z5, z7 = z ** 3, z ** 5, z ** 7
    t = (
        z
        + (z3 + z) / (4 * dof)
        + (5 * z5 + 16 * z3 + 3 * z) / (96 * dof ** 2)
        + (3 * z7 + 19 * z5 + 17 * z3 - 15 * z) / (384 * dof ** 3)
    )
    return float(t * t)


def fit_power_lambda(contrast: np.ndarray, response: np.ndarray) -> PowerFit:
    """Fit the exponent that says *where between log and linear* a response sits.

    This replaces the two-model comparison the module used to run. Racing
    ``D = a + b*log c`` against ``D = a + b*c`` and reporting which loses less
    only makes sense if one of them is right, and on this data neither is: the
    trained net is convex in ``log c`` at 95 % of layer-frequency cells and the
    scrambled control is not even monotone at 41 % of them. Both straight lines
    are wrong, so the comparison was deciding between two wrong answers -- and
    because it summed *squared* residuals, a single contrast point carried
    20-55 % of the verdict.

    Nesting both laws in one family removes the race. ``lam`` is measured, with
    an interval, on a scale that means something physically:

    ======  ==============================================
    ``lam``  response
    ======  ==============================================
    0        ``a + b*ln c``           -- the log law
    0.5      ``propto sqrt(c)``
    1        ``a + b*c``              -- linear in contrast
    < 0      saturating
    > 1      accelerating
    ======  ==============================================

    ``a`` and ``b`` are profiled out by OLS at each ``lam``, leaving a smooth
    1-D objective solved by a grid scan for the basin and golden section inside
    it -- no gradients, no scipy, float64 throughout.

    The interval is the profile-F set ``{lam : RSS(lam) <= RSS_min * (1 +
    F(1, n-3, 0.95)/(n-3))}``. It is what makes the fit self-reporting: pure
    noise returns the entire search range rather than a confident number, so an
    uninformative fit is visible instead of being averaged in.

    Read ``lam`` against ``r2``. The exponent locates a response only to the
    extent the family describes it at all; where R^2 sags, ``lam`` means less.
    """
    contrast = np.asarray(contrast, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    mask = contrast > 0  # c**lam undefined at zero contrast for lam <= 0
    c, y = contrast[mask], response[mask]
    if c.size < 3:
        raise ValueError("need at least three positive-contrast points to fit lambda")

    grid = np.linspace(LAMBDA_LO, LAMBDA_HI, 141)
    rss_grid = np.array([_profile_rss(power_basis(c, L), y) for L in grid])

    i = int(np.argmin(rss_grid))
    lo_b = grid[max(i - 1, 0)]
    hi_b = grid[min(i + 1, grid.size - 1)]
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    x1 = hi_b - inv_phi * (hi_b - lo_b)
    x2 = lo_b + inv_phi * (hi_b - lo_b)
    f1 = _profile_rss(power_basis(c, x1), y)
    f2 = _profile_rss(power_basis(c, x2), y)
    for _ in range(60):  # 0.618**60 * 0.1 is far below float64 resolution
        if f1 < f2:
            hi_b, x2, f2 = x2, x1, f1
            x1 = hi_b - inv_phi * (hi_b - lo_b)
            f1 = _profile_rss(power_basis(c, x1), y)
        else:
            lo_b, x1, f1 = x1, x2, f2
            x2 = lo_b + inv_phi * (hi_b - lo_b)
            f2 = _profile_rss(power_basis(c, x2), y)
    lam = 0.5 * (lo_b + hi_b)

    basis = power_basis(c, lam)
    rss = _profile_rss(basis, y)
    xm, ym = basis.mean(), y.mean()
    sxx = float(((basis - xm) ** 2).sum())
    b = float(((basis - xm) * (y - ym)).sum()) / sxx if sxx > 0 else 0.0
    a = float(ym - b * xm)
    tss = float(((y - ym) ** 2).sum())
    r2 = 1.0 - rss / tss if tss > 0 else float("nan")

    dof = c.size - 3  # a, b, lam
    threshold = rss * (1.0 + _f_quantile_95(dof) / max(dof, 1))
    # Bisect for each endpoint rather than reading them off ``grid``. Off the
    # grid, the interval can exclude its own point estimate: a minimum at 1.18
    # with 0.05 spacing reported [1.20, 1.20].
    ends = []
    for outer in (LAMBDA_LO, LAMBDA_HI):
        if _profile_rss(power_basis(c, outer), y) <= threshold:
            ends.append(outer)  # still inside at the bound: unbounded this side
            continue
        inside, outside = lam, outer
        for _ in range(60):
            mid = 0.5 * (inside + outside)
            if _profile_rss(power_basis(c, mid), y) <= threshold:
                inside = mid
            else:
                outside = mid
        ends.append(0.5 * (inside + outside))
    return PowerFit(
        lam=float(lam), a=a, b=b, r2=r2,
        lo=float(ends[0]), hi=float(ends[1]), n=int(c.size),
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
    readable in a way R^2 is not; decide with ``fit_power_lambda``, which
    measures the shape instead of choosing between two guesses at it.
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

    @cached_property
    def power_fits(self) -> list[PowerFit]:
        """The per-frequency Box-Tidwell fits behind ``lam``/``lam_r2``/``lam_ci``."""
        return [
            fit_power_lambda(self.contrasts, self.response[fi])
            for fi in range(self.response.shape[0])
        ]

    @property
    def lam(self) -> float:
        """Where this layer sits between the log law and linear in contrast.

        The **median** per-frequency exponent -- 0 log, 1 linear, 0.5 square
        root, negative saturating. Median rather than mean on purpose: a
        frequency whose response is non-monotone yields an arbitrary exponent,
        and no run should be swung by one of them. Nothing is dropped, so the
        count of such frequencies stays visible in ``lam_ci`` and ``lam_r2``.

        Free sanity check: ``features.0`` is a convolution, whose output must be
        linear in the grating's amplitude whatever the weights, so ``lam`` there
        must be ~1 regardless of checkpoint or scrambling. It measures 0.92-0.93
        across all four committed 45-tap runs, agreeing to 0.01.
        """
        return float(np.nanmedian([f.lam for f in self.power_fits]))

    @property
    def lam_r2(self) -> float:
        """Mean R^2 of the power fits -- how much ``lam`` is worth believing.

        Always read alongside ``lam``. The exponent locates a response only
        insofar as the family describes it: the scrambled ``IMAGENET1K_V1``
        control returns a log-like ``lam`` ~= 0.17 while fitting at 0.918
        against the trained net's 0.978, and that gap is what separates them.
        """
        return float(np.nanmean([f.r2 for f in self.power_fits]))

    @property
    def lam_ci(self) -> tuple[float, float]:
        """Median 95 % profile-F interval on ``lam``, as ``(lo, hi)``.

        Widens to the full search range where the data determine nothing, which
        is how an uninformative layer announces itself.
        """
        return (
            float(np.nanmedian([f.lo for f in self.power_fits])),
            float(np.nanmedian([f.hi for f in self.power_fits])),
        )

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
