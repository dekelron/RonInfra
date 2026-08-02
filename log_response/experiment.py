"""Run the log-contrast-response experiment end to end.

For each layer and each (contrast, frequency) cell the primary metric is the
paper's: a **distance of means**, not a mean of distances:

    mu_i(c,f) = mean over the 250 random images of unit i's activation
    D(c,f)    = mean_i | mu_i(c,f) - a_i(gray) |

Averaging activations across the random orientation/phase draws happens BEFORE
the absolute value, so phase/orientation-specific activity cancels first (see
wiki/Method.md, Jensen's inequality note).

The **other ordering** is recorded alongside it, for every run:

    D_mod(c,f) = mean over the 250 images of  mean_i | a_i(x_r) - a_i(gray) |

It is not a better metric, it is a different one, and the difference is
load-bearing. Phase ~ U[0,2pi) makes E[grating] = gray exactly, so D has
population value **zero** at any layer that is affine in the input and what a
finite run measures there is 1/sqrt(reps) sampling noise. D_mod takes the
absolute value first, so nothing cancels and shallow layers carry real signal.
Reporting both is what separates "this layer responds linearly to contrast"
from "this layer's signal is below the metric's floor".

It is free: the per-image distance reduces to a scalar immediately, so it costs
one accumulator *number* per layer rather than one accumulator array, and the
forward pass dominates the arithmetic either way. Runs saved before this
existed simply carry no D_mod, and load with it set to None.

A **third** quantity rides along on the same terms, and it is an instrument for
one specific hypothesis rather than another metric:

    G(c,f) = mean over the images of the fraction of units i with
             sign(a_i(x_r)) != sign(a_i(gray))

A ReLU net is piecewise linear, so about the fixed gray operating point the
grating is a perturbation gray + c*g and, *while no rectifier changes state*,
D = |J.(c*g)| = c*|J.g| exactly -- linear in contrast at any depth, i.e.
lambda = 1. On that reading lambda < 1 is the signature of gates actually
switching with contrast, and the log law is what emerges once they do. G counts
the switches, so it turns the reading into a measurement: lambda should leave 1
exactly where G leaves 0.

It needs no pre-activation tap. A ReLU's output is positive iff its input is,
so a post-ReLU tap's sign *is* the gate state, and a conv tap's sign is the gate
state of the ReLU that follows it. Where no hard gate exists (GELU, LayerNorm, a
logit) G is still well defined -- it is then the fraction of units the
perturbation carries across zero, not a gate count -- so read it against what
the tap is.

The gray gate-open fraction is recorded per layer alongside it. That is the
*operating point* itself: the surviving explanation for every lambda in this
repo is where units sit relative to their nonlinearity, and it has been argued
across 29 architecture/checkpoint combinations without once being measured.

Produces, per layer: the L1 surface D(freq, contrast); per-frequency and pooled
linear fits of D vs log10(contrast) with R^2; and optional figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
import json
import math
import os
import warnings
import numpy as np

from .gratings import GratingConfig, sample_gratings, reference_rgb
from .features import FeatureModel, l1_distance
from .fit import (
    LayerLogResult,
    summarise_layer,
    linear_spacing_uniformity,
    power_basis,
)


@dataclass
class ExperimentResult:
    config: GratingConfig
    repetitions: int
    layers: list[str]
    # layer -> (n_freq, n_contrast) L1 surface
    surfaces: dict[str, np.ndarray]
    results: dict[str, LayerLogResult]
    # layer -> (n_freq, n_contrast) standard error of D, or None if not measured.
    noise: dict[str, np.ndarray] | None = None
    # layer -> (n_freq, n_contrast) mean-of-distances surface (the other order of
    # operations; see the module docstring). None for runs saved before it
    # existed -- every committed directory from before 2026-07-27.
    mean_of_distances: dict[str, np.ndarray] | None = None
    # layer -> (n_freq, n_contrast) fraction of units whose sign differs from
    # gray, and layer -> fraction of units positive at gray. The gate-flip
    # instrument; see the module docstring. None for runs saved before
    # 2026-08-02.
    gate_flips: dict[str, np.ndarray] | None = None
    gate_open: dict[str, float] | None = None

    def flip_fraction(self, layer: str, ci: int = -1) -> float:
        """Median over frequencies of G at one contrast (default: the highest).

        The single number to read a tap's lambda against: the perturbation
        argument says lambda = 1 is *forced* wherever this is 0, so a tap at
        lambda ~ 1 with a large flip fraction, or lambda far from 1 with none,
        is the observation that would break it.
        """
        if self.gate_flips is None:
            return float("nan")
        return float(np.median(self.gate_flips[layer][:, ci]))

    @cached_property
    def mod_results(self) -> dict[str, LayerLogResult] | None:
        """The same fits, applied to the mean-of-distances surfaces."""
        if self.mean_of_distances is None:
            return None
        return {
            name: summarise_layer(
                name,
                contrasts=self.config.contrast_array,
                frequencies=self.config.frequency_array,
                response=self.mean_of_distances[name],
            )
            for name in self.layers
        }

    def _chi2_per_dof(self, layer: str) -> float:
        """Best law's chi-squared per degree of freedom, needs ``noise``.

        The absolute-scale companion to ``lam``: ``lam`` says *where* between
        the two laws a response sits, and ``lam_r2`` how well the family
        describes it, but only measured error says whether the residual left
        over is consistent with noise. ~1 means the law fits within measurement
        error; much greater than 1 means real structure is unaccounted for.
        """
        res = self.results[layer]
        sigma = self.noise[layer]
        best = []
        for fi in range(res.response.shape[0]):
            y, s = res.response[fi], sigma[fi]
            if not np.all(np.isfinite(s)) or np.any(s <= 0):
                continue
            c = res.contrasts
            chis = []
            for pred in (
                np.polyval(np.polyfit(np.log10(c), y, 1), np.log10(c)),
                np.polyval(np.polyfit(c, y, 1), c),
            ):
                chis.append(float(np.sum(((y - pred) / s) ** 2) / max(1, len(y) - 2)))
            best.append(min(chis))
        return float(np.mean(best)) if best else float("nan")

    def report(self) -> str:
        lines = []
        contrasts = self.config.contrast_array
        # Say which spacing, rather than assuming: --contrasts linear changes it,
        # and a linear-vs-log verdict is only trustworthy if you know the grid.
        # Whichever axis has the more even gaps is the one it was sampled on.
        # Thresholding log-gap evenness alone misreads the default grid: its
        # bottom end is not evenly log-spaced (test_contrast_grid skips the
        # first three gaps for the same reason).
        def _cv(values: np.ndarray) -> float:
            gaps = np.diff(values)
            return float(gaps.std() / abs(gaps.mean())) if gaps.mean() else np.inf

        spacing = (
            "log-spaced"
            if _cv(np.log10(contrasts)) < _cv(contrasts)
            else "linearly spaced"
        )
        lines.append(
            f"contrasts: {len(contrasts)} {spacing}, "
            f"{contrasts.min():.4g} .. {contrasts.max():.4g}"
        )
        lines.append(
            f"frequencies (cyc/img): {list(self.config.frequencies_cpi)}"
        )
        lines.append(f"repetitions per cell (random orient/phase): {self.repetitions}")
        lines.append("")
        name_w = max(14, max(len(layer) for layer in self.layers) + 2)
        mod_results = self.mod_results
        header = (
            f"{'layer':<{name_w}}{'mean R^2':>10}{'pooled R^2':>12}{'spacing CV':>12}"
            f"{'lambda':>9}{'95% CI':>16}{'lam R^2':>9}{'c%lin':>8}{'c%log':>8}"
            + (f"{'chi2/dof':>12}" if self.noise else "")
            + (f"{'lam(mod)':>10}{'R^2(mod)':>10}" if mod_results else "")
            + (f"{'open%':>8}{'flip%':>8}" if self.gate_flips else "")
        )
        lines.append(header)
        lines.append("-" * len(header))
        log_c = np.log10(contrasts)
        for layer in self.layers:
            res = self.results[layer]
            cvs = [
                linear_spacing_uniformity(res.response[fi], log_c)
                for fi in range(res.response.shape[0])
            ]
            cv = float(np.nanmean(cvs))
            cerr = res.contrast_error
            lo, hi = res.lam_ci
            lines.append(
                f"{layer:<{name_w}}{res.mean_r2:>10.3f}{res.pooled.r2:>12.3f}{cv:>12.3f}"
                f"{res.lam:>+9.2f}{f'[{lo:+.2f}, {hi:+.2f}]':>16}{res.lam_r2:>9.3f}"
                f"{cerr[0]:>8.2f}{cerr[1]:>8.2f}"
                + (f"{self._chi2_per_dof(layer):>12.3g}" if self.noise else "")
                + (
                    f"{mod_results[layer].lam:>+10.2f}"
                    f"{mod_results[layer].lam_r2:>10.3f}"
                    if mod_results else ""
                )
                + (
                    f"{100 * self.gate_open[layer]:>8.1f}"
                    f"{100 * self.flip_fraction(layer):>8.2f}"
                    if self.gate_flips else ""
                )
            )
        lines.append("")
        lines.append(
            "lambda: exponent of D = a + b*(c^lam - 1)/lam -- 0 is the log law, "
            "1 linear in contrast, 0.5 square root, <0 saturating. Measured, not "
            "a choice between two laws; calibrated in test_lambda_calibration."
        )
        lines.append(
            "95% CI: profile-F interval on lambda. Spanning the search range "
            "means the data pin nothing down -- that layer has no exponent."
        )
        lines.append(
            "c%lin / c%log: median relative error when each law is inverted to "
            "predict contrast -- 0.16 = recovers it to within 16%."
        )
        if self.noise:
            lines.append(
                "chi2/dof: the better law, against the measured standard error of "
                "D. ~1 means it fits within noise; >>1 means neither law is right."
            )
        lines.append(
            "lam R^2: fit of the power family. Read lambda against it -- where "
            "this sags the exponent locates a response the family does not "
            "describe, and means correspondingly less."
        )
        if mod_results:
            lines.append(
                "lam(mod): the same exponent on the OTHER order of operations, "
                "mean_r mean_i |a_i(x_r) - gray_i|. That one has no "
                "zero-population floor, so where a layer's two lambdas disagree "
                "the primary metric is reporting its own sampling noise."
            )
        if self.gate_flips:
            lines.append(
                "open%: units positive at the gray operating point. flip%: units "
                "whose sign the grating changes, at the highest contrast, median "
                "over frequency. Piecewise linearity forces lambda = 1 while "
                "flip% is 0, so read the two together -- that is the test of the "
                "perturbation reading, not a second metric."
            )
        return "\n".join(lines)


def run_experiment(
    model: FeatureModel,
    config: GratingConfig | None = None,
    repetitions: int | None = None,
    seed: int = 0,
    verbose: bool = True,
    noise_blocks: int = 0,
) -> ExperimentResult:
    """Measure D(freq, contrast) per layer.

    ``noise_blocks`` > 1 additionally splits the repetitions round-robin into
    that many blocks and reports the standard error of D from their spread. It
    costs no extra forward passes -- only ``noise_blocks`` extra accumulators --
    and it is what turns "which law fits better" into "does either law fit
    within measurement error". Off by default: it multiplies accumulator memory,
    which matters at ``--layers all``.

    The full-precision accumulator is kept separate and untouched, so D is
    bit-identical whether or not blocks are requested and stays comparable with
    every run already committed.

    Caveat: D is a mean of absolute values, a biased nonlinear functional, so a
    block of reps/K draws is more biased than the full estimate. The spread is a
    usable variance estimate; the block means are not unbiased replicates.
    """
    cfg = config or GratingConfig()
    reps = cfg.repetitions if repetitions is None else repetitions
    rng = np.random.default_rng(seed)

    # Reference representation (single gray image). Its keys ARE the layer set:
    # a tap exists if and only if its module actually ran.
    ref_rep = model.represent(reference_rgb(cfg))
    layers = list(ref_rep.keys())

    # A hooked module that never runs contributes no tap, and used to do so in
    # silence -- you asked for N layers and got fewer, with nothing said. It is
    # not hypothetical: torchvision's ViT builds nn.MultiheadAttention, whose
    # forward hands out_proj.weight/bias to F.multi_head_attention_forward
    # rather than calling the module, so `--layers all` on vit_b_16 registers 75
    # modules and 12 of them never fire. The measurement is still correct -- the
    # taps simply do not exist -- but a depth profile silently missing every
    # attention output projection is not something to discover afterwards.
    unfired = [name for name in getattr(model, "layers", []) if name not in ref_rep]
    if unfired:
        warnings.warn(
            f"{len(unfired)} of {len(getattr(model, 'layers', []))} requested "
            f"taps never fired and are absent from this run: "
            f"{', '.join(unfired[:3])}"
            + (f", ... (+{len(unfired) - 3} more)" if len(unfired) > 3 else "")
            + ". Their modules are hooked but never called -- weights used "
            "functionally, or a branch that does not run on this input.",
            RuntimeWarning,
            stacklevel=2,
        )

    ref_rep = {k: v.astype(np.float64) for k, v in ref_rep.items()}
    # The gate state at the operating point. Computed once: it is the reference
    # every grating's gate state is compared against, and the fraction of it
    # that is open is the operating point itself.
    ref_gate = {layer: ref_rep[layer] > 0 for layer in layers}
    gate_open = {layer: float(np.mean(ref_gate[layer])) for layer in layers}

    freqs = cfg.frequency_array
    contrasts = cfg.contrast_array
    surfaces = {layer: np.zeros((len(freqs), len(contrasts))) for layer in layers}
    mod = {layer: np.zeros((len(freqs), len(contrasts))) for layer in layers}
    gate = {layer: np.zeros((len(freqs), len(contrasts))) for layer in layers}
    nblocks = int(noise_blocks) if noise_blocks and noise_blocks > 1 else 0
    noise = (
        {layer: np.zeros((len(freqs), len(contrasts))) for layer in layers}
        if nblocks else None
    )

    total = len(freqs) * len(contrasts)
    done = 0
    for fi, f in enumerate(freqs):
        for ci, c in enumerate(contrasts):
            # Accumulate activations across the random images (mean first).
            sums = {layer: np.zeros_like(ref_rep[layer]) for layer in layers}
            # The other order of operations. Each image's distance collapses to
            # a scalar right away, so this is an accumulator number per layer,
            # not an accumulator array -- it costs no memory worth counting.
            mod_sums = {layer: 0.0 for layer in layers}
            # Same shape of cost as mod_sums: each image's gate-flip count
            # collapses to a scalar immediately, so this is an accumulator
            # number per layer and not an accumulator array.
            gate_sums = {layer: 0.0 for layer in layers}
            blocks = (
                [{layer: np.zeros_like(ref_rep[layer]) for layer in layers}
                 for _ in range(nblocks)]
                if nblocks else None
            )
            counts = [0] * nblocks
            for idx, img in enumerate(
                sample_gratings(c, f, reps, rng, size=cfg.size, mean=cfg.mean)
            ):
                rep = model.represent(img)
                for layer in layers:
                    value = rep[layer].astype(np.float64)
                    sums[layer] += value
                    mod_sums[layer] += l1_distance(value, ref_rep[layer])
                    gate_sums[layer] += float(
                        np.mean((value > 0) != ref_gate[layer])
                    )
                    if blocks is not None:
                        blocks[idx % nblocks][layer] += value
                if blocks is not None:
                    counts[idx % nblocks] += 1
            for layer in layers:
                mu = sums[layer] / reps
                # D = mean_i | mu_i - gray_i |  (distance of the class-mean rep)
                surfaces[layer][fi, ci] = l1_distance(mu, ref_rep[layer])
                # D_mod = mean_r mean_i | a_i(x_r) - gray_i |  (mean of distances)
                mod[layer][fi, ci] = mod_sums[layer] / reps
                # G = mean_r fraction_i [ sign a_i(x_r) != sign gray_i ]
                gate[layer][fi, ci] = gate_sums[layer] / reps
                if blocks is not None:
                    per_block = [
                        l1_distance(blocks[b][layer] / counts[b], ref_rep[layer])
                        for b in range(nblocks)
                        if counts[b] > 0
                    ]
                    # sd of the block estimates / sqrt(K): the standard error of
                    # the full-sample D, which is what the fits should be judged
                    # against.
                    noise[layer][fi, ci] = (
                        float(np.std(per_block, ddof=1) / np.sqrt(len(per_block)))
                        if len(per_block) > 1 else np.nan
                    )
            done += 1
            if verbose and done % max(1, total // 10) == 0:
                print(f"  ... {done}/{total} cells", flush=True)

    results = {
        layer: summarise_layer(
            layer,
            contrasts=contrasts,
            frequencies=freqs,
            response=surfaces[layer],
        )
        for layer in layers
    }
    return ExperimentResult(
        config=cfg, repetitions=reps, layers=layers, surfaces=surfaces,
        results=results, noise=noise, mean_of_distances=mod,
        gate_flips=gate, gate_open=gate_open,
    )


# --------------------------------------------------------------------------- #
# Persistence: save/load the D(freq, contrast) surfaces (the expensive part).
# The fits are re-derived on load (summarise_layer is deterministic), so a
# saved run can be re-fit and re-plotted without touching the model again.
# --------------------------------------------------------------------------- #
def _finite(x) -> float | None:
    """JSON-safe float: NaN/inf (degenerate layers) become null."""
    x = float(x)
    return x if math.isfinite(x) else None


def result_summary(result: ExperimentResult, metadata: dict | None = None) -> dict:
    """A JSON-friendly summary: grids, per-layer fits (slopes, R^2), spacing CV."""
    contrasts = result.config.contrast_array
    log_c = np.log10(contrasts)
    mod_results = result.mod_results
    layers = []
    for name in result.layers:
        res = result.results[name]
        cvs = [
            linear_spacing_uniformity(res.response[fi], log_c)
            for fi in range(res.response.shape[0])
        ]
        mod = mod_results[name] if mod_results else None
        layers.append(
            {
                "layer": name,
                # The other order of operations, recorded alongside rather than
                # instead of: it has no zero-population floor, so where the two
                # disagree the primary metric is measuring its own noise.
                **({} if mod is None else {"mean_of_distances": {
                    "lambda": _finite(mod.lam),
                    "lambda_ci": [_finite(v) for v in mod.lam_ci],
                    "lambda_r2": _finite(mod.lam_r2),
                    "mean_r2": _finite(mod.mean_r2),
                }}),
                # The gate-flip instrument. Recorded per layer so the test of
                # the perturbation reading -- does lambda leave 1 only where
                # gates start switching? -- is a read of one committed file
                # rather than a re-run. The surface itself is in the npz.
                **({} if result.gate_flips is None else {"gates": {
                    "open_fraction": _finite(result.gate_open[name]),
                    "flip_at_min_contrast": _finite(result.flip_fraction(name, 0)),
                    "flip_at_max_contrast": _finite(result.flip_fraction(name, -1)),
                }}),
                "mean_r2": _finite(res.mean_r2),
                # The headline statistic. Recorded here as well as recomputed on
                # load, so a committed directory states its own result rather
                # than depending on whatever fit.py does at read time.
                "lambda": _finite(res.lam),
                "lambda_ci": [_finite(v) for v in res.lam_ci],
                "lambda_r2": _finite(res.lam_r2),
                "pooled_r2": _finite(res.pooled.r2),
                "pooled_slope": _finite(res.pooled.slope),
                "spacing_cv": _finite(np.nanmean(cvs)),
                # ``r2``/``slope``/``intercept`` are the log fit; ``lambda*`` is
                # the power fit, same split as ``mean_r2`` vs ``lambda_r2``
                # above. The eight lambdas are recorded rather than left to be
                # re-derived because the headline ``lambda`` is their median,
                # and it discards more variation than the differences it gets
                # compared on -- within one run lambda spans up to 1.75 across
                # frequency against 0.43 between architectures.
                "per_frequency": [
                    {
                        "frequency": float(f),
                        "r2": _finite(fit.r2),
                        "slope": _finite(fit.slope),
                        "intercept": _finite(fit.intercept),
                        "lambda": _finite(pf.lam),
                        "lambda_ci": [_finite(pf.lo), _finite(pf.hi)],
                        "lambda_r2": _finite(pf.r2),
                    }
                    for f, fit, pf in zip(
                        result.config.frequencies_cpi, res.per_frequency, res.power_fits
                    )
                ],
            }
        )
    return {
        "metadata": dict(metadata or {}),
        "repetitions": int(result.repetitions),
        "size": int(result.config.size),
        "mean": float(result.config.mean),
        "contrasts": [float(c) for c in contrasts],
        "frequencies": [float(f) for f in result.config.frequencies_cpi],
        "layers": layers,
    }


def save_result(
    result: ExperimentResult, path: str, metadata: dict | None = None
) -> dict[str, str]:
    """Persist a run: ``<base>.npz`` (surfaces + grids + metadata, the canonical
    artifact that can be re-fit) and ``<base>.json`` (the fit summary, for
    reading and cross-model aggregation). ``path`` may carry either suffix or
    none. Returns the two written paths.
    """
    base = path
    for suffix in (".npz", ".json"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
    directory = os.path.dirname(base)
    if directory:
        os.makedirs(directory, exist_ok=True)

    layers = list(result.layers)
    surfaces = np.stack(
        [np.asarray(result.surfaces[name], dtype=np.float64) for name in layers], axis=0
    )
    meta = dict(metadata or {})
    meta.setdefault("repetitions", int(result.repetitions))
    meta.setdefault("size", int(result.config.size))
    meta.setdefault("mean", float(result.config.mean))

    arrays = {
        "surfaces": surfaces,
        "layers": np.asarray(layers),
        "contrasts": result.config.contrast_array,
        "frequencies": result.config.frequency_array,
        "meta": np.asarray(json.dumps(meta)),
    }
    if result.mean_of_distances is not None:
        arrays["mean_of_distances"] = np.stack(
            [np.asarray(result.mean_of_distances[name], dtype=np.float64)
             for name in layers],
            axis=0,
        )
    if result.gate_flips is not None:
        arrays["gate_flips"] = np.stack(
            [np.asarray(result.gate_flips[name], dtype=np.float64)
             for name in layers],
            axis=0,
        )
        arrays["gate_open"] = np.asarray(
            [float(result.gate_open[name]) for name in layers], dtype=np.float64
        )
    npz_path = base + ".npz"
    np.savez_compressed(npz_path, **arrays)
    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result_summary(result, metadata), fh, indent=2)
    return {"npz": npz_path, "json": json_path}


NOTES_TEMPLATE = """# {slug}

{summary}

## What this run was for

{notes}

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
{command}
```

Code: `{commit}`{dirty}. Weights: {weights}.
"""


def save_run_dir(
    result: ExperimentResult,
    directory: str,
    metadata: dict,
    notes: str | None = None,
) -> dict[str, str]:
    """Persist a run as a committable directory.

    Layout: ``result.npz`` (canonical surfaces), ``result.json`` (fit summary,
    diffs readably in review), ``run.json`` (provenance -- commit, versions,
    weight identity) and ``notes.md`` (prose, seeded from a template).

    The surfaces are ``n_layers x n_freq x n_contrast`` floats -- a few KB
    regardless of ``--reps`` -- so every run is cheap to keep in git. Figures are
    deliberately not written here: they are ~100x larger than the data behind
    them and regenerate from the npz with ``--load``.
    """
    os.makedirs(directory, exist_ok=True)
    base = os.path.join(directory, "result")
    written = save_result(result, base, metadata=metadata)

    run_path = os.path.join(directory, "run.json")
    with open(run_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, sort_keys=True)

    notes_path = os.path.join(directory, "notes.md")
    if not os.path.exists(notes_path):  # never clobber written-up prose
        code = metadata.get("code", {})
        weights = metadata.get("weights", {})
        verified = weights.get("pretrained_verified")
        weights_desc = weights.get("source", "unknown")
        if verified is False:
            weights_desc += " -- NOT pretrained, numbers are a control only"
        # NaN mean_r2 (a degenerate layer) must not win the max: NaN comparisons
        # are all False, so pick with an explicit nan-safe key.
        best = max(
            ((result.results[name].mean_r2, name) for name in result.layers),
            key=lambda pair: (
                pair[0] if math.isfinite(pair[0]) else float("-inf")
            ),
            default=(float("nan"), "n/a"),
        )
        with open(notes_path, "w", encoding="utf-8") as fh:
            fh.write(
                NOTES_TEMPLATE.format(
                    slug=os.path.basename(os.path.normpath(directory)),
                    summary=(
                        f"`{metadata.get('model', '?')}`, "
                        f"{result.repetitions} reps/cell, "
                        f"best mean R² {best[0]:.3f} at `{best[1]}`."
                    ),
                    notes=notes or "_(fill in)_",
                    command=metadata.get("command", "?"),
                    commit=(code.get("commit") or "unknown")[:12],
                    dirty=" (dirty tree)" if code.get("dirty") else "",
                    weights=weights_desc,
                )
            )
    written["run"] = run_path
    written["notes"] = notes_path
    return written


def load_result(path: str) -> tuple[ExperimentResult, dict]:
    """Load a saved ``.npz`` and re-fit it into an ``ExperimentResult``.

    The fits are recomputed from the stored surfaces (no model needed), so
    ``report()`` and ``save_figures()`` work on the returned object exactly as
    on a fresh run. ``path`` may omit the ``.npz`` suffix. Returns
    ``(result, metadata)``.
    """
    npz_path = path
    if os.path.isdir(path):  # a save_run_dir() directory
        npz_path = os.path.join(path, "result.npz")
    elif not os.path.exists(npz_path) and os.path.exists(path + ".npz"):
        npz_path = path + ".npz"
    if not os.path.exists(npz_path):
        raise FileNotFoundError(npz_path)
    data = np.load(npz_path, allow_pickle=False)

    layers = [str(name) for name in data["layers"]]
    contrasts = np.asarray(data["contrasts"], dtype=np.float64)
    frequencies = np.asarray(data["frequencies"], dtype=np.float64)
    surfaces_arr = np.asarray(data["surfaces"], dtype=np.float64)
    meta = json.loads(data["meta"].item()) if "meta" in data.files else {}

    cfg = GratingConfig(
        size=int(meta.get("size", 224)),
        contrasts=tuple(float(c) for c in contrasts),
        frequencies_cpi=tuple(float(f) for f in frequencies),
        mean=float(meta.get("mean", 0.5)),
    )
    surfaces = {name: surfaces_arr[i] for i, name in enumerate(layers)}
    results = {
        name: summarise_layer(name, contrasts, frequencies, surfaces[name])
        for name in layers
    }
    # Absent from every run saved before 2026-07-27; those load with it None.
    mod = None
    if "mean_of_distances" in data.files:
        mod_arr = np.asarray(data["mean_of_distances"], dtype=np.float64)
        mod = {name: mod_arr[i] for i, name in enumerate(layers)}
    # Likewise absent from every run saved before 2026-08-02. Both arrays are
    # written together, so either both load or neither does.
    gate_flips = gate_open = None
    if "gate_flips" in data.files and "gate_open" in data.files:
        gate_arr = np.asarray(data["gate_flips"], dtype=np.float64)
        open_arr = np.asarray(data["gate_open"], dtype=np.float64)
        gate_flips = {name: gate_arr[i] for i, name in enumerate(layers)}
        gate_open = {name: float(open_arr[i]) for i, name in enumerate(layers)}
    reps = int(meta.get("repetitions", cfg.repetitions))
    result = ExperimentResult(
        config=cfg, repetitions=reps, layers=layers, surfaces=surfaces,
        results=results, mean_of_distances=mod,
        gate_flips=gate_flips, gate_open=gate_open,
    )
    return result, meta


def save_figures(
    result: ExperimentResult, out_dir: str, metadata: dict | None = None
) -> list[str]:
    """Write per-layer response figures and the lambda profile. Returns paths.

    ``metadata`` is forwarded to the profile so a figure that travels alone
    still names its run and its weight state.
    """
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    contrasts = result.config.contrast_array
    freqs = result.config.frequency_array

    for layer in result.layers:
        res = result.results[layer]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

        # (1) contrast-response family, one curve per spatial frequency
        for fi, f in enumerate(freqs):
            ax1.plot(contrasts, res.response[fi], marker="o", ms=3, label=f"{f:g}")
        ax1.set_xscale("log")
        ax1.set_xlabel("Michelson contrast (log axis)")
        ax1.set_ylabel("D = mean|meanrep - gray|")
        ax1.set_title(f"{layer}: contrast response")
        ax1.legend(fontsize=6, title="cyc/img", ncol=2)

        # (2) where the response sits between the two laws. Each frequency is
        # scaled to its own range so eight gains do not hide eight shapes, and
        # the two grey references are what lambda is measured against: on this
        # log axis the LOG law is the straight one and linear-in-contrast bends
        # up, which is the opposite of what the eye expects.
        fine = np.geomspace(contrasts.min(), contrasts.max(), 200)
        for lam_ref, style in ((0.0, "--"), (1.0, ":")):
            b = power_basis(fine, lam_ref)
            ax2.plot(fine, (b - b.min()) / (b.max() - b.min()), style,
                     color="#9a9a9a", lw=1.3, zorder=1)
        for fi, f in enumerate(freqs):
            y = res.response[fi]
            lo, hi = float(y.min()), float(y.max())
            if hi <= lo:
                continue
            unit = lambda v: (v - lo) / (hi - lo)
            ax2.plot(contrasts, unit(y), "o", ms=3, zorder=3)
            ax2.plot(fine, unit(res.power_fits[fi].predict(fine)), "-", lw=1,
                     alpha=0.8, zorder=2)
        ax2.set_xscale("log")
        ax2.set_xlabel("Michelson contrast (log axis; dashed = log law, dotted = linear)")
        ax2.set_ylabel("D, scaled to its own range")
        ci_lo, ci_hi = res.lam_ci
        ax2.set_title(
            f"{layer}: lambda = {res.lam:+.3f} [{ci_lo:+.2f}, {ci_hi:+.2f}]"
            f"   R^2 = {res.lam_r2:.3f}"
        )

        fig.tight_layout()
        path = os.path.join(out_dir, f"logresponse_{_safe(layer)}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)

    # The depth profile is the figure the result is actually read off, so it
    # comes out of --figures rather than having to be rebuilt by hand.
    from .panels import save_lambda_profile

    paths.append(
        save_lambda_profile(
            result, os.path.join(out_dir, "lambda_profile.png"), metadata
        )
    )
    return paths


def _safe(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")
