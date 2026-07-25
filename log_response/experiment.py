"""Run the log-contrast-response experiment end to end.

For each layer and each (contrast, frequency) cell the metric is a **distance of
means**, not a mean of distances:

    mu_i(c,f) = mean over the 250 random images of unit i's activation
    D(c,f)    = mean_i | mu_i(c,f) - a_i(gray) |

Averaging activations across the random orientation/phase draws happens BEFORE
the absolute value, so phase/orientation-specific activity cancels first (see
wiki/Method.md, Jensen's inequality note).

Produces, per layer: the L1 surface D(freq, contrast); per-frequency and pooled
linear fits of D vs log10(contrast) with R^2; and optional figures.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
import numpy as np

from .gratings import GratingConfig, sample_gratings, reference_rgb
from .features import FeatureModel, l1_distance
from .fit import LayerLogResult, summarise_layer, linear_spacing_uniformity


@dataclass
class ExperimentResult:
    config: GratingConfig
    repetitions: int
    layers: list[str]
    # layer -> (n_freq, n_contrast) L1 surface
    surfaces: dict[str, np.ndarray]
    results: dict[str, LayerLogResult]

    def report(self) -> str:
        lines = []
        contrasts = self.config.contrast_array
        lines.append(
            f"contrasts: {len(contrasts)} log-spaced, "
            f"{contrasts.min():.4g} .. {contrasts.max():.4g}"
        )
        lines.append(
            f"frequencies (cyc/img): {list(self.config.frequencies_cpi)}"
        )
        lines.append(f"repetitions per cell (random orient/phase): {self.repetitions}")
        lines.append("")
        name_w = max(14, max(len(layer) for layer in self.layers) + 2)
        header = f"{'layer':<{name_w}}{'mean R^2':>10}{'pooled R^2':>12}{'spacing CV':>12}"
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
            lines.append(
                f"{layer:<{name_w}}{res.mean_r2:>10.3f}{res.pooled.r2:>12.3f}{cv:>12.3f}"
            )
        return "\n".join(lines)


def run_experiment(
    model: FeatureModel,
    config: GratingConfig | None = None,
    repetitions: int | None = None,
    seed: int = 0,
    verbose: bool = True,
) -> ExperimentResult:
    cfg = config or GratingConfig()
    reps = cfg.repetitions if repetitions is None else repetitions
    rng = np.random.default_rng(seed)

    # Reference representation (single gray image).
    ref_rep = model.represent(reference_rgb(cfg))
    layers = list(ref_rep.keys())
    ref_rep = {k: v.astype(np.float64) for k, v in ref_rep.items()}

    freqs = cfg.frequency_array
    contrasts = cfg.contrast_array
    surfaces = {layer: np.zeros((len(freqs), len(contrasts))) for layer in layers}

    total = len(freqs) * len(contrasts)
    done = 0
    for fi, f in enumerate(freqs):
        for ci, c in enumerate(contrasts):
            # Accumulate activations across the random images (mean first).
            sums = {layer: np.zeros_like(ref_rep[layer]) for layer in layers}
            for img in sample_gratings(c, f, reps, rng, size=cfg.size, mean=cfg.mean):
                rep = model.represent(img)
                for layer in layers:
                    sums[layer] += rep[layer].astype(np.float64)
            for layer in layers:
                mu = sums[layer] / reps
                # D = mean_i | mu_i - gray_i |  (distance of the class-mean rep)
                surfaces[layer][fi, ci] = l1_distance(mu, ref_rep[layer])
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
        config=cfg, repetitions=reps, layers=layers, surfaces=surfaces, results=results
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
    layers = []
    for name in result.layers:
        res = result.results[name]
        cvs = [
            linear_spacing_uniformity(res.response[fi], log_c)
            for fi in range(res.response.shape[0])
        ]
        layers.append(
            {
                "layer": name,
                "mean_r2": _finite(res.mean_r2),
                "pooled_r2": _finite(res.pooled.r2),
                "pooled_slope": _finite(res.pooled.slope),
                "spacing_cv": _finite(np.nanmean(cvs)),
                "per_frequency": [
                    {
                        "frequency": float(f),
                        "r2": _finite(fit.r2),
                        "slope": _finite(fit.slope),
                        "intercept": _finite(fit.intercept),
                    }
                    for f, fit in zip(result.config.frequencies_cpi, res.per_frequency)
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

    npz_path = base + ".npz"
    np.savez_compressed(
        npz_path,
        surfaces=surfaces,
        layers=np.asarray(layers),
        contrasts=result.config.contrast_array,
        frequencies=result.config.frequency_array,
        meta=np.asarray(json.dumps(meta)),
    )
    json_path = base + ".json"
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(result_summary(result, metadata), fh, indent=2)
    return {"npz": npz_path, "json": json_path}


def load_result(path: str) -> tuple[ExperimentResult, dict]:
    """Load a saved ``.npz`` and re-fit it into an ``ExperimentResult``.

    The fits are recomputed from the stored surfaces (no model needed), so
    ``report()`` and ``save_figures()`` work on the returned object exactly as
    on a fresh run. ``path`` may omit the ``.npz`` suffix. Returns
    ``(result, metadata)``.
    """
    npz_path = path
    if not os.path.exists(npz_path) and os.path.exists(path + ".npz"):
        npz_path = path + ".npz"
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
    reps = int(meta.get("repetitions", cfg.repetitions))
    result = ExperimentResult(
        config=cfg, repetitions=reps, layers=layers, surfaces=surfaces, results=results
    )
    return result, meta


def save_figures(result: ExperimentResult, out_dir: str) -> list[str]:
    """Write per-layer response-surface and log-linearity figures. Returns paths."""
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

        # (2) log-linearity: D vs log10(contrast), per-freq fit lines
        logc = np.log10(contrasts)
        for fi, f in enumerate(freqs):
            ax2.plot(logc, res.response[fi], "o", ms=3)
            fit = res.per_frequency[fi]
            ax2.plot(logc, fit.predict(contrasts), "-", lw=1, alpha=0.7)
        ax2.set_xlabel("log10(contrast)")
        ax2.set_ylabel("D")
        ax2.set_title(f"{layer}: mean R^2 = {res.mean_r2:.3f}")

        fig.tight_layout()
        path = os.path.join(out_dir, f"logresponse_{_safe(layer)}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def _safe(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")
