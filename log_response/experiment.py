"""Run the log-contrast-response experiment end to end.

For each layer and each (contrast, frequency) cell the metric is a **distance of
means**, not a mean of distances:

    mu_i(c,f) = mean over the 250 random images of unit i's activation
    D(c,f)    = mean_i | mu_i(c,f) - a_i(gray) |

Averaging activations across the random orientation/phase draws happens BEFORE
the absolute value, so phase/orientation-specific activity cancels first (see
METHOD.md, Jensen's inequality note).

Produces, per layer: the L1 surface D(freq, contrast); per-frequency and pooled
linear fits of D vs log10(contrast) with R^2; and optional figures.
"""

from __future__ import annotations

from dataclasses import dataclass
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
        header = f"{'layer':<14}{'mean R^2':>10}{'pooled R^2':>12}{'spacing CV':>12}"
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
                f"{layer:<14}{res.mean_r2:>10.3f}{res.pooled.r2:>12.3f}{cv:>12.3f}"
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
