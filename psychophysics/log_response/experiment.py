"""Run the log-contrast-response experiment end to end.

Produces, for each tapped layer:

* the L1 response surface  L1(frequency, contrast)  (mean absolute change in the
  layer's representation between the gray reference and each grating, averaged
  over stimulus phases),
* per-frequency and pooled linear fits of  L1 vs log10(contrast)  with R^2,
* optional figures: the response surface and the log-linearity plot.

The core 2017 claims this is set up to test:
  (1) end-computation layers show a band-pass response at low contrast that
      flattens toward contrast constancy at high contrast;
  (2) L1 vs log(contrast) is close to linear at the output layer (R^2 ~= 0.98
      averaged across spatial frequency), i.e. log-spaced contrasts become
      evenly (linearly) spaced in representation space.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .gratings import GratingConfig, build_grid
from .features import FeatureModel, l1_distance
from .fit import LayerLogResult, summarise_layer, linear_spacing_uniformity


@dataclass
class ExperimentResult:
    config: GratingConfig
    layers: list[str]
    # layer -> (n_freq, n_contrast) L1 surface
    surfaces: dict[str, np.ndarray]
    results: dict[str, LayerLogResult]

    def report(self) -> str:
        lines = []
        contrasts = np.asarray(self.config.contrasts)
        lines.append(
            f"contrasts (log-spaced): {contrasts.min():.3g} .. {contrasts.max():.3g} "
            f"({len(contrasts)} levels)"
        )
        lines.append(
            f"frequencies (cyc/img): {list(self.config.frequencies_cpi)}"
        )
        lines.append("")
        header = f"{'layer':<16}{'mean R^2':>10}{'pooled R^2':>12}{'spacing CV':>12}"
        lines.append(header)
        lines.append("-" * len(header))
        for layer in self.layers:
            res = self.results[layer]
            # spacing CV: uniformity of consecutive gaps, averaged over frequency
            cvs = [linear_spacing_uniformity(res.response[fi]) for fi in range(res.response.shape[0])]
            cv = float(np.nanmean(cvs))
            lines.append(
                f"{layer:<16}{res.mean_r2:>10.3f}{res.pooled.r2:>12.3f}{cv:>12.3f}"
            )
        return "\n".join(lines)


def run_experiment(
    model: FeatureModel,
    config: GratingConfig | None = None,
    verbose: bool = True,
) -> ExperimentResult:
    cfg = config or GratingConfig()
    grid = build_grid(cfg)
    n_freq, n_contrast, n_phase = grid.images.shape[:3]

    # Reference representation (single gray image).
    ref_rep = model.represent(grid.reference)
    layers = list(ref_rep.keys())

    surfaces = {layer: np.zeros((n_freq, n_contrast)) for layer in layers}

    total = n_freq * n_contrast
    done = 0
    for fi in range(n_freq):
        for ci in range(n_contrast):
            # Average L1 over phases (per layer).
            per_layer_l1 = {layer: [] for layer in layers}
            for pi in range(n_phase):
                rep = model.represent(grid.images[fi, ci, pi])
                for layer in layers:
                    per_layer_l1[layer].append(l1_distance(rep[layer], ref_rep[layer]))
            for layer in layers:
                surfaces[layer][fi, ci] = float(np.mean(per_layer_l1[layer]))
            done += 1
            if verbose and done % max(1, total // 10) == 0:
                print(f"  ... {done}/{total} stimuli", flush=True)

    results = {
        layer: summarise_layer(
            layer,
            contrasts=grid.contrasts,
            frequencies=grid.frequencies,
            response=surfaces[layer],
        )
        for layer in layers
    }
    return ExperimentResult(config=cfg, layers=layers, surfaces=surfaces, results=results)


def save_figures(result: ExperimentResult, out_dir: str) -> list[str]:
    """Write per-layer response-surface and log-linearity figures. Returns paths."""
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    contrasts = np.asarray(result.config.contrasts)
    freqs = np.asarray(result.config.frequencies_cpi)

    for layer in result.layers:
        res = result.results[layer]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

        # (1) contrast-response family, one curve per spatial frequency
        for fi, f in enumerate(freqs):
            ax1.plot(contrasts, res.response[fi], marker="o", ms=3, label=f"{f:g} cpi")
        ax1.set_xscale("log")
        ax1.set_xlabel("Michelson contrast (log axis)")
        ax1.set_ylabel("L1 representation change")
        ax1.set_title(f"{layer}: contrast response")
        ax1.legend(fontsize=7, title="spatial freq")

        # (2) log-linearity: L1 vs log10(contrast), per-freq fit lines
        logc = np.log10(contrasts)
        for fi, f in enumerate(freqs):
            ax2.plot(logc, res.response[fi], "o", ms=3)
            fit = res.per_frequency[fi]
            ax2.plot(logc, fit.predict(contrasts), "-", lw=1, alpha=0.7)
        ax2.set_xlabel("log10(contrast)")
        ax2.set_ylabel("L1 representation change")
        ax2.set_title(f"{layer}: mean R^2 = {res.mean_r2:.3f}")

        fig.tight_layout()
        path = os.path.join(out_dir, f"logresponse_{_safe(layer)}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        paths.append(path)
    return paths


def _safe(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")
