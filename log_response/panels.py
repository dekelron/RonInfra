"""Per-layer scatter panels: contrast linear (top row) vs log (bottom row).

The pair of rows is the point. On a linear contrast axis every layer looks like
the same saturating curve; on a log axis the late layers straighten into lines
and the early ones visibly do not. That contrast *is* the result, so the two
scales belong in one figure rather than in two.

Spatial frequency is an **ordered** quantity (cycles/image), so it is encoded as
a single-hue light-to-dark ramp with a colourbar -- not as cycled categorical
hues, which would present an ordered variable as unordered identities.
"""

from __future__ import annotations

import numpy as np

# Validated blue ramp. Up to five frequencies get the discrete ordinal steps
# (each adjacent pair separated enough to read individually on a light surface);
# past five no discrete ramp clears that bar, so the band is interpolated and
# the colourbar carries the mapping -- the reader reads "darker = higher" as a
# gradient rather than telling eight swatches apart.
ORDINAL_STEPS = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95", "#0d366b"]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e6e5e1"


def frequency_colours(n: int) -> list[str]:
    """``n`` colours along the ramp, light (low frequency) to dark (high)."""
    from matplotlib.colors import LinearSegmentedColormap, to_hex

    if n <= 1:
        return [ORDINAL_STEPS[2]]
    if n <= len(ORDINAL_STEPS):  # validated discrete steps, maximally spread
        idx = np.linspace(0, len(ORDINAL_STEPS) - 1, n).round().astype(int)
        return [ORDINAL_STEPS[i] for i in idx]
    cmap = LinearSegmentedColormap.from_list("freq", ORDINAL_STEPS)
    return [to_hex(cmap(v)) for v in np.linspace(0.0, 1.0, n)]


def _stamp(metadata: dict | None) -> str:
    """Weight state, so a figure that travels alone still says what it is."""
    meta = metadata or {}
    verified = (meta.get("weights") or {}).get("pretrained_verified", "unknown")
    text = {
        True: "pretrained verified",
        False: "UNTRAINED control",
        None: "weight-free",
    }.get(verified, "weight state unrecorded")
    if meta.get("scramble"):
        text += ", then scrambled within layers (control)"
    return text


def save_panels(
    result,
    path: str,
    metadata: dict | None = None,
    title: str | None = None,
) -> str:
    """Write the two-row panel figure for ``result``. Returns the path written."""
    import os
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    freqs = result.config.frequency_array
    contrasts = result.config.contrast_array
    layers = list(result.layers)
    colours = frequency_colours(len(freqs))
    n = len(layers)

    fig, axes = plt.subplots(
        2, n, figsize=(3.05 * n, 6.4), sharex="row", squeeze=False,
        gridspec_kw={"hspace": 0.42, "wspace": 0.34},
    )
    fig.patch.set_facecolor(SURFACE)

    for col, layer in enumerate(layers):
        res = result.results[layer]
        lo = min(float(res.response.min()), 0.0)
        hi = float(res.response.max())
        pad = 0.08 * (hi - lo) if hi > lo else 1.0

        for row in (0, 1):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(GRID)
            ax.tick_params(colors=INK_MUTED, labelsize=7.5, length=3, width=0.6)
            ax.set_ylim(lo - pad, hi + pad)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
            ax.yaxis.get_offset_text().set(color=INK_MUTED, size=7)

            for fi in range(len(freqs)):
                if row == 1:  # the fit lines carry the log-linearity claim
                    ax.plot(
                        contrasts, res.per_frequency[fi].predict(contrasts),
                        color=colours[fi], lw=1.1, alpha=0.55, zorder=2,
                    )
                # A surface-coloured ring keeps overlapping points separable.
                ax.scatter(
                    contrasts, res.response[fi], s=26, color=colours[fi],
                    edgecolors=SURFACE, linewidths=0.7, zorder=3,
                )
            ax.set_xlabel(
                f"Michelson contrast ({'log' if row else 'linear'})",
                color=INK_MUTED, fontsize=8,
            )

        axes[1][col].set_xscale("log")
        axes[0][col].set_title(layer, color=INK, fontsize=10, pad=8)
        axes[1][col].text(
            0.04, 0.94, f"mean R² {res.mean_r2:.3f}",
            transform=axes[1][col].transAxes, color=INK, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.28", fc=SURFACE, ec=GRID, lw=0.6),
        )

    for row in (0, 1):
        axes[row][0].set_ylabel(
            "D = mean |mean rep − gray|", color=INK_MUTED, fontsize=8.5
        )

    if len(freqs) > 1:
        cmap = LinearSegmentedColormap.from_list("freq", colours, N=len(freqs))
        norm = BoundaryNorm(np.arange(len(freqs) + 1) - 0.5, len(freqs))
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
            ticks=np.arange(len(freqs)), fraction=0.016, pad=0.015,
        )
        cbar.ax.set_yticklabels(
            [f"{f:g}" for f in freqs], color=INK_MUTED, fontsize=7.5
        )
        cbar.set_label(
            "spatial frequency (cycles/image), light → dark = low → high",
            color=INK_MUTED, fontsize=8,
        )
        cbar.outline.set_edgecolor(GRID)
        cbar.ax.tick_params(length=0)

    model = (metadata or {}).get("model", "model")
    fig.suptitle(
        f"{title or model} — log-contrast response by layer",
        color=INK, fontsize=13, x=0.045, ha="left", y=0.985,
    )
    fig.text(
        0.045, 0.935,
        "Top: contrast linear. Bottom: contrast log, with per-frequency fits.  "
        f"{result.repetitions} reps/cell · {_stamp(metadata)}",
        color=INK_MUTED, fontsize=8.5, ha="left",
    )

    fig.savefig(path, dpi=170, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path
