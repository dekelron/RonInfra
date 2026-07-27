"""Digitise Figure 3b of the source paper, and compare it to a committed run.

Why this exists: the paper's §5 claims are summary numbers (`prob` R² = 98%,
"much lower … up to fc7"), and matching a summary number is weak evidence. The
figure holds the whole measurement -- four representations × 14 contrasts × 8
spatial frequencies -- so comparing against *it* turns four numbers into 448.

The axis labels were converted to outlines when the MATLAB figure was embedded,
so there is no text to read. The plotted curves survive as vector polylines,
and everything needed is recoverable from their geometry:

* `Figure3.pdf` is form XObject **470**; its four sub-panels (`data`, `conv1_1`,
  `fc8`, `prob`) each hold **15 polylines of 8 vertices**.
* All curves share one set of 8 x-positions -- the **frequency grid**, whose
  spacing is six equal steps with a shorter one at each end.
* The 15 stroke colours are the jet colormap in order, identically in all four
  panels, so colour gives the **contrast index** more robustly than curve
  height does (heights cross).
* The lowest curve is flat at the *same* height in all four panels: it is
  `c = 0`. That fixes the y-origin, and -- because a log axis could not place
  zero at a finite height -- it also proves **the y-axis is linear**. So each
  panel needs exactly one free scale factor and nothing else.

Run ``python -m log_response.figure3`` for the grids, ``--compare`` for the
panel-by-panel comparison against the committed runs.
"""

from __future__ import annotations

import os
import re
import zlib

import numpy as np

PDF = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "wiki", "1701.04674-adaptation-as-readout.pdf")
FIGURE_XOBJECT = 470
PANELS = ("data", "conv1_1", "fc8", "prob")

# The jet colormap as sampled by the figure, darkest blue (c=0) to dark red.
JET: tuple[tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.749), (0.0, 0.0, 1.0), (0.0, 0.251, 1.0), (0.0, 0.502, 1.0),
    (0.0, 0.749, 1.0), (0.0, 1.0, 1.0), (0.251, 1.0, 0.749), (0.502, 1.0, 0.502),
    (0.749, 1.0, 0.251), (1.0, 1.0, 0.0), (1.0, 0.749, 0.0), (1.0, 0.502, 0.0),
    (1.0, 0.251, 0.0), (1.0, 0.0, 0.0), (0.749, 0.0, 0.0),
)


def _object_stream(raw: bytes, number: int) -> bytes:
    for m in re.finditer(rb"(\d+)\s+0\s+obj\b", raw):
        if int(m.group(1)) != number:
            continue
        body = raw[m.end():raw.find(b"endobj", m.end())]
        start = body.find(b"\n", body.find(b"stream")) + 1
        return zlib.decompress(body[start:body.find(b"endstream", start)])
    raise KeyError(f"object {number} not found in PDF")


def _polylines(content: bytes) -> list[tuple[list[tuple[float, float]], tuple]]:
    """Every stroked/filled subpath in a content stream, with its stroke colour.

    A minimal PDF path interpreter: enough of the operator set (q/Q/cm for the
    transform, m/l/c for path construction, RG/rg for colour) to recover plotted
    data. Bezier control points are dropped and only the endpoint kept, which is
    exact here because the curves are drawn as straight segments.
    """
    tokens = re.findall(rb"[-\d.]+|[A-Za-z'\"*]+", content)
    stack: list[np.ndarray] = []
    ctm = np.eye(3)
    current: list[tuple[float, float]] = []
    out: list[tuple[list[tuple[float, float]], tuple]] = []
    nums: list[float] = []
    colour: tuple = (0.0, 0.0, 0.0)

    def to_device(pt):
        v = np.array([pt[0], pt[1], 1.0]) @ ctm
        return (float(v[0]), float(v[1]))

    for token in tokens:
        try:
            nums.append(float(token))
            continue
        except ValueError:
            pass
        op = token.decode("latin-1")
        if op == "q":
            stack.append(ctm.copy())
        elif op == "Q" and stack:
            ctm = stack.pop()
        elif op == "cm" and len(nums) >= 6:
            a, b, c, d, e, f = nums[-6:]
            ctm = np.array([[a, b, 0.0], [c, d, 0.0], [e, f, 1.0]]) @ ctm
        elif op in ("RG", "rg") and len(nums) >= 3:
            colour = tuple(round(v, 3) for v in nums[-3:])
        elif op == "m" and len(nums) >= 2:
            if len(current) > 1:
                out.append((current, colour))
            current = [to_device(nums[-2:])]
        elif op in ("l", "c") and len(nums) >= 2:
            current.append(to_device(nums[-2:]))
        elif op in ("S", "s", "f", "F", "f*", "B", "b", "n"):
            if len(current) > 1:
                out.append((current, colour))
            current = []
        nums = []
    if len(current) > 1:
        out.append((current, colour))
    return out


def extract(pdf_path: str = PDF) -> dict:
    """Digitise Figure 3b.

    Returns ``{'frequencies': (8,), 'contrasts': (14,), 'panels': {name: (14, 8)}}``
    -- panel values are baseline-subtracted (so `c = 0` is exactly 0) and in
    arbitrary units, one unknown positive scale factor per panel.
    """
    with open(pdf_path, "rb") as fh:
        raw = fh.read()
    curves = [(sp, col) for sp, col in _polylines(_object_stream(raw, FIGURE_XOBJECT))
              if len(sp) == 8]

    grouped: dict[float, dict[tuple, list[float]]] = {}
    for sp, col in curves:
        grouped.setdefault(round(sp[0][0], 1), {})[col] = [p[1] for p in sp]
    keys = sorted(grouped)
    if len(keys) != 4:
        raise ValueError(f"expected 4 sub-panels, found {len(keys)}")

    xs = np.array([p[0] for p in curves[0][0]], dtype=np.float64)
    steps = np.diff(xs)
    unit = float(np.median(steps[1:-1]))          # the doubling step
    frequencies = np.concatenate([[1.0], np.cumprod(2.0 ** (steps / unit))])

    panels = {}
    for key, name in zip(keys, PANELS):
        curve = grouped[key]
        missing = [c for c in JET if c not in curve]
        if missing:
            raise ValueError(f"panel {name}: {len(missing)} colours missing")
        baseline = float(np.mean(curve[JET[0]]))  # the c = 0 line
        panels[name] = np.array([[y - baseline for y in curve[c]] for c in JET[1:]])

    # Contrast reads off the panels that are linear in it: 'data' is the image
    # itself and 'conv1_1' a convolution, so curve height is proportional to c.
    # Average the two, and normalise by the top curve (c = 1 by construction).
    heights = np.mean([panels["data"].mean(axis=1), panels["conv1_1"].mean(axis=1)], axis=0)
    contrasts = heights / heights[-1]
    return {"frequencies": frequencies, "contrasts": contrasts, "panels": panels}


def compare(measured: dict[str, np.ndarray], pdf_path: str = PDF) -> dict:
    """Compare digitised panels against measured ``{panel: (n_freq, n_contrast)}``.

    Reports both the raw agreement and the agreement with the shared contrast
    trend divided out. The second is the one that means something: D rises with
    contrast in every panel and in every run, so a raw correlation is flattered
    by structure neither side had to get right.
    """
    figure = extract(pdf_path)
    report = {}
    for name, paper in figure["panels"].items():
        if name not in measured:
            continue
        mine = np.asarray(measured[name], dtype=np.float64).T   # -> (contrast, freq)
        scale = float((paper * mine).sum() / (mine * mine).sum())
        per_row = lambda a: a / a.mean(axis=1, keepdims=True)
        pn, mn = per_row(paper), per_row(mine)
        report[name] = {
            "scale": scale,
            "r": float(np.corrcoef(paper.ravel(), mine.ravel())[0, 1]),
            "residual": float(np.median(np.abs(paper - scale * mine)
                                        / np.maximum(paper, 1e-12))),
            "r_frequency_only": float(np.corrcoef(pn.ravel(), mn.ravel())[0, 1]),
            "residual_frequency_only": float(np.median(np.abs(pn - mn) / mn)),
            "cells": int(paper.size),
        }
    return report


def main(argv=None) -> None:
    import argparse

    from .gratings import CONTRASTS, FREQUENCIES_CPI

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pdf", default=PDF)
    p.add_argument(
        "--compare", action="store_true",
        help="also compare the panels against results/vgg19-r250-s0-alllayers-"
             "fixed-caffe (and results/data-r250-s0 for the 'data' panel)",
    )
    args = p.parse_args(argv)

    fig = extract(args.pdf)
    print("Frequency grid recovered from the shared x-positions (cycles/image):")
    print("   figure:", np.round(fig["frequencies"], 3))
    print("   repo:  ", np.asarray(FREQUENCIES_CPI))
    print("\nContrast grid recovered from the two contrast-linear panels (x128):")
    print("   figure:", np.round(fig["contrasts"] * 128, 2))
    print("   repo:  ", np.round(np.asarray(CONTRASTS) * 128).astype(int))
    print(f"\n   {len(fig['contrasts'])} contrasts + a c=0 reference line, "
          f"{len(fig['frequencies'])} frequencies, 4 panels")

    if not args.compare:
        return

    from .experiment import load_result

    caffe, _ = load_result("results/vgg19-r250-s0-alllayers-fixed-caffe")
    pixels, _ = load_result("results/data-r250-s0")
    report = compare({
        "data": pixels.surfaces["data"],
        "conv1_1": caffe.surfaces["features.0"],
        "fc8": caffe.surfaces["logits"],
        "prob": caffe.surfaces["prob"],
    }, args.pdf)

    print("\nFigure 3b vs the committed runs, one free scale factor per panel:")
    print(f"   {'panel':<9}{'r':>9}{'resid':>8}   {'r (freq only)':>15}{'resid':>8}{'cells':>7}")
    for name, row in report.items():
        print(f"   {name:<9}{row['r']:>9.4f}{row['residual']:>7.1%}   "
              f"{row['r_frequency_only']:>15.3f}{row['residual_frequency_only']:>8.1%}"
              f"{row['cells']:>7}")
    print("\n'freq only' divides out the shared contrast trend. 'data' must "
          "collapse there\n(it is the noise floor, drawn independently); the "
          "other three must survive.")


if __name__ == "__main__":
    main()
