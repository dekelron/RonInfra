"""What a *training recipe* leaves behind in a checkpoint, measured gauge-freely.

`wiki/Results.md` records a difference between two trainings of one architecture
on one dataset: the converted Oxford/Caffe VGG-19 holds λ ≈ 1 across its whole
conv stack, while torchvision's `IMAGENET1K_V1` leaves the linear regime from
mid-stack. Same layers, same task, same stimulus — so whatever separates them is
in the recipe. This module measures the traces a recipe difference can leave.

Most of the work is done before any measurement, by an exact symmetry.

**The gauge argument.** ReLU is positively homogeneous, and max-pooling and
average-pooling commute with positive scaling, so a plain feed-forward ReLU net
has an exact rescaling symmetry. For weight layers ``0..n`` in execution order,
pick any ``α_i > 0`` with ``α_{-1} = α_n = 1`` and set::

    W_i <- W_i · α_i / α_{i-1}        b_i <- b_i · α_i

The layer-``i`` activations are multiplied by ``α_i``; **every ReLU gate in the
network is unchanged, and so are the logits**. The experiment's ``D(c,f)`` at
layer ``i`` is therefore multiplied by ``α_i``, and λ — an *exponent*, fitted as
``D = a + b·(c^λ−1)/λ`` — does not move at all, because rescaling ``D`` rescales
``a`` and ``b`` and leaves λ alone.

That is a theorem, not a hypothesis, and it prunes the candidate list hard: **no
recipe difference that acts only on the scale of the weights can be the cause.**
Learning-rate scale, the overall strength of a weight decay applied uniformly,
and — the one that looks most guilty and is most thoroughly innocent — the input
normalisation, all fall to it. Caffe subtracts a mean pixel and stops;
torchvision divides by a standard deviation as well. Dividing an input by a
constant is a gauge transformation at conv1, so the two preprocessings cannot by
themselves produce different λ. (`convert_weights.py` already folds one into the
other exactly, which is the same statement from the other end.)

What survives the pruning is anything that changes weights *relative to each
other*, and there are two families of it:

1. **Where units sit relative to their own ReLU** — the operating point. It is
   the gauge-invariant ratio of what the bias contributes to a pre-activation
   against what the image drives into it. A unit held far from threshold cannot
   have its gate flipped by a low-contrast grating, and while no gate flips the
   network is exactly linear in contrast: λ = 1.
2. **The shape of the weight distribution** — sparsity, kurtosis, how much of a
   kernel is zero-frequency, effective rank. These are ratios within a layer, so
   they are gauge-invariant too, and they are what distinguishes a strong from a
   weak weight decay and a scale-jittered from a crop-jittered augmentation.

Both families are measured here, per layer, and written as one small JSON that
can be committed next to a run.

Usage::

    python -m log_response.operating_point --model vgg19 --weights vgg19_caffe.pth \
        --out results/oppoint-vgg19-caffe/operating_point.json
    python -m log_response.operating_point --model vgg19 \
        --compare results/vgg19-r250-s0-alllayers-fixed-caffe

Cost is a few hundred forward passes — minutes, not the hours a ``D`` surface
takes — because nothing here needs the metric's repetition count. Flip fractions
are counted over every unit at every spatial position, so a handful of draws
already averages over millions of gates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
import argparse
import json
import os
import numpy as np

from .gratings import CONTRASTS, FREQUENCIES_CPI, make_grating, make_reference, to_rgb

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- #
# The gauge group
# --------------------------------------------------------------------------- #
def weight_layers(net) -> list[tuple[str, object]]:
    """``(name, module)`` for every Conv2d/Linear, in execution order.

    Registration order equals execution order for a net built out of ``Sequential``
    containers, which covers the plain stacks this module is about (VGG, AlexNet).
    Anything with branching is rejected by :func:`relu_taps` rather than guessed at.
    """
    import torch.nn as nn

    return [
        (name, module)
        for name, module in net.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear))
    ]


def relu_taps(net) -> list[str]:
    """Names of the weight layers whose output feeds straight into a ReLU.

    These are the pre-activations whose operating point is meaningful: the ReLU
    immediately downstream is the gate whose flipping the metric would see.
    """
    import torch.nn as nn

    ordered = [
        (name, module)
        for name, module in net.named_modules()
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.ReLU))
    ]
    taps = []
    for (name, module), (_, nxt) in zip(ordered, ordered[1:]):
        if isinstance(module, (nn.Conv2d, nn.Linear)) and isinstance(nxt, nn.ReLU):
            taps.append(name)
    return taps


def apply_gauge(net, alphas: dict[str, float]) -> None:
    """Rescale in place by the exact ReLU symmetry; the network function is unchanged.

    ``alphas`` maps a weight-layer name to the factor its *output* is multiplied
    by. The last weight layer is forced to 1.0 so the logits are untouched. Used
    to verify that every statistic this module reports is gauge-invariant — a
    statistic that moves under this transformation is measuring the coordinate
    system rather than the training.
    """
    import torch

    layers = weight_layers(net)
    with torch.no_grad():
        previous = 1.0
        for index, (name, module) in enumerate(layers):
            alpha = 1.0 if index == len(layers) - 1 else float(alphas.get(name, 1.0))
            module.weight.mul_(alpha / previous)
            if module.bias is not None:
                module.bias.mul_(alpha)
            previous = alpha


def scale_biases(net, factor: float, taps: list[str] | None = None) -> None:
    """Multiply biases by ``factor``. **Not** a gauge transformation.

    Moving biases without moving weights slides every unit along its own ReLU,
    which is the one intervention that changes the operating point while leaving
    the filters — what the layer is tuned for — exactly as trained.
    """
    import torch

    with torch.no_grad():
        for name, module in weight_layers(net):
            if module.bias is None or (taps is not None and name not in taps):
                continue
            module.bias.mul_(factor)


# --------------------------------------------------------------------------- #
# Gauge-invariant weight-shape statistics
# --------------------------------------------------------------------------- #
def _effective_rank(matrix) -> float | None:
    """Entropy effective rank of the singular spectrum, as a fraction of full rank.

    Returns ``None`` where the SVD would dominate the probe's cost (the fc layers);
    a missing value is reported as missing rather than approximated.
    """
    shape = tuple(matrix.shape)
    if min(shape) > 1024 or shape[0] * shape[1] > 5_000_000:
        return None
    matrix = np.asarray(matrix.detach().cpu().double() if hasattr(matrix, "detach") else matrix)
    spectrum = np.linalg.svd(matrix, compute_uv=False)
    total = spectrum.sum()
    if total <= 0:
        return None
    p = spectrum / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()) / min(matrix.shape))


def _row_chunks(flat, budget: int = 4_000_000):
    """Slice a (units, fan_in) tensor into blocks of at most ``budget`` elements.

    VGG's first fc layer holds 102M parameters; a float64 copy of it is 0.8 GB and
    the fourth-moment term is another. Every statistic below is a mean or a
    per-row reduction, so all of them stream.
    """
    rows = max(1, budget // max(1, flat.shape[1]))
    for start in range(0, flat.shape[0], rows):
        yield flat[start : start + rows]


def weight_shape(module) -> dict:
    """Shape statistics of one layer's kernel. Every entry is gauge-invariant.

    - ``dc_fraction``  median over output units of ``|Σ W| / ||W||``: how much of a
      filter answers a *uniform* image rather than a modulation. A unit with a
      large DC fraction is pushed far from its threshold by the gray background
      alone, which is the operating point arriving from the weights' side.
    - ``sparsity``     fraction of weights below a tenth of the layer's own rms.
    - ``kurtosis``     excess kurtosis of the weight distribution.
    - ``effective_rank`` entropy rank of the spectrum, over full rank.

    The last three are what a weight decay acts on: shrinkage toward zero shows up
    as sparsity and kurtosis before it shows up anywhere else, and neither can be
    faked by a rescaling.
    """
    import torch

    w = module.weight.detach()
    flat = w.reshape(w.shape[0], -1)
    total = flat.numel()

    moments = torch.zeros(4, dtype=torch.float64)
    dc, norms = [], []
    for block in _row_chunks(flat):
        b = block.double()
        for k in range(4):
            moments[k] += b.pow(k + 1).sum()
        dc.append(b.sum(dim=1).abs())
        norms.append(b.pow(2).sum(dim=1).sqrt())
    m1, m2, m3, m4 = (float(m / total) for m in moments)

    rms = float(np.sqrt(m2))
    variance = m2 - m1**2
    # central fourth moment from the raw moments, so nothing is materialised twice
    fourth = m4 - 4 * m1 * m3 + 6 * m1**2 * m2 - 3 * m1**4

    below = 0
    if rms > 0:
        for block in _row_chunks(flat):
            below += int((block.abs() < 0.1 * rms).sum())

    ratio = torch.cat(dc) / torch.cat(norms).clamp_min(torch.finfo(torch.float64).tiny)
    return {
        "dc_fraction": float(ratio.median()),
        "sparsity": below / total if rms > 0 else float("nan"),
        "kurtosis": (fourth / variance**2 - 3.0) if variance > 0 else float("nan"),
        "effective_rank": _effective_rank(flat),
    }


# --------------------------------------------------------------------------- #
# From a margin distribution to a lambda
# --------------------------------------------------------------------------- #
MARGIN_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def margin_response(margins, contrasts) -> np.ndarray:
    """``D(c)`` implied by a population of gate margins, under a Gaussian drive.

    One rectified unit at pre-activation ``z``, driven by a zero-mean perturbation
    of scale ``s·c``, contributes ``E[(z + s·c·u)^+] − z^+`` to the distance of
    means. Writing the margin ``m = z/s`` and taking ``u`` standard normal, that
    is ``s·c·h(m/c)`` with

        h(m) = φ(m) − |m|·Φ(−|m|)

    so the layer's response is ``D(c) = c · mean_i h(m_i / c)``, up to the overall
    scale the metric cannot see anyway.

    This is a **model**, not a measurement: it assumes the perturbation reaching
    each unit is Gaussian, zero-mean and exactly proportional to contrast. The
    first two hold well where the drive is a sum of many filter responses and fail
    once earlier rectifications have skewed it. Its value is that it has no free
    parameters -- feed in the measured margins and it returns a λ that can be put
    next to the committed one.
    """
    import torch

    m = torch.as_tensor(np.asarray(margins, dtype=np.float64)).abs()
    out = []
    for c in contrasts:
        if c <= 0:
            out.append(0.0)
            continue
        v = m / c
        phi = torch.exp(-0.5 * v * v) / np.sqrt(2.0 * np.pi)
        big_phi = 0.5 * torch.erfc(v / np.sqrt(2.0))
        out.append(float(c * (phi - v * big_phi).mean()))
    return np.asarray(out)


def lambda_from_margins(margins, contrasts) -> tuple[float, float]:
    """Fit the repo's own λ to the response a margin distribution implies."""
    from .fit import fit_power_lambda

    contrasts = np.asarray(contrasts, dtype=float)
    response = margin_response(margins, contrasts)
    # The fitter needs three positive-contrast points; a short diagnostic grid
    # (or a degenerate response) reports itself missing rather than raising.
    if (contrasts > 0).sum() < 3 or not np.all(np.isfinite(response)) or response.max() <= 0:
        return float("nan"), float("nan")
    fit = fit_power_lambda(contrasts, response)
    return float(fit.lam), float(fit.r2)


# --------------------------------------------------------------------------- #
# The operating point
# --------------------------------------------------------------------------- #
@dataclass
class LayerOperatingPoint:
    """One weight layer's gauge-invariant operating point and weight shape."""

    layer: str
    kind: str
    units: int
    # Operating point, measured at the gray reference and against the gratings.
    bias_drive_ratio: float
    off_fraction: float
    margin_median: float
    margin_below_one: float
    margin_quantiles: list[float] = field(default_factory=list)
    # λ the margin distribution implies under the Gaussian-drive model, and the
    # quality of that family's fit to it. A model, not a measurement -- see
    # margin_response().
    predicted_lambda: float = float("nan")
    predicted_lambda_r2: float = float("nan")
    flip_fraction: list[float] = field(default_factory=list)
    # Weight shape.
    dc_fraction: float = float("nan")
    sparsity: float = float("nan")
    kurtosis: float = float("nan")
    effective_rank: float | None = None


class OperatingPointProbe:
    """Hooks a plain ReLU net and accumulates gate statistics over a grating grid."""

    def __init__(self, net, device: str = "cpu"):
        import torch

        self.torch = torch
        self.net = net.eval()
        self.device = torch.device(device)
        self.taps = relu_taps(net)
        if not self.taps:
            raise RuntimeError(
                "no Conv2d/Linear layer is followed directly by a ReLU: this probe "
                "is for plain rectifier stacks (VGG, AlexNet). A branching net "
                "needs its gates identified explicitly, not guessed from module order."
            )
        self._modules = dict(net.named_modules())
        self._acts: dict[str, "torch.Tensor"] = {}
        self._handles = [
            self._modules[name].register_forward_hook(self._make_hook(name))
            for name in self.taps
        ]

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            # clone(): torchvision builds VGG with inplace ReLUs, which would
            # otherwise overwrite the pre-activation before the forward returns.
            self._acts[name] = output.detach().clone()

        return hook

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _forward(self, image: np.ndarray) -> dict:
        torch = self.torch
        x = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1)[None]
        mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32)[None, :, None, None]
        std = torch.tensor(IMAGENET_STD, dtype=torch.float32)[None, :, None, None]
        with torch.no_grad():
            self.net((x - mean) / std)
        return dict(self._acts)

    def run(
        self,
        contrasts=CONTRASTS,
        frequencies=FREQUENCIES_CPI,
        reps: int = 4,
        size: int = 224,
        mean: float = 0.5,
        seed: int = 0,
    ) -> list[LayerOperatingPoint]:
        torch = self.torch
        rng = np.random.default_rng(seed)
        contrasts = list(contrasts)
        frequencies = list(frequencies)

        gray_acts = {k: v.clone() for k, v in self._forward(to_rgb(make_reference(size, mean))).items()}
        gates = {k: (v > 0) for k, v in gray_acts.items()}

        flips = {k: np.zeros(len(contrasts)) for k in self.taps}
        counts = np.zeros(len(contrasts))
        top = len(contrasts) - 1
        ac_sq = {k: torch.zeros_like(v) for k, v in gray_acts.items()}
        ac_draws = 0

        for ci, contrast in enumerate(contrasts):
            for frequency in frequencies:
                for _ in range(reps):
                    theta = rng.uniform(0.0, np.pi)
                    phase = rng.uniform(0.0, 2.0 * np.pi)
                    image = to_rgb(
                        make_grating(contrast, frequency, theta, phase, size=size, mean=mean)
                    )
                    acts = self._forward(image)
                    for name in self.taps:
                        z = acts[name]
                        flips[name][ci] += float((z > 0).ne(gates[name]).double().mean())
                        if ci == top:
                            ac_sq[name] += (z - gray_acts[name]) ** 2
                    counts[ci] += 1
                    if ci == top:
                        ac_draws += 1

        results = []
        for name in self.taps:
            module = self._modules[name]
            z = gray_acts[name].double()
            bias = module.bias
            if bias is None:
                drive, bias_rms = z, 0.0
            else:
                shape = [1, -1] + [1] * (z.dim() - 2)
                b = bias.detach().double().reshape(shape)
                drive = z - b
                # rms over units, with the bias broadcast the way it is applied.
                bias_rms = float(b.expand_as(z).pow(2).mean().sqrt())
            drive_rms = float(drive.pow(2).mean().sqrt())
            ac_rms = (ac_sq[name].double() / max(ac_draws, 1)).sqrt()
            positive = ac_rms > 0
            margin = (z.abs()[positive] / ac_rms[positive]).cpu().numpy()
            # The prediction is an average over units, so a subsample of a few
            # hundred thousand of conv1's 3.2M is already exact to well past the
            # precision anything downstream is quoted to.
            sub = margin
            if sub.size > 200_000:
                sub = np.random.default_rng(seed).choice(sub, 200_000, replace=False)
            predicted, predicted_r2 = (
                lambda_from_margins(sub, contrasts) if sub.size else (float("nan"),) * 2
            )
            results.append(
                LayerOperatingPoint(
                    layer=name,
                    kind=type(module).__name__,
                    units=int(z.numel()),
                    bias_drive_ratio=(
                        bias_rms / drive_rms if drive_rms > 0 else float("nan")
                    ),
                    off_fraction=float((z <= 0).double().mean()),
                    margin_median=float(np.median(margin)) if margin.size else float("nan"),
                    margin_below_one=float((margin < 1.0).mean()) if margin.size else float("nan"),
                    margin_quantiles=(
                        [float(q) for q in np.quantile(margin, MARGIN_QUANTILES)]
                        if margin.size
                        else []
                    ),
                    predicted_lambda=predicted,
                    predicted_lambda_r2=predicted_r2,
                    flip_fraction=list(flips[name] / np.maximum(counts, 1)),
                    **weight_shape(module),
                )
            )
        return results


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def load_net(model: str, weights: str | None, device: str = "cpu"):
    """A torchvision architecture with verified weights, or a loud failure.

    Mirrors `features.TorchvisionModel`'s refusal to fall back to random init:
    the operating point of an untrained net is not the thing being measured.
    """
    import torch
    import torchvision

    net = getattr(torchvision.models, model)(weights=None)
    if weights:
        state = torch.load(weights, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        net.load_state_dict(state)
        source = f"local state_dict: {weights}"
    else:
        net = getattr(torchvision.models, model)(weights="IMAGENET1K_V1")
        source = f"torchvision {model} IMAGENET1K_V1"
    return net.eval().to(torch.device(device)), source


def committed_lambdas(run_dir: str) -> dict[str, float]:
    """Per-layer λ from a committed run, keyed by tap name."""
    with open(os.path.join(run_dir, "result.json")) as fh:
        payload = json.load(fh)
    return {entry["layer"]: entry["lambda"] for entry in payload["layers"]}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default="vgg19", help="torchvision architecture")
    p.add_argument("--weights", default=None, help="local state_dict (blank = IMAGENET1K_V1)")
    p.add_argument("--reps", type=int, default=4, help="grating draws per (contrast, frequency)")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=None, help="write the statistics here as JSON")
    p.add_argument(
        "--compare",
        default=None,
        help="a committed results/<slug>/ whose per-layer λ is joined onto the output",
    )
    p.add_argument(
        "--gauge-check",
        action="store_true",
        help="re-measure after a random rescaling symmetry and report the largest drift",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    net, source = load_net(args.model, args.weights, args.device)
    probe = OperatingPointProbe(net, args.device)
    stats = probe.run(reps=args.reps, size=args.size, seed=args.seed)
    probe.close()

    payload = {
        "model": args.model,
        "weights_source": source,
        "reps": args.reps,
        "size": args.size,
        "seed": args.seed,
        "contrasts": list(CONTRASTS),
        "frequencies": list(FREQUENCIES_CPI),
        "layers": [asdict(s) for s in stats],
    }

    if args.compare:
        lambdas = committed_lambdas(args.compare)
        payload["compare_run"] = args.compare
        for entry in payload["layers"]:
            entry["lambda"] = lambdas.get(entry["layer"])

    if args.gauge_check:
        rng = np.random.default_rng(args.seed + 1)
        names = [name for name, _ in weight_layers(net)]
        apply_gauge(net, {n: float(np.exp(rng.uniform(-1.5, 1.5))) for n in names})
        again = OperatingPointProbe(net, args.device)
        after = again.run(reps=args.reps, size=args.size, seed=args.seed)
        again.close()
        drift = max(
            abs(getattr(a, k) - getattr(b, k))
            for a, b in zip(stats, after)
            for k in ("bias_drive_ratio", "off_fraction", "margin_median", "dc_fraction")
        )
        flip_drift = max(
            abs(x - y)
            for a, b in zip(stats, after)
            for x, y in zip(a.flip_fraction, b.flip_fraction)
        )
        payload["gauge_check"] = {"max_drift": drift, "max_flip_drift": flip_drift}

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    if not args.quiet:
        print(f"{args.model}  {source}")
        header = (
            f"{'layer':<14}{'b/drive':>9}{'off':>7}{'margin':>9}{'flip@c=1':>10}"
            f"{'DC':>7}{'sparse':>8}{'lam_pred':>10}"
        )
        if args.compare:
            header += f"{'lambda':>9}"
        print(header)
        for entry in payload["layers"]:
            line = (
                f"{entry['layer']:<14}{entry['bias_drive_ratio']:>9.3f}"
                f"{entry['off_fraction']:>7.3f}{entry['margin_median']:>9.3f}"
                f"{entry['flip_fraction'][-1]:>10.4f}{entry['dc_fraction']:>7.3f}"
                f"{entry['sparsity']:>8.3f}{entry['predicted_lambda']:>10.3f}"
            )
            if args.compare and entry.get("lambda") is not None:
                line += f"{entry['lambda']:>9.3f}"
            print(line)
        if "gauge_check" in payload:
            check = payload["gauge_check"]
            print(
                f"gauge check: max drift {check['max_drift']:.2e}, "
                f"max flip drift {check['max_flip_drift']:.2e}"
            )


if __name__ == "__main__":
    main()
