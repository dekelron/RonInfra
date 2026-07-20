"""CLI entry point for the log-response experiment.

Examples
--------
Offline pipeline verification (no downloaded weights, runs anywhere). Use a
small repetition count so it finishes quickly::

    python -m log_response.run --model synthetic --reps 8 --figures out/

Real ImageNet CNN (needs torchvision pretrained weights, or a local state_dict).
The fully-sampled grid is 14 contrasts x 8 frequencies x 250 reps ~= 28k passes
per model::

    python -m log_response.run --model vgg19 --figures out/
    python -m log_response.run --model vgg19 --scramble --figures out_scrambled/
    python -m log_response.run --model resnet152 --weights r152.pth --figures out/
"""

from __future__ import annotations

import argparse

from .gratings import GratingConfig
from .features import TorchvisionModel, SyntheticFrontEnd
from .experiment import run_experiment, save_figures


def build_model(args):
    if args.model == "synthetic":
        if args.scramble:
            raise SystemExit("--scramble only applies to the torchvision back-end")
        return SyntheticFrontEnd()
    return TorchvisionModel(
        arch=args.model,
        weights_path=args.weights,
        pretrained=args.weights is None,
        device=args.device,
        scramble=args.scramble,
        scramble_seed=args.seed,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="synthetic",
        help="'synthetic' (offline) or a torchvision arch name (vgg19, resnet152, ...)",
    )
    p.add_argument("--weights", default=None, help="path to a local state_dict")
    p.add_argument("--device", default="cpu")
    p.add_argument("--size", type=int, default=224, help="image side in px")
    p.add_argument(
        "--reps",
        type=int,
        default=None,
        help="random orient/phase draws per (c,f) cell (default: 250)",
    )
    p.add_argument("--seed", type=int, default=0, help="RNG / scramble seed")
    p.add_argument(
        "--scramble",
        action="store_true",
        help="scramble learned weights within each layer (control: R^2 -> 0.60)",
    )
    p.add_argument("--figures", default=None, help="directory to write figures into")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = GratingConfig(size=args.size)
    model = build_model(args)
    print(f"model: {args.model}" + (" (weights scrambled)" if args.scramble else ""))
    result = run_experiment(
        model, cfg, repetitions=args.reps, seed=args.seed, verbose=not args.quiet
    )
    print()
    print(result.report())

    if args.figures:
        paths = save_figures(result, args.figures)
        print()
        print("figures:")
        for path in paths:
            print(f"  {path}")


if __name__ == "__main__":
    main()
