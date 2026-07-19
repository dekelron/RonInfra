"""CLI entry point for the log-response experiment.

Examples
--------
Offline pipeline verification (no downloaded weights, runs anywhere)::

    python -m psychophysics.log_response.run --model synthetic --figures out/

Real ImageNet CNN (needs torchvision pretrained weights available)::

    python -m psychophysics.log_response.run --model vgg19 --figures out/
    python -m psychophysics.log_response.run --model resnet50 \
        --weights /path/to/resnet50.pth --figures out/
"""

from __future__ import annotations

import argparse

from .gratings import GratingConfig
from .features import TorchvisionModel, SyntheticFrontEnd
from .experiment import run_experiment, save_figures


def build_model(args):
    if args.model == "synthetic":
        return SyntheticFrontEnd()
    return TorchvisionModel(
        arch=args.model,
        weights_path=args.weights,
        pretrained=args.weights is None,
        device=args.device,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="synthetic",
        help="'synthetic' (offline) or a torchvision arch name (vgg19, resnet50, ...)",
    )
    p.add_argument("--weights", default=None, help="path to a local state_dict")
    p.add_argument("--device", default="cpu")
    p.add_argument("--size", type=int, default=224, help="stimulus/image side in px")
    p.add_argument("--figures", default=None, help="directory to write figures into")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    cfg = GratingConfig(size=args.size)
    model = build_model(args)
    print(f"model: {args.model}")
    result = run_experiment(model, cfg, verbose=not args.quiet)
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
