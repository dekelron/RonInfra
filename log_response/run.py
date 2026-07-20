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
    python -m log_response.run --model vit_b_16 --layers encoder.layers.encoder_layer_5,encoder.ln

CLIP (needs open_clip_torch; 'prob' is a zero-shot softmax over a fixed prompt
set -- see CLIPModel)::

    python -m log_response.run --model clip:ViT-B-32 --figures out_clip/
    python -m log_response.run --model clip:ViT-B-32 --scramble --figures out_clip_scr/
    python -m log_response.run --model clip:ViT-B-32:laion2b_s34b_b79k --figures out_laion/
"""

from __future__ import annotations

import argparse

from .gratings import GratingConfig
from .features import (
    CLIPModel,
    DEFAULT_PROMPTS,
    SyntheticFrontEnd,
    TorchvisionModel,
    load_prompts,
    parse_clip_spec,
)
from .experiment import run_experiment, save_figures


def build_model(args):
    layers = [s.strip() for s in args.layers.split(",") if s.strip()] if args.layers else None
    if args.model == "synthetic":
        for value, flag in (
            (args.scramble, "--scramble"),
            (layers, "--layers"),
            (args.prompts, "--prompts"),
            (args.weights, "--weights"),
        ):
            if value:
                raise SystemExit(f"{flag} does not apply to the synthetic back-end")
        return SyntheticFrontEnd()
    if args.model == "clip" or args.model.startswith("clip:"):
        arch, tag = parse_clip_spec(args.model)
        return CLIPModel(
            arch=arch,
            pretrained_tag=args.weights or tag,
            layers=layers,
            prompts=load_prompts(args.prompts) if args.prompts else None,
            device=args.device,
            scramble=args.scramble,
            scramble_seed=args.seed,
        )
    if args.prompts:
        raise SystemExit("--prompts only applies to the CLIP back-end")
    return TorchvisionModel(
        arch=args.model,
        layers=layers,
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
        help="'synthetic' (offline), a torchvision arch name (vgg19, resnet152, "
        "...), or 'clip[:ARCH[:PRETRAINED]]' (e.g. clip:ViT-B-32, "
        "clip:ViT-B-32:laion2b_s34b_b79k)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="local weights: a torchvision state_dict path, or (with clip:) an "
        "open_clip checkpoint path overriding the pretrained tag",
    )
    p.add_argument(
        "--layers",
        default=None,
        help="comma-separated module names to tap (default: a per-arch spread; "
        "the terminal layers -- 'logits'+'prob', or 'embed'+'zs_logits'+'prob' "
        "for clip -- are always added)",
    )
    p.add_argument(
        "--prompts",
        default=None,
        help="clip only: text file with one zero-shot prompt per line "
        f"(default: built-in {len(DEFAULT_PROMPTS)}-prompt set)",
    )
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--size",
        type=int,
        default=None,
        help="image side in px (default: the model's native input size, else 224)",
    )
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

    model = build_model(args)
    size = args.size or getattr(model, "input_size", None) or 224
    cfg = GratingConfig(size=size)
    print(f"model: {args.model}" + (" (weights scrambled)" if args.scramble else ""))
    if isinstance(model, CLIPModel):
        print(
            f"zero-shot prompt set: {len(model.prompts)} prompts "
            "('prob' = softmax over prompt similarities)"
        )
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
