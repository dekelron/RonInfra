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

Generative VLM (needs transformers + pillow; 'prob' is the next-token
distribution given a fixed instruction -- see HFVLMModel). The grid is heavy
for a 7B model, so use a GPU and shrink reps/frequencies::

    python -m log_response.run --model hf:llava-hf/llava-1.5-7b-hf \
        --device cuda --dtype float16 --reps 50 --figures out_llava/
    python -m log_response.run --model hf:Qwen/Qwen2-VL-2B-Instruct \
        --device cuda --dtype bfloat16 --frequencies 3.5,7,14,28 --reps 25

Segment Anything encoder (no 'prob' analogue -- encoder representation only;
add --mask-decoder for fixed-center-point 'mask_logits'/'iou_scores')::

    python -m log_response.run --model sam --device cuda --reps 50 --figures out_sam/
    python -m log_response.run --model sam:facebook/sam-vit-huge --mask-decoder \
        --device cuda --dtype float16 --frequencies 3.5,7,14,28 --reps 25

Persist an expensive run and re-fit / re-plot it later without re-running the
model (``--save`` writes ``<base>.npz`` + ``<base>.json``; ``--load`` reads the
npz back)::

    python -m log_response.run --model vgg19 --save runs/vgg19
    python -m log_response.run --load runs/vgg19 --figures out_vgg19/
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timezone

from .gratings import GratingConfig
from .features import (
    CLIPModel,
    DEFAULT_INSTRUCTION,
    DEFAULT_PROMPTS,
    HFVLMModel,
    SAMModel,
    SyntheticFrontEnd,
    TorchvisionModel,
    load_prompts,
    parse_clip_spec,
    parse_hf_spec,
    parse_sam_spec,
)
from .experiment import (
    run_experiment,
    save_figures,
    save_result,
    save_run_dir,
    load_result,
)
from .panels import save_panels
from .provenance import (
    environment,
    file_fingerprint,
    git_provenance,
    package_versions,
)


def build_metadata(args, model, result, wall_seconds: float) -> dict:
    """Everything needed to trust and reproduce this run.

    ``weights.pretrained_verified`` is the field that matters most: True only if
    trained weights demonstrably loaded, False if the net is random, None where
    the question does not apply (the synthetic back-end).
    """
    weights: dict = {
        "pretrained_verified": getattr(model, "weights_ok", None),
        "source": getattr(model, "weights_source", "unknown"),
    }
    if args.weights:
        weights["file"] = file_fingerprint(args.weights)
    metadata = {
        "model": args.model,
        "scramble": bool(args.scramble),
        "seed": args.seed,
        "device": args.device,
        "layers": list(result.layers),
        "weights": weights,
        "command": " ".join([os.path.basename(sys.argv[0]), *sys.argv[1:]]),
        "code": git_provenance(),
        "versions": package_versions(),
        "environment": {**environment(), "wall_seconds": round(wall_seconds, 1)},
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if args.notes:
        metadata["notes"] = args.notes
    if isinstance(model, CLIPModel):
        metadata["prompts"] = len(model.prompts)
    if isinstance(model, HFVLMModel):
        metadata["instruction"] = model.instruction
    if isinstance(model, SAMModel):
        metadata["mask_decoder"] = model.mask_decoder
    return metadata


def build_model(args):
    layers = [s.strip() for s in args.layers.split(",") if s.strip()] if args.layers else None
    if args.model == "synthetic":
        for value, flag in (
            (args.scramble, "--scramble"),
            (layers, "--layers"),
            (args.prompts, "--prompts"),
            (args.weights, "--weights"),
            (args.instruction, "--instruction"),
            (args.dtype, "--dtype"),
            (args.mask_decoder, "--mask-decoder"),
        ):
            if value:
                raise SystemExit(f"{flag} does not apply to the synthetic back-end")
        return SyntheticFrontEnd()
    is_clip = args.model == "clip" or args.model.startswith("clip:")
    is_hf = args.model.startswith("hf:")
    is_sam = args.model == "sam" or args.model.startswith("sam:")
    if args.prompts and not is_clip:
        raise SystemExit("--prompts only applies to the CLIP back-end")
    if args.instruction and not is_hf:
        raise SystemExit("--instruction only applies to the hf: VLM back-end")
    if args.dtype and not (is_hf or is_sam):
        raise SystemExit("--dtype only applies to the hf: and sam: back-ends")
    if args.mask_decoder and not is_sam:
        raise SystemExit("--mask-decoder only applies to the sam: back-end")
    if is_clip:
        arch, tag = parse_clip_spec(args.model)
        return CLIPModel(
            arch=arch,
            pretrained_tag=args.weights or tag,
            layers=layers,
            prompts=load_prompts(args.prompts) if args.prompts else None,
            device=args.device,
            scramble=args.scramble,
            scramble_seed=args.seed,
            allow_random_init=args.allow_random_init,
        )
    if is_hf:
        return HFVLMModel(
            model_id=args.weights or parse_hf_spec(args.model),
            layers=layers,
            instruction=args.instruction,
            device=args.device,
            dtype=args.dtype,
            scramble=args.scramble,
            scramble_seed=args.seed,
        )
    if is_sam:
        return SAMModel(
            model_id=args.weights or parse_sam_spec(args.model),
            layers=layers,
            device=args.device,
            dtype=args.dtype,
            mask_decoder=args.mask_decoder,
            scramble=args.scramble,
            scramble_seed=args.seed,
        )
    return TorchvisionModel(
        arch=args.model,
        layers=layers,
        weights_path=args.weights,
        pretrained=args.weights is None,
        device=args.device,
        scramble=args.scramble,
        scramble_seed=args.seed,
        allow_random_init=args.allow_random_init,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model",
        default="synthetic",
        help="'synthetic' (offline), a torchvision arch name (vgg19, resnet152, "
        "...), 'clip[:ARCH[:PRETRAINED]]' (e.g. clip:ViT-B-32, "
        "clip:ViT-B-32:laion2b_s34b_b79k), 'hf:MODEL_ID' for a generative "
        "VLM (e.g. hf:llava-hf/llava-1.5-7b-hf), or 'sam[:MODEL_ID]' for a "
        "Segment Anything encoder (default facebook/sam-vit-base)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="local weights: a torchvision state_dict path, an open_clip "
        "checkpoint (clip:), or a local model directory (hf:)",
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
    p.add_argument(
        "--instruction",
        default=None,
        help="hf only: the fixed conditioning prompt "
        f"(default: {DEFAULT_INSTRUCTION!r})",
    )
    p.add_argument(
        "--dtype",
        default=None,
        choices=["auto", "float32", "float16", "bfloat16"],
        help="hf/sam only: model dtype (use float16/bfloat16 on GPU for big models)",
    )
    p.add_argument(
        "--mask-decoder",
        action="store_true",
        help="sam only: also run the mask decoder with a fixed center-point "
        "prompt, adding 'mask_logits' and 'iou_scores' terminal layers",
    )
    p.add_argument(
        "--frequencies",
        default=None,
        help="comma-separated spatial frequencies in cycles/image "
        "(default: the full 8-frequency grid; shrink for expensive models)",
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
        help="scramble learned weights within each layer (control: collapses the "
        "prob-layer R^2; see results/vgg19-scramble-r50-s0)",
    )
    p.add_argument("--figures", default=None, help="directory to write figures into")
    p.add_argument(
        "--panels",
        default=None,
        help="write the per-layer panel figure to this .png path: one column per "
        "layer, contrast linear on the top row and log on the bottom",
    )
    p.add_argument(
        "--save",
        default=None,
        help="persist the run to <base>.npz (surfaces + grids + metadata) and "
        "<base>.json (fit summary); re-fit/re-plot later with --load",
    )
    p.add_argument(
        "--save-run",
        default=None,
        help="persist the run as a committable directory: <dir>/result.npz, "
        "result.json, run.json (full provenance) and a notes.md stub",
    )
    p.add_argument(
        "--notes",
        default=None,
        help="one-line description of what this run is for; stored in run.json "
        "and seeded into notes.md",
    )
    p.add_argument(
        "--allow-random-init",
        action="store_true",
        help="permit an untrained net when pretrained weights cannot be loaded. "
        "Off by default: the log response only exists in a trained net, so a "
        "silent fallback yields meaningless numbers. Saved runs are stamped "
        "pretrained_verified=false.",
    )
    p.add_argument(
        "--load",
        default=None,
        help="load a saved .npz and re-report/re-plot without running a model "
        "(model flags are ignored)",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    # Re-fit / re-plot a saved run without touching any model.
    if args.load:
        try:
            result, meta = load_result(args.load)
        except FileNotFoundError:
            raise SystemExit(f"--load: no saved run at {args.load!r} (expected a .npz)")
        print(f"loaded: {args.load}")
        if meta.get("model"):
            print(
                f"model: {meta['model']}"
                + (" (weights scrambled)" if meta.get("scramble") else "")
            )
        print()
        print(result.report())
        if args.save:
            written = save_result(result, args.save, metadata=meta)
            print()
            print(f"saved: {written['npz']}, {written['json']}")
        if args.figures:
            paths = save_figures(result, args.figures)
            print()
            print("figures:")
            for path in paths:
                print(f"  {path}")
        if args.panels:
            print()
            print(f"panels: {save_panels(result, args.panels, meta)}")
        return

    try:
        model = build_model(args)
    except RuntimeError as exc:
        # Untrusted-weights refusal: report it as a clean CLI failure (non-zero
        # exit), not a traceback. The message is preserved verbatim.
        raise SystemExit(f"error: {exc}")
    size = args.size or getattr(model, "input_size", None) or 224
    cfg_kwargs = {"size": size}
    if args.frequencies:
        freqs = tuple(float(s) for s in args.frequencies.split(",") if s.strip())
        if not freqs or any(f <= 0 for f in freqs):
            raise SystemExit("--frequencies needs positive cycles/image values")
        cfg_kwargs["frequencies_cpi"] = freqs
    cfg = GratingConfig(**cfg_kwargs)
    print(f"model: {args.model}" + (" (weights scrambled)" if args.scramble else ""))
    if isinstance(model, CLIPModel):
        print(
            f"zero-shot prompt set: {len(model.prompts)} prompts "
            "('prob' = softmax over prompt similarities)"
        )
    if isinstance(model, HFVLMModel):
        print(
            f"instruction: {model.instruction!r} "
            "('prob' = next-token distribution at the final position)"
        )
    if isinstance(model, SAMModel):
        print(
            "encoder-only measurement (no 'prob' analogue)"
            + (
                "; mask decoder on a fixed center-point prompt "
                "-> 'mask_logits' + 'iou_scores'"
                if model.mask_decoder
                else ""
            )
        )
    started = time.time()
    result = run_experiment(
        model, cfg, repetitions=args.reps, seed=args.seed, verbose=not args.quiet
    )
    wall_seconds = time.time() - started
    print()
    print(result.report())

    if args.save or args.save_run:
        metadata = build_metadata(args, model, result, wall_seconds)
        # A run whose weights never loaded measures nothing; refuse to persist it
        # as a result unless the caller asked for an untrained control on purpose.
        if metadata["weights"]["pretrained_verified"] is False and not args.allow_random_init:
            raise SystemExit(
                "refusing to save: the model does not carry trained weights, so "
                "these numbers are meaningless. Pass --weights, or "
                "--allow-random-init to save it as a deliberate control."
            )
        if args.save:
            written = save_result(result, args.save, metadata=metadata)
            print()
            print(f"saved: {written['npz']}, {written['json']}")
        if args.save_run:
            written = save_run_dir(result, args.save_run, metadata, notes=args.notes)
            print()
            print(f"saved run: {args.save_run}/")
            for key in ("npz", "json", "run", "notes"):
                print(f"  {os.path.basename(written[key])}")

    if args.figures:
        paths = save_figures(result, args.figures)
        print()
        print("figures:")
        for path in paths:
            print(f"  {path}")

    if args.panels:
        # Reuse the saved metadata when there is one, so the figure's weight-state
        # stamp matches what was persisted.
        meta = metadata if (args.save or args.save_run) else build_metadata(
            args, model, result, wall_seconds
        )
        print()
        print(f"panels: {save_panels(result, args.panels, meta)}")


if __name__ == "__main__":
    main()
