"""Offline self-tests for the log-response pipeline (no downloaded weights).

Verifies:
1. Input generator: gray mean preserved, Michelson contrast correct, uniform
   reference.
2. Fitting: a synthetic log law is recovered (R^2 ~ 1); a linear-in-contrast
   signal is not; log-spaced contrasts land evenly under a true log law.
3. Metric plumbing: the experiment computes the distance-of-means and the
   synthetic band-pass + compressive front-end yields a high-R^2 log-contrast
   response at its compressive "output" stage but not at the linear "energy"
   stage.
4. CLIP helpers: the ``clip:ARCH[:PRETRAINED]`` model-spec parser, the built-in
   zero-shot prompt set, and the prompt-file loader (the CLIPModel itself needs
   torch + open_clip and is exercised by the real runs, not here).
5. VLM back-end: the ``hf:MODEL_ID`` spec parser, and -- when torch +
   transformers are installed (SKIPPED otherwise, still no downloads) -- the
   full HFVLMModel plumbing against a tiny random-config LLaVA built in
   memory: default layer taps, terminal logits/prob, shape stability.
6. SAM back-end: the ``sam[:MODEL_ID]`` spec parser, and (same skip rule) the
   SAMModel plumbing against a tiny random-config SAM: encoder-only 'embed',
   and the fixed-center-point mask-decoder taps.
7. Persistence: a run saved to .npz/.json and loaded back re-fits to identical
   surfaces, fits, and report text without touching a model.

Run:  python -m pytest log_response/test_pipeline.py -q
  or: python log_response/test_pipeline.py
"""

from __future__ import annotations

import numpy as np

if __package__ in (None, ""):  # script mode: python log_response/test_pipeline.py
    import os
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import log_response  # noqa: F401  (makes the relative imports below resolvable)

    __package__ = "log_response"

from .gratings import (
    GratingConfig,
    make_grating,
    make_reference,
    sample_gratings,
    to_rgb,
    CONTRASTS,
)
from .fit import fit_log_linear, linear_spacing_uniformity
from .features import (
    DEFAULT_PROMPTS,
    FeatureModel,
    SyntheticFrontEnd,
    l1_distance,
    load_prompts,
    parse_clip_spec,
    parse_hf_spec,
    parse_sam_spec,
)
from .experiment import run_experiment, save_result, save_run_dir, load_result
from .provenance import git_provenance, package_versions


def test_grating_mean_and_contrast():
    g = make_grating(contrast=0.5, frequency_cpi=8, size=128)
    # Mean level preserved at 0.5 (same as the gray reference).
    assert abs(g.mean() - 0.5) < 1e-3
    # Michelson contrast of an unclipped grating equals the requested contrast.
    lmax, lmin = g.max(), g.min()
    michelson = (lmax - lmin) / (lmax + lmin)
    assert abs(michelson - 0.5) < 1e-2


def test_reference_uniform():
    ref = make_reference(64)
    assert np.allclose(ref, 0.5)
    assert ref.std() == 0.0


def test_contrast_grid():
    c = np.asarray(CONTRASTS)
    assert len(c) == 14
    assert abs(c[0] - 1 / 128) < 1e-9
    assert abs(c[-1] - 1.0) < 1e-9
    # log-spacing: consecutive log-gaps are roughly constant above the low end
    gaps = np.diff(np.log10(c))[3:]
    assert gaps.std() / gaps.mean() < 0.25


def test_fit_recovers_log_law():
    contrasts = np.logspace(-2, 0, 14)
    y = 3.0 * np.log10(contrasts) + 7.0
    fit = fit_log_linear(contrasts, y)
    assert fit.r2 > 0.999
    assert abs(fit.slope - 3.0) < 1e-6


def test_linear_in_contrast_is_not_log_linear():
    contrasts = np.logspace(-2, 0, 14)
    fit = fit_log_linear(contrasts, contrasts.copy())
    assert fit.r2 < 0.95


def test_log_spaced_becomes_evenly_spaced():
    # A perfect log law has constant local log-slope even on a non-uniform
    # contrast grid, once gaps are normalised by the log-contrast spacing.
    contrasts = np.asarray(CONTRASTS)
    logc = np.log10(contrasts)
    y = 3.0 * logc + 7.0
    assert linear_spacing_uniformity(y, logc) < 1e-9
    # Without normalisation, the non-uniform grid looks uneven.
    assert linear_spacing_uniformity(y) > 0.1


def test_synthetic_frontend_shows_log_response():
    # small grid; a few more reps than 1 to average the phase-invariant energy.
    cfg = GratingConfig(size=96, frequencies_cpi=(3.5, 7, 14, 28))
    result = run_experiment(SyntheticFrontEnd(), cfg, repetitions=8, verbose=False)
    out = result.results["output"]
    energy = result.results["energy"]
    assert out.mean_r2 > 0.9  # compressive stage is log-linear in contrast
    assert out.mean_r2 > energy.mean_r2  # ... more so than the linear stage


class _PhaseSignedModel(FeatureModel):
    """A model whose single unit reads the center pixel (minus gray).

    For a full-image sinusoid this is a *signed* quantity uniform in the phase,
    so it cancels when activations are averaged over random phases first but not
    when per-image absolute values are averaged. Orientation-agnostic.
    """

    def __init__(self):
        self.layers = ["signed"]

    def represent(self, image):
        h, w = image.shape[0] // 2, image.shape[1] // 2
        return {"signed": np.array([[float(image[h, w, 0]) - 0.5]])}


def test_experiment_uses_distance_of_means_not_mean_of_distances():
    cfg = GratingConfig(size=96, frequencies_cpi=(7.0,))
    reps = 60
    model = _PhaseSignedModel()
    result = run_experiment(model, cfg, repetitions=reps, seed=1, verbose=False)
    # Mean-first ordering: the signed phase-dependent activity cancels -> small D.
    d_mean_first = result.surfaces["signed"][0, -1]  # highest contrast

    # Compare against the *other* ordering (mean of per-image |.|), which does
    # not cancel and is therefore substantially larger.
    rng = np.random.default_rng(1)
    ref = model.represent(np.full((cfg.size, cfg.size, 3), 0.5))["signed"]
    per_image = [
        l1_distance(model.represent(img)["signed"], ref)
        for img in sample_gratings(1.0, 7.0, reps, rng, size=cfg.size)
    ]
    d_mean_of_dists = float(np.mean(per_image))

    assert d_mean_of_dists > 0.05  # the per-image signal is real
    assert d_mean_first < 0.4 * d_mean_of_dists  # ... but it cancels mean-first


def test_parse_clip_spec():
    assert parse_clip_spec("clip") == ("ViT-B-32", "openai")
    assert parse_clip_spec("clip:ViT-L-14") == ("ViT-L-14", "openai")
    assert parse_clip_spec("clip:ViT-B-32:laion2b_s34b_b79k") == (
        "ViT-B-32",
        "laion2b_s34b_b79k",
    )
    for bad in ("vgg19", "clip:a:b:c"):
        try:
            parse_clip_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def test_default_prompts_wellformed():
    assert len(DEFAULT_PROMPTS) >= 16  # enough classes for a meaningful softmax
    assert len(set(DEFAULT_PROMPTS)) == len(DEFAULT_PROMPTS)  # no duplicates
    assert all(p and p == p.strip() for p in DEFAULT_PROMPTS)


def test_load_prompts():
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "prompts.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("a photo of a dog\n\n  a photo of a cat  \n")
        assert load_prompts(path) == ["a photo of a dog", "a photo of a cat"]
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("only one prompt\n")
        try:
            load_prompts(path)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for a one-prompt file")


def test_save_load_roundtrip():
    import json
    import os
    import tempfile

    cfg = GratingConfig(size=64, frequencies_cpi=(3.5, 7.0, 14.0))
    result = run_experiment(SyntheticFrontEnd(), cfg, repetitions=6, seed=3, verbose=False)

    with tempfile.TemporaryDirectory() as td:
        base = os.path.join(td, "sub", "run")  # nested dir must be created
        written = save_result(result, base, metadata={"model": "synthetic"})
        assert os.path.exists(written["npz"]) and os.path.exists(written["json"])

        loaded, meta = load_result(base)  # suffix omitted on purpose
        assert meta["model"] == "synthetic"
        assert loaded.layers == result.layers
        assert loaded.repetitions == result.repetitions
        # Surfaces persisted exactly; fits re-derived identically.
        for layer in result.layers:
            assert np.allclose(loaded.surfaces[layer], result.surfaces[layer])
            assert abs(loaded.results[layer].mean_r2 - result.results[layer].mean_r2) < 1e-12
        assert loaded.report() == result.report()

        # JSON summary is valid and carries the per-layer fits.
        with open(written["json"], encoding="utf-8") as fh:
            summary = json.load(fh)
        assert summary["metadata"]["model"] == "synthetic"
        assert [d["layer"] for d in summary["layers"]] == result.layers
        assert len(summary["frequencies"]) == 3


def test_parse_hf_spec():
    assert parse_hf_spec("hf:llava-hf/llava-1.5-7b-hf") == "llava-hf/llava-1.5-7b-hf"
    assert parse_hf_spec("hf:/local/model/dir") == "/local/model/dir"
    for bad in ("hf:", "hf", "clip:ViT-B-32"):
        try:
            parse_hf_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def _tiny_llava():
    """A tiny random-config LLaVA + processor, built in memory (no downloads)."""
    import torch
    from tokenizers import Tokenizer, models as tok_models, pre_tokenizers
    from transformers import (
        CLIPImageProcessor,
        CLIPVisionConfig,
        LlamaConfig,
        LlavaConfig,
        LlavaForConditionalGeneration,
        LlavaProcessor,
        PreTrainedTokenizerFast,
    )

    vocab = {"<unk>": 0, "<pad>": 1, "<image>": 2, "<s>": 3}
    for word in ("describe", "this", "image", "a", "photo", "the", "."):
        vocab[word] = len(vocab)
    tok = Tokenizer(tok_models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        additional_special_tokens=["<image>"],
    )

    size, patch = 32, 8
    vision_cfg = CLIPVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, image_size=size, patch_size=patch,
    )
    text_cfg = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=2, vocab_size=len(vocab),
        max_position_embeddings=128, pad_token_id=1,
    )
    cfg = LlavaConfig(
        vision_config=vision_cfg, text_config=text_cfg, image_token_index=2,
        vision_feature_select_strategy="default", vision_feature_layer=-1,
    )
    torch.manual_seed(0)
    model = LlavaForConditionalGeneration(cfg)
    image_processor = CLIPImageProcessor(
        do_resize=True, size={"shortest_edge": size},
        do_center_crop=True, crop_size={"height": size, "width": size},
    )
    processor = LlavaProcessor(
        image_processor=image_processor, tokenizer=fast, patch_size=patch,
        vision_feature_select_strategy="default", num_additional_image_tokens=1,
    )
    return model, processor


def test_vlm_backend_tiny_offline():
    import unittest

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch/transformers not installed")

    from .features import HFVLMModel

    model, processor = _tiny_llava()
    m = HFVLMModel(model=model, processor=processor)
    assert m.input_size == 32
    # Default taps: last vision block, projector, mid + last LLM layer.
    assert any("vision" in layer for layer in m.layers)
    assert any("projector" in layer for layer in m.layers)

    img = to_rgb(make_grating(0.5, 7, size=m.input_size))
    acts = m.represent(img)
    assert "logits" in acts and "prob" in acts
    assert abs(float(acts["prob"].sum()) - 1.0) < 1e-5
    # Shapes must be stable across images for the distance-of-means metric.
    acts2 = m.represent(to_rgb(make_grating(1.0, 14, 1.0, 2.0, size=m.input_size)))
    assert all(acts2[k].shape == acts[k].shape for k in acts)

    # The driver runs end to end on the tiny model.
    cfg = GratingConfig(size=m.input_size, contrasts=(0.05, 0.2, 1.0), frequencies_cpi=(7.0,))
    result = run_experiment(m, cfg, repetitions=2, verbose=False)
    assert set(result.layers) == set(m.layers) | {"logits", "prob"}
    m.close()


def test_parse_sam_spec():
    assert parse_sam_spec("sam") == "facebook/sam-vit-base"
    assert parse_sam_spec("sam:facebook/sam-vit-huge") == "facebook/sam-vit-huge"
    for bad in ("sam:", "hf:x", "samx"):
        try:
            parse_sam_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {bad!r}")


def _tiny_sam():
    """A tiny random-config SAM + processor, built in memory (no downloads)."""
    import torch
    from transformers import (
        SamConfig,
        SamImageProcessor,
        SamMaskDecoderConfig,
        SamModel,
        SamProcessor,
        SamPromptEncoderConfig,
        SamVisionConfig,
    )

    size, patch = 32, 8
    # NB: HF SAM defaults to initializer_range=1e-10, which makes a *random*
    # init produce vanishing activations; use a sane range for the test model.
    vision_cfg = SamVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, image_size=size, patch_size=patch,
        output_channels=16, num_pos_feats=8,  # num_pos_feats must be hidden/2 of the prompt encoder
        global_attn_indexes=[1], window_size=2, initializer_range=0.02,
    )
    prompt_cfg = SamPromptEncoderConfig(
        hidden_size=16, image_size=size, patch_size=patch,
        image_embedding_size=size // patch,
    )
    decoder_cfg = SamMaskDecoderConfig(
        hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
        mlp_dim=32, iou_head_hidden_dim=16,
    )
    torch.manual_seed(0)
    model = SamModel(SamConfig(
        vision_config=vision_cfg,
        prompt_encoder_config=prompt_cfg,
        mask_decoder_config=decoder_cfg,
        initializer_range=0.02,
    ))
    processor = SamProcessor(image_processor=SamImageProcessor(
        size={"longest_edge": size}, pad_size={"height": size, "width": size},
    ))
    return model, processor


def test_sam_backend_tiny_offline():
    import unittest

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch/transformers not installed")

    from .features import SAMModel

    model, processor = _tiny_sam()
    m = SAMModel(model=model, processor=processor)
    assert m.input_size == 32
    assert any("vision" in layer for layer in m.layers)

    img = to_rgb(make_grating(0.5, 7, size=m.input_size))
    acts = m.represent(img)
    assert "embed" in acts and "mask_logits" not in acts  # encoder-only default
    acts2 = m.represent(to_rgb(make_grating(1.0, 14, 1.0, 2.0, size=m.input_size)))
    assert all(acts2[k].shape == acts[k].shape for k in acts)
    m.close()

    # Mask-decoder mode adds the fixed-center-point terminal taps.
    md = SAMModel(model=model, processor=processor, mask_decoder=True)
    acts = md.represent(img)
    assert "mask_logits" in acts and "iou_scores" in acts
    assert np.isfinite(acts["mask_logits"]).all()

    cfg = GratingConfig(size=md.input_size, contrasts=(0.05, 0.2, 1.0), frequencies_cpi=(7.0,))
    result = run_experiment(md, cfg, repetitions=2, verbose=False)
    assert "iou_scores" in result.layers
    md.close()


def test_torchvision_refuses_random_init_by_default():
    """The critical guard: an untrained net must fail loudly, not silently.

    A blocked weight download used to fall back to random init with only a
    warning, so the run reported plausible-looking meaningless numbers. Both the
    refusal and the opt-in escape hatch are pinned here.
    """
    import unittest

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch/torchvision not installed")
    from .features import TorchvisionModel

    # pretrained=False without an explicit opt-in is refused.
    try:
        TorchvisionModel(arch="vgg11", pretrained=False)
    except RuntimeError as exc:
        assert "random" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for an untrained net")

    # The opt-in works and is honestly labelled.
    m = TorchvisionModel(arch="vgg11", pretrained=False, allow_random_init=True)
    assert m.weights_ok is False
    assert "random" in m.weights_source
    m.close()


def test_synthetic_reports_weights_not_applicable():
    """Weight-free is None ('does not apply'), never False ('untrained')."""
    model = SyntheticFrontEnd()
    assert model.weights_ok is None
    assert "synthetic" in model.weights_source


def test_save_run_dir_records_provenance():
    """A committed run must carry commit, versions and weight state."""
    import json
    import os
    import tempfile

    model = SyntheticFrontEnd()
    cfg = GratingConfig(size=64, contrasts=(0.05, 0.2, 1.0), frequencies_cpi=(7.0, 14.0))
    result = run_experiment(model, cfg, repetitions=2, verbose=False)

    metadata = {
        "model": "synthetic",
        "command": "python -m log_response.run --model synthetic",
        "weights": {"pretrained_verified": None, "source": "synthetic (weight-free)"},
        "code": git_provenance(),
        "versions": package_versions(),
    }
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = os.path.join(tmp, "synthetic-r2-s0")
        written = save_run_dir(result, run_dir, metadata, notes="pipeline check")
        for key in ("npz", "json", "run", "notes"):
            assert os.path.exists(written[key]), key

        with open(written["run"], encoding="utf-8") as fh:
            saved = json.load(fh)
        assert saved["weights"]["pretrained_verified"] is None
        assert "numpy" in saved["versions"]
        assert "pipeline check" in open(written["notes"], encoding="utf-8").read()

        # The directory itself reloads, and re-fits identically.
        reloaded, meta = load_result(run_dir)
        assert reloaded.layers == result.layers
        for layer in result.layers:
            assert np.allclose(reloaded.surfaces[layer], result.surfaces[layer])

        # notes.md is never clobbered once written up.
        with open(written["notes"], "w", encoding="utf-8") as fh:
            fh.write("hand-written analysis")
        save_run_dir(result, run_dir, metadata)
        assert open(written["notes"], encoding="utf-8").read() == "hand-written analysis"


def test_git_provenance_reports_unavailability_explicitly():
    """A missing field must never look like a clean one."""
    prov = git_provenance()
    assert "available" in prov
    if prov["available"]:
        assert len(prov["commit"]) == 40 and "dirty" in prov
    else:
        assert prov["reason"]


def test_git_provenance_records_dirty_paths_verbatim():
    """The first dirty path must not lose its first character.

    ``git status --porcelain`` encodes index/worktree state in two leading
    columns, so an unstaged-only change begins with a space. Stripping leading
    whitespace off the whole output eats that column on the *first* line alone,
    truncating that one path -- turning `.github/x` into `github/x`, a path that
    does not exist. A trust record that names the wrong file is worse than one
    that names none.
    """
    import os
    import shutil
    import subprocess
    import tempfile
    import unittest

    if shutil.which("git") is None:
        raise unittest.SkipTest("git not available")

    with tempfile.TemporaryDirectory() as tmp:

        def git(*args):
            subprocess.run(
                ["git", *args], cwd=tmp, check=True, capture_output=True, text=True
            )

        git("init", "-q")
        git("config", "user.email", "test@example.invalid")
        git("config", "user.name", "test")
        dotfile = os.path.join(tmp, ".hidden")
        with open(dotfile, "w") as fh:
            fh.write("one\n")
        git("add", ".hidden")
        git("commit", "-qm", "seed")

        # Unstaged edit only -> the porcelain line is exactly " M .hidden".
        with open(dotfile, "w") as fh:
            fh.write("two\n")

        prov = git_provenance(tmp)
        assert prov["available"] and prov["dirty"], prov
        assert prov["dirty_files"] == [".hidden"], prov["dirty_files"]


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    import unittest

    passed = skipped = 0
    for fn in fns:
        try:
            fn()
        except unittest.SkipTest as skip:
            skipped += 1
            print(f"SKIP {fn.__name__} ({skip})")
        else:
            passed += 1
            print(f"PASS {fn.__name__}")
    print(f"\n{passed} tests passed" + (f", {skipped} skipped" if skipped else ""))


if __name__ == "__main__":
    _run_all()
