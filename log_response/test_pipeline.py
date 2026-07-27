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
from .fit import (
    fit_log_linear,
    fit_power_lambda,
    linear_spacing_uniformity,
    power_basis,
    summarise_layer,
)
from .features import (
    DEFAULT_PROMPTS,
    FeatureModel,
    RawPixelModel,
    SyntheticFrontEnd,
    l1_distance,
    load_prompts,
    parse_clip_spec,
    parse_hf_spec,
    parse_sam_spec,
)
from .experiment import (
    run_experiment,
    save_result,
    save_run_dir,
    load_result,
    save_figures,
)
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


def test_lambda_calibration():
    """lambda must recover the exponent of a response built to have one.

    Every case here has a known answer by construction, on BOTH contrast grids:
    the exponent is a property of the response, not of how the contrast axis
    was sampled. This is what the retired ``logness`` could not do -- its scale
    moved with the grid (ceiling 0.264 log-spaced, 0.294 linear), so a run on
    one grid was not comparable with a run on the other.
    """
    grids = {
        "log": np.asarray(CONTRASTS, dtype=np.float64),
        "linear": np.linspace(min(CONTRASTS), max(CONTRASTS), len(CONTRASTS)),
    }
    freqs = np.asarray([4.0])
    for name, c in grids.items():
        def lam_of(response):
            surface = np.asarray(response, dtype=np.float64)[None, :]
            return summarise_layer("x", c, freqs, surface).lam

        # ln and log10 differ by a constant factor, which b absorbs: both are
        # the log law, both must give exactly 0.
        assert abs(lam_of(np.log10(c))) < 1e-6, name
        assert abs(lam_of(np.log(c))) < 1e-6, name
        # An affine response in c is lam = 1 whatever a and b are.
        assert abs(lam_of(c.copy()) - 1.0) < 1e-6, name
        assert abs(lam_of(5.0 * c - 2.0) - 1.0) < 1e-6, name
        # ...and the family reaches everything between and outside.
        for expected, shape in ((0.5, np.sqrt(c)), (2.0, c ** 2),
                                (3.0, c ** 3), (-0.5, c ** -0.5)):
            assert abs(lam_of(shape) - expected) < 1e-5, (name, expected)


def test_lambda_reports_its_own_ignorance_on_noise():
    """An unstructured response must widen the interval, not fake an exponent.

    The retired statistic answered 0 for "the two laws tie" and 0 for "neither
    law fits", and one scalar could not separate them. lambda separates them by
    construction: the point estimate is meaningless on noise, so the test pins
    the *interval*, which must span essentially the whole search range, and the
    R2, which must be low.
    """
    c = np.asarray(CONTRASTS, dtype=np.float64)
    freqs = np.asarray([4.0])
    rng = np.random.default_rng(0)
    widths, r2s = [], []
    for _ in range(200):
        layer = summarise_layer("x", c, freqs, rng.normal(size=c.size)[None, :])
        lo, hi = layer.lam_ci
        widths.append(hi - lo)
        r2s.append(layer.lam_r2)
    # A clean log response pins lambda to well under 0.2 of range; noise must
    # not come anywhere near that.
    assert float(np.median(widths)) > 3.0, float(np.median(widths))
    assert float(np.mean(r2s)) < 0.5, float(np.mean(r2s))

    tight = summarise_layer("x", c, freqs, np.log10(c)[None, :])
    assert (tight.lam_ci[1] - tight.lam_ci[0]) < 0.2
    assert tight.lam_r2 > 0.999


def test_lambda_interval_contains_its_own_estimate():
    """The interval is bisected, not read off the search grid.

    Read off a 0.05-spaced grid, a minimum at 1.18 reported [1.20, 1.20] -- an
    interval excluding the estimate it brackets, and a zero width where the fit
    was merely very good. Both were real: features.2 of the Caffe 45-tap run.
    """
    c = np.asarray(CONTRASTS, dtype=np.float64)
    rng = np.random.default_rng(3)
    for target in (0.0, 0.18, 0.5, 1.18, 1.53, 2.0, -0.4):
        y = power_basis(c, target) + 1e-4 * rng.normal(size=c.size)
        fit = fit_power_lambda(c, y)
        assert fit.lo <= fit.lam <= fit.hi, (target, fit.lo, fit.lam, fit.hi)
        assert fit.hi > fit.lo, (target, fit.lo, fit.hi)


def test_lambda_survives_the_grid_it_is_measured_on():
    """The same response shape returns the same exponent on either grid."""
    freqs = np.asarray([4.0])
    for shape, expected in ((np.log10, 0.0), (np.sqrt, 0.5)):
        got = []
        for c in (np.asarray(CONTRASTS, dtype=np.float64),
                  np.linspace(min(CONTRASTS), max(CONTRASTS), len(CONTRASTS))):
            got.append(summarise_layer("x", c, freqs, shape(c)[None, :]).lam)
        assert abs(got[0] - got[1]) < 1e-5, (expected, got)
        assert abs(got[0] - expected) < 1e-5, (expected, got)


def test_power_basis_is_continuous_through_the_log_law():
    """(c^lam - 1)/lam must approach ln c as lam -> 0, not blow up or flatten.

    The reason for the Box-Cox form over a bare ``c**lam``: the latter tends to
    the constant 1, which would put the log law outside the family altogether.
    """
    c = np.asarray(CONTRASTS, dtype=np.float64)
    exact = np.log(c)
    # The expansion is ln c + lam*(ln c)^2/2 + O(lam^2), so the error is
    # first-order in lam and the bound has to be too -- a fixed tolerance would
    # be testing the contrast range rather than the continuity.
    bound = np.max(np.log(c) ** 2)
    for lam in (1e-3, 1e-5, 1e-7, 0.0, -1e-7, -1e-5, -1e-3):
        got = power_basis(c, lam)
        assert np.max(np.abs(got - exact)) <= abs(lam) * bound, lam
    # ...and it is genuinely different from the naive basis nearby.
    assert np.max(np.abs(power_basis(c, 1e-3) - c ** 1e-3)) > 1.0


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


def test_raw_pixels_report_weights_not_applicable():
    model = RawPixelModel()
    assert model.weights_ok is None
    assert model.layers == ["data"]


def test_affine_layer_measures_only_sampling_noise():
    """D at a layer upstream of every nonlinearity is a 1/sqrt(reps) noise floor.

    Phase ~ U[0, 2pi) makes E[grating] = gray exactly, so the distance-of-means
    metric has population value 0 at any affine layer. This pins the property
    that ``features.0`` turns out to be measuring (wiki/Results.md): D must fall
    as 1/sqrt(reps) rather than stay put, which is what separates a dead tap
    from a linear-responding one.
    """
    cfg = GratingConfig(size=64, frequencies_cpi=(7.0,), contrasts=(0.25, 1.0))
    few = run_experiment(RawPixelModel(), cfg, repetitions=16, seed=0, verbose=False)
    many = run_experiment(RawPixelModel(), cfg, repetitions=256, seed=0, verbose=False)
    ratio = few.surfaces["data"] / many.surfaces["data"]
    # sqrt(16) = 4 exactly for the underlying sd; D is a mean of absolute values
    # (a biased functional), so allow a generous band that still excludes 1.
    assert np.all(ratio > 2.0), ratio
    assert np.all(ratio < 6.0), ratio


def test_noise_floor_still_carries_the_frequency_profile():
    """The floor is on the contrast axis only.

    D = c * mean|W . gbar_f|, and gbar_f stays spectrally concentrated at f, so
    an affine layer's *frequency* profile is a real measurement of its filter
    bank even though its magnitude is 1/sqrt(reps) and its lambda is forced to
    1. Pins the distinction that wiki/Results.md rests on: a band-pass filter
    must show frequency structure far beyond the raw-pixel floor.
    """
    class _LowPass(FeatureModel):
        """One unit: the image blurred by a wide Gaussian, i.e. strongly low-pass."""

        def __init__(self):
            self.layers = ["blur"]
            self.weights_ok = None
            self.weights_source = "synthetic low-pass"

        def represent(self, image):
            g = np.asarray(image[..., 0], dtype=np.float64)
            f = np.fft.fftshift(np.fft.fft2(g))
            h, w = g.shape
            yy, xx = np.ogrid[:h, :w]
            r = np.sqrt((yy - h / 2) ** 2 + (xx - w / 2) ** 2)
            return {"blur": np.real(np.fft.ifft2(np.fft.ifftshift(f * np.exp(-(r / 3.0) ** 2))))}

    cfg = GratingConfig(
        size=64, frequencies_cpi=(1.0, 4.0, 16.0), contrasts=(0.125, 0.25, 0.5, 1.0)
    )
    flat = run_experiment(RawPixelModel(), cfg, repetitions=24, seed=0, verbose=False)
    tuned = run_experiment(_LowPass(), cfg, repetitions=24, seed=0, verbose=False)

    spread = lambda s: float(s[:, -1].max() / s[:, -1].min())
    # Raw pixels have no filter, so their profile is flat to within noise; the
    # low-pass model must fall off with frequency by far more than that.
    assert spread(flat.surfaces["data"]) < 3.0, spread(flat.surfaces["data"])
    assert spread(tuned.surfaces["blur"]) > 10.0, spread(tuned.surfaces["blur"])
    # ...while both stay consistent with lambda = 1 on this short grid: same
    # contrast axis, different frequency axis. Read via the interval, because
    # four contrast points do not pin the exponent tightly.
    for res in (flat.results["data"], tuned.results["blur"]):
        lo, hi = res.lam_ci
        assert lo <= 1.0 <= hi, (res.layer, res.lam, res.lam_ci)


def test_noise_floor_is_linear_in_contrast_whatever_the_grid():
    """...and its shape is exactly linear in contrast, so lambda ~ 1.

    D = c * mean|gbar| with gbar independent of c. This is why lambda ~ 1 at a
    high power-family R^2 cannot on its own be read as "this layer responds
    linearly to contrast" -- an empty tap looks identical.
    """
    cfg = GratingConfig(size=64, frequencies_cpi=(7.0,))
    result = run_experiment(RawPixelModel(), cfg, repetitions=24, seed=0, verbose=False)
    res = result.results["data"]
    assert abs(res.lam - 1.0) < 0.25, res.lam
    assert res.lam_r2 > 0.9, res.lam_r2


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


def test_panel_frequency_colours_scale_with_count():
    """Ordered frequencies get an ordered ramp at any grid size."""
    from .panels import ORDINAL_STEPS, frequency_colours

    assert frequency_colours(1) == [ORDINAL_STEPS[2]]
    # At or below the discrete-step count, use the validated steps end to end.
    for n in range(2, len(ORDINAL_STEPS) + 1):
        got = frequency_colours(n)
        assert len(got) == n and len(set(got)) == n
        assert got[0] == ORDINAL_STEPS[0] and got[-1] == ORDINAL_STEPS[-1]
    # Past them, interpolate the band -- still monotone, still distinct.
    for n in (8, 14):
        got = frequency_colours(n)
        assert len(got) == n and len(set(got)) == n


def test_save_panels_writes_a_figure():
    import os
    import tempfile

    model = SyntheticFrontEnd()
    cfg = GratingConfig(size=64, contrasts=(0.05, 0.2, 1.0), frequencies_cpi=(7.0, 14.0))
    result = run_experiment(model, cfg, repetitions=2, verbose=False)
    from .panels import save_panels

    with tempfile.TemporaryDirectory() as tmp:
        out = save_panels(
            result,
            os.path.join(tmp, "sub", "panels.png"),  # nested dir is created
            {"model": "synthetic", "weights": {"pretrained_verified": None}},
        )
        assert os.path.exists(out) and os.path.getsize(out) > 5000


def test_lambda_profile_and_figures_are_written():
    """--figures emits the depth profile, not just the per-layer panels.

    The profile is how the depth result is read, so it has to come out of the
    CLI rather than being rebuilt by hand each time. Needs >= 4 contrasts:
    lambda costs three parameters, so the interval has a degree of freedom.
    """
    import os
    import tempfile

    model = SyntheticFrontEnd()
    cfg = GratingConfig(size=64, contrasts=(0.05, 0.1, 0.3, 1.0), frequencies_cpi=(7.0, 14.0))
    result = run_experiment(model, cfg, repetitions=2, verbose=False)
    from .panels import save_lambda_profile

    with tempfile.TemporaryDirectory() as tmp:
        out = save_lambda_profile(
            result,
            os.path.join(tmp, "nested", "lambda.png"),  # nested dir is created
            {"model": "synthetic", "weights": {"pretrained_verified": None}},
        )
        assert os.path.exists(out) and os.path.getsize(out) > 5000

        paths = save_figures(result, os.path.join(tmp, "figs"))
        assert any(p.endswith("lambda_profile.png") for p in paths), paths
        assert all(os.path.exists(p) and os.path.getsize(p) > 2000 for p in paths)


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


def test_pre_activation_taps_survive_inplace_relu():
    """A conv tap must not come back holding its ReLU's output.

    torchvision builds VGG with ``nn.ReLU(inplace=True)``. On CPU with float32
    activations, ``.float().cpu().numpy()`` returns a *view* onto the module's
    own storage, so the in-place ReLU overwrites the captured conv output before
    the forward returns. The failure is silent and plausible-looking: the conv
    tap equals the ReLU tap bit-for-bit and its surface simply has no negative
    values, which reads as a real measurement of a rectified layer.
    """
    import unittest

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest(f"torch/torchvision not installed ({exc})")

    from .features import TorchvisionModel

    model = TorchvisionModel(
        arch="vgg19",
        pretrained=False,
        layers=["features.0", "features.1"],
        allow_random_init=True,
    )
    rep = model.represent(np.random.default_rng(0).random((64, 64, 3)))
    conv, relu = rep["features.0"], rep["features.1"]

    assert not np.array_equal(conv, relu), "conv tap captured its ReLU's output"
    assert conv.min() < 0, "pre-activation conv output has no negative values"
    assert relu.min() == 0, "post-ReLU output should be rectified"


def test_preprocessing_fold_is_exact():
    """The Caffe->torchvision conv1 fold must introduce no input gain.

    This is the arithmetic under suspicion when a converted checkpoint disagrees
    with the canonical one: a gain error rescales a grating's effective contrast
    and slides the whole contrast-response curve along its own axis, which would
    look exactly like a difference between checkpoints. Checked here against a
    directly-computed caffe path, on small random tensors so it needs neither
    h5py nor the 575 MB checkpoint.
    """
    from .convert_weights import (
        CAFFE_MEAN_RGB,
        IMAGENET_MEAN,
        IMAGENET_STD,
        fold_preprocessing,
    )

    rng = np.random.default_rng(0)
    weight = rng.normal(size=(4, 3, 3, 3))  # (out, in, kh, kw), RGB input
    bias = rng.normal(size=4)
    image = rng.random((3, 16, 16))  # RGB in [0,1]

    folded_w, folded_b = fold_preprocessing(weight, bias)

    def conv(x, w, b):
        """Valid (unpadded) correlation -- the fold is exact on the interior."""
        out = np.zeros((w.shape[0], x.shape[1] - 2, x.shape[2] - 2))
        for o in range(w.shape[0]):
            for i in range(x.shape[1] - 2):
                for j in range(x.shape[2] - 2):
                    out[o, i, j] = (x[:, i : i + 3, j : j + 3] * w[o]).sum() + b[o]
        return out

    # (a) caffe path: [0,255] RGB minus the caffe mean, original kernel.
    caffe_x = image * 255.0 - CAFFE_MEAN_RGB[:, None, None]
    expected = conv(caffe_x, weight, bias)

    # (b) folded path: torchvision-normalised input, folded kernel.
    norm_x = (image - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
    got = conv(norm_x, folded_w, folded_b)

    scale = np.abs(expected).max()
    assert np.abs(expected - got).max() / scale < 1e-12, "fold is not exact"

    # A pure gain error would scale one path against the other; pin it at 1.
    gain = float((expected * got).sum() / (got * got).sum())
    assert abs(gain - 1.0) < 1e-12, f"input gain {gain} != 1"


def test_weights_digest_is_stable_across_resaves():
    """The trust record must survive a re-save; a file hash does not.

    ``torch.save`` is not byte-reproducible, so ``run.json``'s file sha256
    changes when a conversion is regenerated even though the weights are
    identical -- which would read as a provenance failure. The weights digest is
    what actually pins a checkpoint, so it is pinned here.
    """
    import os
    import tempfile
    import unittest

    try:
        import torch
    except ImportError:
        raise unittest.SkipTest("torch not installed")
    from .provenance import file_fingerprint, state_dict_digest

    state = {
        "b.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "a.bias": torch.tensor([1.5, -2.0]),
    }
    with tempfile.TemporaryDirectory() as tmp:
        one, two = os.path.join(tmp, "one.pth"), os.path.join(tmp, "two.pth")
        torch.save(state, one)
        torch.save(state, two)
        first, second = file_fingerprint(one), file_fingerprint(two)

        assert first["weights_sha256"] == second["weights_sha256"]
        assert first["weights_sha256"] == state_dict_digest(state)
        # Key order must not change the digest; tensor values must.
        assert state_dict_digest(dict(reversed(list(state.items())))) == state_dict_digest(state)
        changed = dict(state, **{"a.bias": torch.tensor([1.5, -2.5])})
        assert state_dict_digest(changed) != state_dict_digest(state)

        # A file that is not a state_dict says so rather than going silent.
        text = os.path.join(tmp, "not-weights.bin")
        with open(text, "wb") as fh:
            fh.write(b"not a checkpoint")
        info = file_fingerprint(text)
        assert info["sha256"] and info["weights_sha256"] is None and info["weights_note"]


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
