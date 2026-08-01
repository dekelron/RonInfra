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
    FREQUENCIES_CPI,
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
    parse_timm_spec,
)
from .experiment import (
    run_experiment,
    save_result,
    save_run_dir,
    load_result,
    save_figures,
    result_summary,
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


def test_mean_of_distances_is_recorded_alongside_and_dominates():
    """Both orderings are measured, and Jensen fixes their relative size.

    ||E[A] - b||_1 <= E[||A - b||_1], so the mean-of-distances surface can never
    be the smaller of the two. Equality only where nothing cancels.
    """
    cfg = GratingConfig(size=96, frequencies_cpi=(7.0,))
    result = run_experiment(_PhaseSignedModel(), cfg, repetitions=40, seed=1,
                            verbose=False)
    assert result.mean_of_distances is not None
    dom = result.surfaces["signed"]
    mod = result.mean_of_distances["signed"]
    assert mod.shape == dom.shape
    assert np.all(mod >= dom - 1e-12), (dom, mod)
    # This model is built so the signed activity cancels mean-first, so the two
    # must differ by a lot, not by rounding.
    assert dom[0, -1] < 0.4 * mod[0, -1]


def test_mean_of_distances_has_no_noise_floor():
    """The point of recording it: it survives where the primary metric cannot.

    At a layer affine in the input the paper's metric has population value 0 and
    falls as 1/sqrt(reps). Taking the absolute value per image first leaves
    nothing to cancel, so the same layer keeps a rep-invariant signal.
    """
    cfg = GratingConfig(size=64, frequencies_cpi=(7.0,), contrasts=(0.25, 1.0))
    few = run_experiment(RawPixelModel(), cfg, repetitions=16, seed=0, verbose=False)
    many = run_experiment(RawPixelModel(), cfg, repetitions=256, seed=0, verbose=False)

    primary = few.surfaces["data"] / many.surfaces["data"]
    other = few.mean_of_distances["data"] / many.mean_of_distances["data"]
    assert np.all(primary > 2.0), primary          # sqrt(16) = 4 for pure noise
    assert np.allclose(other, 1.0, atol=0.05), other  # rep-invariant: real signal


def test_mean_of_distances_hits_the_closed_form_at_raw_pixels():
    """An exact calibration point, which the primary metric cannot provide.

    For a grating of Michelson contrast c about mean mu, the mean absolute
    deviation from gray is mu*c*mean|sin| = mu*c*(2/pi), i.e. c/pi at mu = 0.5 --
    independent of frequency, orientation and phase. So mean-of-distances on raw
    pixels has a closed form, and any error in the grating generator, the
    contrast convention or the metric shows up as a deviation from it.

    The paper's metric has no such check: its population value at raw pixels is
    zero, so there is nothing to compare a measurement against.
    """
    cfg = GratingConfig(size=128, frequencies_cpi=(3.5, 14.0))
    result = run_experiment(RawPixelModel(), cfg, repetitions=24, seed=0, verbose=False)
    measured = result.mean_of_distances["data"]
    expected = cfg.mean * (2.0 / np.pi) * cfg.contrast_array[None, :]
    assert np.allclose(measured, expected, rtol=0.02), measured / expected
    # ...and therefore lambda = 1 exactly, not the noise floor's 0.92.
    res = result.mod_results["data"]
    assert abs(res.lam - 1.0) < 0.02, res.lam
    assert res.lam_r2 > 0.999, res.lam_r2


def test_mean_of_distances_round_trips_and_old_runs_still_load():
    import os
    import tempfile

    cfg = GratingConfig(size=48, frequencies_cpi=(7.0,), contrasts=(0.25, 0.5, 1.0))
    result = run_experiment(RawPixelModel(), cfg, repetitions=4, seed=0, verbose=False)
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "run")
        save_result(result, base, metadata={"model": "data"})
        back, _ = load_result(base)
        assert back.mean_of_distances is not None
        assert np.allclose(back.mean_of_distances["data"],
                           result.mean_of_distances["data"])
        assert back.mod_results["data"].lam == result.mod_results["data"].lam

        # A run saved without it -- i.e. every directory committed before
        # 2026-07-27 -- must still load, with the field simply absent.
        stripped = os.path.join(tmp, "old")
        data = dict(np.load(base + ".npz", allow_pickle=False))
        data.pop("mean_of_distances")
        np.savez_compressed(stripped + ".npz", **data)
        old, _ = load_result(stripped)
        assert old.mean_of_distances is None
        assert old.mod_results is None
        assert np.allclose(old.surfaces["data"], result.surfaces["data"])
        old.report()  # must not raise with the columns absent


def test_committed_runs_predate_the_second_metric():
    """Adding it must not have disturbed a single committed surface."""
    import glob
    import os

    for npz in sorted(glob.glob("results/*/result.npz")):
        result, _ = load_result(os.path.dirname(npz))
        assert result.surfaces, npz
        result.report()


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


def test_parse_timm_spec_preserves_checkpoint_tag():
    name = "gmlp_s16_224.ra3_in1k"
    assert parse_timm_spec(f"timm:{name}") == name
    for bad in ("gmlp_s16_224.ra3_in1k", "timm:"):
        try:
            parse_timm_spec(bad)
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


def test_normalisation_census_counts_the_kinds_the_scramble_rule_turns_on():
    """The census is shared by TimmModel and HFVLMModel, so test it directly.

    Both back-ends report it, but the only test that reaches it through a
    back-end needs transformers, and the case it matters most for -- a
    BatchNorm-bearing net, where the weight scramble decalibrates rather than
    degrades -- is a timm one. This exercises the helper on a hand-built module
    so it stays covered wherever the optional deps are missing.
    """
    import unittest

    try:
        import torch
        from torch import nn
    except ImportError:
        raise unittest.SkipTest("torch not installed")

    from .features import _TorchBackend

    net = nn.Sequential(
        nn.Conv2d(3, 4, 3), nn.BatchNorm2d(4), nn.ReLU(),
        nn.Conv2d(4, 4, 3), nn.BatchNorm2d(4),
        nn.GroupNorm(2, 4), nn.Flatten(), nn.LayerNorm(4),
    )
    probe = _TorchBackend.__new__(_TorchBackend)
    probe.net = net
    census, bn = probe._normalisation_census()

    # BatchNorm is counted separately because it is the one kind that reads
    # fixed buffers rather than renormalising by the current input.
    assert bn == 2, census
    assert census["BatchNorm2d"] == 2
    assert census["GroupNorm"] == 1 and census["LayerNorm"] == 1
    assert "ReLU" not in census and "Conv2d" not in census

    # A net with nothing to desynchronise reports zero.
    _, bn_free = _TorchBackend.__new__(_TorchBackend).__class__._normalisation_census(
        type("P", (), {"net": nn.Sequential(nn.LayerNorm(4))})()
    )
    assert bn_free == 0


def test_vlm_records_the_vocabulary_the_prob_bound_depends_on():
    """``prob``'s ceiling is 2/V, so a run is uninterpretable without V.

    Every other back-end has a 1000-way softmax and the bound is a constant;
    here it is a property of the checkpoint, so it has to be recorded. Pinned
    against the actual width of the ``logits`` tap rather than against the
    config, because a padded vocabulary makes those two differ and the tap is
    what the metric sees.
    """
    import json
    import unittest

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch/transformers not installed")

    from .features import HFVLMModel

    model, processor = _tiny_llava()
    m = HFVLMModel(model=model, processor=processor)
    meta = m.model_metadata

    acts = m.represent(to_rgb(make_grating(0.5, 7, size=m.input_size)))
    assert meta["vocab_size"] == acts["logits"].shape[-1]
    assert float(acts["prob"].max()) <= 1.0

    # The measurement is conditional on this exact string; the instruction
    # alone does not reconstruct it once a chat template is involved.
    assert meta["instruction"] in meta["conditioning_text"]
    assert meta["backend"] == "hf-vlm"
    assert meta["parameter_count"] > 0
    # A LLaVA stack has no BatchNorm and plenty of LayerNorm. Both halves are
    # asserted because BN-free alone does not make the scramble valid -- the
    # census is what supports that judgement.
    assert meta["batchnorm_modules"] == 0
    assert meta["batchnorm_free"] is True
    assert sum(meta["normalisation_census"].values()) > 0
    assert any("Norm" in kind for kind in meta["normalisation_census"])
    json.dumps(meta)  # must survive the trip into run.json
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


def test_layers_all_resolves_on_the_non_torchvision_backends():
    """``--layers all`` is how the depth profile is measured, so every back-end
    has to answer it.

    The expansion used to live inside ``TorchvisionModel.__init__``, so
    ``--layers all`` on CLIP / VLM / SAM fell through as a literal layer name
    and raised ``KeyError: layer 'all' not found``: the one measurement a result
    is actually read off could not be made on any of them. It is now on the
    shared back-end.

    Exercised here on SAM because the tiny model builds offline; CLIP and the
    VLM take the same path through ``_resolve_layers``.
    """
    import unittest

    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch/transformers not installed")

    from .features import SAMModel

    model, processor = _tiny_sam()
    m = SAMModel(model=model, processor=processor, layers=["all"])

    assert len(m.layers) > 5, m.layers
    # Confined to the subtree that actually runs: encoder-only mode never calls
    # the prompt encoder or the mask decoder, and a hook on a module that never
    # fires contributes no tap while still costing a registration.
    assert all(n.startswith("vision_encoder") for n in m.layers), m.layers

    acts = m.represent(to_rgb(make_grating(0.5, 7, size=m.input_size)))
    assert "embed" in acts
    # Every registered tap fired (reuse slots may add more, never fewer).
    assert set(acts) >= set(m.layers), sorted(set(m.layers) - set(acts))
    m.close()

    # With the decoder on, the restriction lifts -- those modules now run.
    md = SAMModel(model=model, processor=processor, layers=["all"], mask_decoder=True)
    assert any(n.startswith("mask_decoder") for n in md.layers), md.layers
    md.close()


def test_timm_refuses_random_init_by_default():
    """The trap, for the timm back-end: untrained must fail loudly.

    The log response only exists in a trained net, so a back-end that quietly
    falls back to random init reports meaningless numbers that look real once
    saved. ``TorchvisionModel`` is pinned below; this is the same guard for
    ``TimmModel``, which the screening patch added without one.
    """
    import unittest

    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch/timm not installed")
    from .features import TimmModel

    try:
        TimmModel("resnet18", pretrained=False)
    except RuntimeError as exc:
        assert "allow_random_init" in str(exc) or "random" in str(exc).lower()
    else:
        raise AssertionError("expected RuntimeError for an untrained timm net")

    # The deliberate opt-in still works and stamps the run as unverified.
    model = TimmModel("resnet18", pretrained=False, allow_random_init=True)
    assert model.weights_ok is False
    assert "random" in model.weights_source.lower()


def test_timm_refuses_the_standard_scramble_on_batchnorm_nets():
    """A weight-only permutation is not a control on a net with BN buffers.

    ``--scramble`` permutes every ``*weight*`` tensor; on a BatchNorm net that
    moves gamma across channels while ``running_mean``/``running_var`` stay put,
    so each channel gets one channel's statistics and another's scale. That
    decalibrates rather than degrades -- measured, it saturates the softmax ten
    orders of magnitude below the affine regime. The back-end must refuse.
    """
    import unittest

    try:
        import torch  # noqa: F401
        import timm  # noqa: F401
    except ImportError:
        raise unittest.SkipTest("torch/timm not installed")
    from .features import TimmModel

    try:
        TimmModel("resnet18", pretrained=False, allow_random_init=True,
                  scramble=True)
    except RuntimeError as exc:
        assert "BatchNorm" in str(exc), str(exc)
    else:
        raise AssertionError("expected the BatchNorm scramble refusal")

    # A BN-free net scrambles normally -- the refusal must not be blanket.
    model = TimmModel("vit_tiny_patch16_224", pretrained=False,
                      allow_random_init=True, scramble=True)
    assert model.model_metadata["batchnorm_modules"] == 0
    # ``batchnorm_free`` is a fact, not the verdict the old
    # ``standard_scramble_valid`` name asserted: resmlp-12 is BN-free and its
    # scramble broke anyway, so the census is what a validity claim rests on.
    assert model.model_metadata["batchnorm_free"] is True
    assert sum(model.model_metadata["normalisation_census"].values()) > 0


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


def test_figure3_digitisation_recovers_the_documented_grids():
    """The paper's own figure must agree with the grids this repo runs.

    Guards both the extraction and the claim in wiki/Method.md that the grids
    are the paper's. The PDF is in-tree, so this runs offline like everything
    else. Frequencies come from pure geometry and are exact; contrasts are read
    off plotted data and carry that data's noise, so they are checked loosely.
    """
    import os

    from .figure3 import PDF, PANELS, extract

    if not os.path.exists(PDF):
        print("SKIP test_figure3_digitisation_recovers_the_documented_grids (no PDF)")
        return
    fig = extract(PDF)

    assert set(fig["panels"]) == set(PANELS)
    for name, panel in fig["panels"].items():
        assert panel.shape == (len(CONTRASTS), len(FREQUENCIES_CPI)), (name, panel.shape)

    # Frequencies are pure geometry, so they are exact.
    assert np.allclose(fig["frequencies"], FREQUENCIES_CPI, rtol=0.01), fig["frequencies"]
    # Contrasts are read off plotted data and carry its noise; the span and the
    # ordering are what the figure pins down, not the individual values.
    assert np.allclose(fig["contrasts"], CONTRASTS, rtol=0.12), fig["contrasts"]
    assert abs(fig["contrasts"][-1] - 1.0) < 1e-9
    assert np.all(np.diff(fig["contrasts"]) > 0)


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


def test_reused_modules_are_not_collapsed_into_one_tap():
    """A module called twice in one forward must yield two taps, not one.

    torchvision's ResNet holds a *single* ``nn.ReLU`` per block and calls it
    more than once -- BasicBlock after conv1 and again after the residual add,
    Bottleneck three times. One module name therefore denotes several
    activations, and a hook that assigns to ``self._acts[name]`` keeps only the
    last firing: ``--layers all`` would report the post-add activation under a
    name that reads as the first rectification and drop the others entirely.

    That is the same silent mislabelling as the in-place ReLU bug above -- the
    numbers look plausible and the layer is wrong -- and it is why no ResNet
    depth profile could be trusted before this. Firings after the first get
    their own ``<name>@n`` slot.

    VGG-19 is unaffected (each ReLU is its own module), so no committed run
    moves.
    """
    import unittest

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest(f"torch/torchvision not installed ({exc})")

    from .features import TorchvisionModel

    model = TorchvisionModel(
        arch="resnet18",
        pretrained=False,
        layers=["layer1.0.relu"],
        allow_random_init=True,
    )
    # One module in the tree -- the whole point is that it fires more than once.
    assert model._all_layers().count("layer1.0.relu") == 1

    rep = model.represent(np.random.default_rng(0).random((64, 64, 3)))
    assert "layer1.0.relu" in rep, sorted(rep)
    assert "layer1.0.relu@2" in rep, f"second firing was dropped: {sorted(rep)}"

    first, second = rep["layer1.0.relu"], rep["layer1.0.relu@2"]
    assert first.shape == second.shape
    assert not np.array_equal(first, second), "both slots hold the same tensor"

    # The metric needs a layer set that does not move between images.
    rep2 = model.represent(np.random.default_rng(1).random((64, 64, 3)))
    assert set(rep2) == set(rep)
    model.close()


def test_taps_that_never_fire_are_reported_not_silently_dropped():
    """Hooking a module that never runs must say so.

    torchvision's ViT builds ``nn.MultiheadAttention``, whose forward passes
    ``out_proj.weight``/``bias`` to ``F.multi_head_attention_forward`` instead
    of calling the module -- so the hook never fires and the tap is absent.
    ``--layers all`` on vit_b_16 registers 75 modules and 12 of them produce
    nothing. The run is still correct; the danger is discovering afterwards
    that a depth profile was missing every attention output projection.
    """
    import unittest
    import warnings as warnings_mod

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest(f"torch/torchvision not installed ({exc})")

    from .features import TorchvisionModel

    model = TorchvisionModel(
        arch="vit_b_16", pretrained=False, layers=["all"], allow_random_init=True
    )
    rep = model.represent(np.random.default_rng(0).random((224, 224, 3)))
    unfired = [n for n in model.layers if n not in rep]
    assert unfired and all("out_proj" in n for n in unfired), unfired

    cfg = GratingConfig(size=224, contrasts=(0.05, 1.0), frequencies_cpi=(7.0,))
    with warnings_mod.catch_warnings(record=True) as caught:
        warnings_mod.simplefilter("always")
        result = run_experiment(model, cfg, repetitions=1, verbose=False)
    messages = [str(w.message) for w in caught]
    assert any("never fired" in m for m in messages), messages
    # What is measured is exactly what ran -- no empty taps in the result.
    assert not (set(result.layers) & set(unfired))
    model.close()


def test_vgg_all_layers_is_unchanged_by_the_reuse_fix():
    """The 45-tap VGG-19 layer set must be exactly what was committed.

    Every committed depth profile is a 45-tap VGG-19 run, and comparability
    with them is the reason ``_all_layers`` keeps its original predicate. If
    hoisting it or adding the reuse slots moved this list, the new runs would
    not be comparable with the old ones.
    """
    import unittest

    try:
        import torch  # noqa: F401
        import torchvision  # noqa: F401
    except Exception as exc:
        raise unittest.SkipTest(f"torch/torchvision not installed ({exc})")

    from .features import TorchvisionModel

    model = TorchvisionModel(
        arch="vgg19", pretrained=False, layers=["all"], allow_random_init=True
    )
    # 43 leaf modules; 'logits' and 'prob' are added in represent() for 45.
    assert len(model.layers) == 43, len(model.layers)
    assert model.layers[0] == "features.0"
    assert model.layers[-1] == "classifier.6"
    assert not any("@" in name for name in model.layers)

    rep = model.represent(np.random.default_rng(0).random((64, 64, 3)))
    # No module fires twice in VGG, so no reuse slot may appear.
    assert set(rep) == set(model.layers) | {"logits", "prob"}, sorted(rep)
    model.close()


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


def test_result_json_records_the_lambdas_the_median_is_taken_over():
    """The headline ``lambda`` is a median over frequency; the eight must persist.

    They vary by more than the architecture differences the median gets
    compared on, so a committed directory that states only the median cannot
    support the comparison being read off it. ``result.json`` is the artifact
    a reader has -- re-deriving these from the surfaces needs the fitter.
    """
    import json

    model = SyntheticFrontEnd()
    cfg = GratingConfig(size=64, contrasts=(0.05, 0.2, 0.6, 1.0), frequencies_cpi=(7.0, 14.0, 28.0))
    result = run_experiment(model, cfg, repetitions=2, verbose=False)
    report = json.loads(json.dumps(result_summary(result)))  # must survive a round trip

    for entry in report["layers"]:
        res = result.results[entry["layer"]]
        per_freq = entry["per_frequency"]
        assert len(per_freq) == len(cfg.frequencies_cpi)

        recorded = [cell["lambda"] for cell in per_freq]
        fitted = [pf.lam for pf in res.power_fits]
        assert np.allclose(recorded, fitted, equal_nan=True), entry["layer"]

        # The median of what is recorded must be the headline number, or the
        # file contradicts itself.
        assert np.isclose(float(np.nanmedian(recorded)), entry["lambda"]), entry["layer"]
        assert np.isclose(
            float(np.nanmean([cell["lambda_r2"] for cell in per_freq])), entry["lambda_r2"]
        ), entry["layer"]

        for cell, pf in zip(per_freq, res.power_fits):
            lo, hi = cell["lambda_ci"]
            assert np.isclose(lo, pf.lo) and np.isclose(hi, pf.hi)
            assert lo <= cell["lambda"] <= hi, (entry["layer"], cell["frequency"])
            # The log fit's r2 is a different statistic and keeps its own key.
            assert "r2" in cell and "lambda_r2" in cell


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
