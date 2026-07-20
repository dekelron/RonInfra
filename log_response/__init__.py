"""Log-contrast-response experiment.

See METHOD.md for the exact procedure (inputs, metric, fit) and README.md for
the implementation and how to run it.
"""

from .gratings import (
    GratingConfig,
    CONTRASTS,
    FREQUENCIES_CPI,
    make_grating,
    make_reference,
    sample_gratings,
    reference_rgb,
    to_rgb,
)
from .features import (
    FeatureModel,
    TorchvisionModel,
    CLIPModel,
    SyntheticFrontEnd,
    l1_distance,
    DEFAULT_PROMPTS,
    parse_clip_spec,
    load_prompts,
)
from .fit import fit_log_linear, summarise_layer, LayerLogResult
from .experiment import run_experiment, save_figures, ExperimentResult

__all__ = [
    "GratingConfig",
    "CONTRASTS",
    "FREQUENCIES_CPI",
    "make_grating",
    "make_reference",
    "sample_gratings",
    "reference_rgb",
    "to_rgb",
    "FeatureModel",
    "TorchvisionModel",
    "CLIPModel",
    "SyntheticFrontEnd",
    "l1_distance",
    "DEFAULT_PROMPTS",
    "parse_clip_spec",
    "load_prompts",
    "fit_log_linear",
    "summarise_layer",
    "LayerLogResult",
    "run_experiment",
    "save_figures",
    "ExperimentResult",
]
