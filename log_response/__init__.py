"""Log-contrast-response experiment.

See METHOD.md for the exact procedure (stimuli, metric, fit) and README.md for
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
from .features import FeatureModel, TorchvisionModel, SyntheticFrontEnd, l1_distance
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
    "SyntheticFrontEnd",
    "l1_distance",
    "fit_log_linear",
    "summarise_layer",
    "LayerLogResult",
    "run_experiment",
    "save_figures",
    "ExperimentResult",
]
