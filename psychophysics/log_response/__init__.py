"""Reverse-engineered log-contrast-response experiment from Dekel (2017).

See METHOD.md for the exact procedure (Section 5 / Equation 4) and README.md for
the implementation and how to run it.
"""

from .gratings import (
    GratingConfig,
    PAPER_CONTRASTS,
    PAPER_FREQUENCIES_CPI,
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
    "PAPER_CONTRASTS",
    "PAPER_FREQUENCIES_CPI",
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
