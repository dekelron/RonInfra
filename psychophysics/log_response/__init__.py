"""Reverse-engineered log-contrast-response experiment from Dekel (2017).

See README.md in this directory for the derivation and the mapping to the
paper's reported numbers.
"""

from .gratings import GratingConfig, build_grid, make_grating, make_reference
from .features import FeatureModel, TorchvisionModel, SyntheticFrontEnd, l1_distance
from .fit import fit_log_linear, summarise_layer, LayerLogResult
from .experiment import run_experiment, save_figures, ExperimentResult

__all__ = [
    "GratingConfig",
    "build_grid",
    "make_grating",
    "make_reference",
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
