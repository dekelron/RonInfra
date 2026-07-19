"""Sinusoidal grating stimuli for the log-contrast-response experiment.

Reverse-engineered from Dekel (2017), "Human perception in computer vision"
(arXiv:1701.04674). The relevant measurement uses sinusoidal gratings at all
combinations of spatial frequency and contrast, compared against a uniform gray
reference image, and reads out the change in a DNN's internal representation.

Conventions used here (documented so they can be defended in an interview):

* Luminance is represented in [0, 1] pixel space with a mid-gray background of
  0.5. This makes the mean luminance of every grating exactly 0.5, so the only
  thing that changes with contrast/frequency is the modulation -- the reference
  gray and every grating share the same mean.
* Contrast is Michelson contrast, ``c = (Lmax - Lmin) / (Lmax + Lmin)``. With a
  mean of 0.5 and amplitude ``a`` the luminance is ``0.5 + a * sin(...)`` and
  ``c = a / 0.5 = 2a``; i.e. amplitude ``a = c / 2``. A contrast of 1.0 spans
  the full [0, 1] range; higher contrasts are clipped and therefore avoided.
* Spatial frequency is specified in cycles-per-image so it is resolution
  independent for a fixed model input size.
* Gratings are grayscale (identical R, G, B) so the manipulation is purely
  achromatic contrast, matching the psychophysics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(frozen=True)
class GratingConfig:
    """Parameters of the grating stimulus grid."""

    size: int = 224  # square image side, in pixels (model input size)
    # Log-spaced Michelson contrasts. The log spacing is the whole point: the
    # 2017 result is that these log-spaced inputs become *linearly* spaced in
    # the end-layer representation.
    contrasts: tuple[float, ...] = tuple(
        np.round(np.logspace(np.log10(0.01), np.log10(1.0), 12), 6)
    )
    # Spatial frequencies in cycles per image.
    frequencies_cpi: tuple[float, ...] = (2, 4, 8, 16, 32, 64)
    # Phases (radians) to average over, to remove phase-specific sampling
    # artifacts. Averaging over 4 quadrature phases is a standard trick.
    phases: tuple[float, ...] = (0.0, np.pi / 2, np.pi, 3 * np.pi / 2)
    orientation_deg: float = 90.0  # 90 = vertical bars (grating varies in x)
    mean: float = 0.5


def _coordinate_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x, y in cycles-normalised units on [0, 1)."""
    axis = np.arange(size, dtype=np.float64) / size
    x, y = np.meshgrid(axis, axis)
    return x, y


def make_grating(
    frequency_cpi: float,
    contrast: float,
    phase: float = 0.0,
    orientation_deg: float = 90.0,
    size: int = 224,
    mean: float = 0.5,
) -> np.ndarray:
    """Create one grayscale sinusoidal grating as a float array in [0, 1].

    Returns an ``(size, size)`` array (single channel).
    """
    x, y = _coordinate_grid(size)
    theta = np.deg2rad(orientation_deg)
    # Project coordinates onto the orientation direction; frequency is in
    # cycles per image, so multiply the [0,1) coordinate by 2*pi*freq.
    proj = x * np.cos(theta) + y * np.sin(theta)
    amplitude = contrast * mean  # Michelson contrast about `mean`
    img = mean + amplitude * np.sin(2.0 * np.pi * frequency_cpi * proj + phase)
    return np.clip(img, 0.0, 1.0)


def make_reference(size: int = 224, mean: float = 0.5) -> np.ndarray:
    """The uniform gray reference image, single channel, in [0, 1]."""
    return np.full((size, size), mean, dtype=np.float64)


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Stack a single-channel image into an (H, W, 3) RGB array."""
    return np.repeat(gray[:, :, None], 3, axis=2)


@dataclass
class StimulusGrid:
    """Materialised grid of stimuli for a configuration.

    ``images`` is indexed ``[freq_idx, contrast_idx, phase_idx]`` and each entry
    is an ``(H, W, 3)`` float image in [0, 1]. ``reference`` is the single gray
    image every grating is compared against.
    """

    config: GratingConfig
    reference: np.ndarray
    images: np.ndarray = field(repr=False)

    @property
    def contrasts(self) -> np.ndarray:
        return np.asarray(self.config.contrasts, dtype=np.float64)

    @property
    def frequencies(self) -> np.ndarray:
        return np.asarray(self.config.frequencies_cpi, dtype=np.float64)


def build_grid(config: GratingConfig | None = None) -> StimulusGrid:
    """Build the full frequency x contrast x phase stimulus grid."""
    cfg = config or GratingConfig()
    freqs = cfg.frequencies_cpi
    contrasts = cfg.contrasts
    phases = cfg.phases
    imgs = np.empty((len(freqs), len(contrasts), len(phases), cfg.size, cfg.size, 3))
    for fi, f in enumerate(freqs):
        for ci, c in enumerate(contrasts):
            for pi, p in enumerate(phases):
                g = make_grating(
                    frequency_cpi=f,
                    contrast=c,
                    phase=p,
                    orientation_deg=cfg.orientation_deg,
                    size=cfg.size,
                    mean=cfg.mean,
                )
                imgs[fi, ci, pi] = to_rgb(g)
    ref = to_rgb(make_reference(cfg.size, cfg.mean))
    return StimulusGrid(config=cfg, reference=ref, images=imgs)
