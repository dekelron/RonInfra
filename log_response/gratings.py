"""Sinusoidal grating images for the log-contrast-response experiment.

Grating:

    I(c,f,theta,phi; x,y) = mu * [1 + c * sin(2*pi*f*(x*cos t + y*sin t)/W + phi)]

with mu the mean gray level, f in cycles-per-image, W the image width, and c the
Michelson contrast about mu. Every grating and the gray reference share the same
mean level mu, so only the modulation changes with contrast/frequency.

The experiment draws 250 images per (contrast, frequency) with **random
orientation** and **random phase**; these are generated on the fly (materialising
the full 14x8x250 grid would be tens of GB), so this module exposes single-grating
and sampler helpers rather than a pre-built array.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# 14 contrasts: integer amplitudes /128, ~half-octave apart above the low end --
# the near-geometric spacing is what makes log(c) nearly even.
CONTRASTS: tuple[float, ...] = tuple(
    a / 128.0 for a in (1, 2, 3, 4, 6, 8, 11, 16, 23, 33, 46, 64, 92, 128)
)

# 8 spatial frequencies in cycles/image.
FREQUENCIES_CPI: tuple[float, ...] = (1.0, 1.75, 3.5, 7.0, 14.0, 28.0, 56.0, 75.0)


@dataclass(frozen=True)
class GratingConfig:
    """Parameters of the grating image grid."""

    size: int = 224  # square image side, in pixels (model input size)
    contrasts: tuple[float, ...] = CONTRASTS
    frequencies_cpi: tuple[float, ...] = FREQUENCIES_CPI
    repetitions: int = 250  # random (orientation, phase) draws per (c, f)
    mean: float = 0.5

    @property
    def contrast_array(self) -> np.ndarray:
        return np.asarray(self.contrasts, dtype=np.float64)

    @property
    def frequency_array(self) -> np.ndarray:
        return np.asarray(self.frequencies_cpi, dtype=np.float64)


def _coordinate_grid(size: int) -> tuple[np.ndarray, np.ndarray]:
    """x, y in normalised [0, 1) units (cycles measured per image)."""
    axis = np.arange(size, dtype=np.float64) / size
    return np.meshgrid(axis, axis)


def make_grating(
    contrast: float,
    frequency_cpi: float,
    orientation_rad: float = 0.0,
    phase: float = 0.0,
    size: int = 224,
    mean: float = 0.5,
) -> np.ndarray:
    """One grayscale sinusoidal grating as a float array in [0, 1], shape (H, W).

    Orientation is in radians; ``frequency_cpi`` is cycles per image. Values are
    clipped to [0, 1] (a Michelson contrast of 1.0 exactly spans the range).
    """
    x, y = _coordinate_grid(size)
    proj = x * np.cos(orientation_rad) + y * np.sin(orientation_rad)
    img = mean * (1.0 + contrast * np.sin(2.0 * np.pi * frequency_cpi * proj + phase))
    return np.clip(img, 0.0, 1.0)


def make_reference(size: int = 224, mean: float = 0.5) -> np.ndarray:
    """The uniform gray reference image, single channel, in [0, 1]."""
    return np.full((size, size), mean, dtype=np.float64)


def to_rgb(gray: np.ndarray) -> np.ndarray:
    """Stack a single-channel image into an (H, W, 3) RGB array."""
    return np.repeat(gray[:, :, None], 3, axis=2)


def sample_gratings(
    contrast: float,
    frequency_cpi: float,
    repetitions: int,
    rng: np.random.Generator,
    size: int = 224,
    mean: float = 0.5,
):
    """Yield ``repetitions`` RGB gratings with random orientation and phase.

    Orientation ~ U[0, pi), phase ~ U[0, 2*pi).
    """
    for _ in range(repetitions):
        theta = rng.uniform(0.0, np.pi)
        phi = rng.uniform(0.0, 2.0 * np.pi)
        gray = make_grating(contrast, frequency_cpi, theta, phi, size=size, mean=mean)
        yield to_rgb(gray)


def reference_rgb(config: GratingConfig) -> np.ndarray:
    """RGB uniform-gray reference for a configuration."""
    return to_rgb(make_reference(config.size, config.mean))
