"""DNN feature extraction and the L1 representation-distance correlate.

The quantity measured is the *mean absolute change in DNN representation*
between the gray reference and a stimulus -- an L1 distance computed per layer.
Here we:

* run images through a torchvision CNN,
* tap a chosen set of layers with forward hooks,
* flatten each layer's activation and take the mean absolute difference from the
  reference activation.

Two model back-ends are provided:

* ``TorchvisionModel`` -- a real ImageNet-trained CNN (vgg19, resnet50, ...).
  Pretrained weights normally come from torchvision's hub. In network-restricted
  environments that host may be blocked; pass ``weights_path`` to load a local
  ``state_dict`` you fetched elsewhere. Without trained weights the log response
  does not emerge (it is a *consequence of training*), so a warning is emitted.
* ``SyntheticFrontEnd`` -- a NumPy, weight-free model whose response is a
  band-pass front-end followed by a compressive (log-like) nonlinearity. It
  exists to VERIFY the analysis pipeline offline (no downloaded weights): a
  linear stage that fails the log fit vs. a compressive stage that passes it,
  and a band-pass gain that makes the response peak at mid spatial frequency. It
  is a check on the *method*, not a model of any real network.
"""

from __future__ import annotations

from dataclasses import dataclass
import warnings
import numpy as np

# ImageNet preprocessing constants (used by the torch back-end).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def l1_distance(rep: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute difference between two flattened representations."""
    return float(np.mean(np.abs(rep.reshape(-1) - ref.reshape(-1))))


class FeatureModel:
    """Interface: map an (H, W, 3) [0,1] image to a dict of layer -> activation."""

    layers: list[str]

    def represent(self, image: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Real torchvision CNN back-end
# --------------------------------------------------------------------------- #
class TorchvisionModel(FeatureModel):
    def __init__(
        self,
        arch: str = "vgg19",
        layers: list[str] | None = None,
        weights_path: str | None = None,
        pretrained: bool = True,
        device: str = "cpu",
        scramble: bool = False,
        scramble_seed: int = 0,
    ):
        import torch
        import torchvision

        self.torch = torch
        self.device = torch.device(device)
        self.arch = arch

        model_fn = getattr(torchvision.models, arch)
        weights = None
        if pretrained and weights_path is None:
            try:
                weights = "IMAGENET1K_V1"
                self.net = model_fn(weights=weights)
            except Exception as exc:  # weight download blocked, etc.
                warnings.warn(
                    f"could not load torchvision pretrained weights ({exc}); "
                    "falling back to random init -- the log law will NOT appear. "
                    "Pass weights_path to load a local state_dict.",
                    RuntimeWarning,
                )
                self.net = model_fn(weights=None)
        else:
            self.net = model_fn(weights=None)
            if weights_path is not None:
                state = torch.load(weights_path, map_location="cpu")
                if isinstance(state, dict) and "state_dict" in state:
                    state = state["state_dict"]
                self.net.load_state_dict(state)

        # Control: permute each weight tensor's own elements (scramble the learned
        # weights within each layer). Drops the prob-layer R^2 from ~0.98 to ~0.60.
        if scramble:
            self._scramble_within_layers(scramble_seed)

        self.net.eval().to(self.device)

        self.layers = layers or self._default_layers(arch)
        self._acts: dict[str, np.ndarray] = {}
        self._register(self.layers)

    def _scramble_within_layers(self, seed: int) -> None:
        torch = self.torch
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, p in self.net.named_parameters():
                if "weight" in name and p.numel() > 1:
                    flat = p.reshape(-1)
                    perm = torch.randperm(flat.numel(), generator=gen)
                    p.copy_(flat[perm].reshape(p.shape))

    def _default_layers(self, arch: str) -> list[str]:
        # A spread from early to end computation. 'logits'/'prob' are appended in
        # represent(); for VGG-19 these tap conv1_1 (pre-ReLU), a mid conv, fc7.
        if arch.startswith("vgg"):
            return ["features.0", "features.19", "classifier.3"]
        if arch.startswith("resnet"):
            return ["layer1", "layer2", "layer3", "layer4"]
        # Generic: just tap the final module.
        return [list(dict(self.net.named_modules()).keys())[-1]]

    def _register(self, names: list[str]) -> None:
        modules = dict(self.net.named_modules())
        for name in names:
            if name not in modules:
                raise KeyError(f"layer {name!r} not found in {self.arch}")

            def hook(_m, _in, out, key=name):
                self._acts[key] = out.detach().cpu().numpy()

            modules[name].register_forward_hook(hook)

    def _preprocess(self, image: np.ndarray):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(image)).float()  # H,W,3
        x = x.permute(2, 0, 1)  # 3,H,W
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        x = (x - mean) / std
        return x.unsqueeze(0).to(self.device)

    def represent(self, image: np.ndarray) -> dict[str, np.ndarray]:
        self._acts = {}
        with self.torch.no_grad():
            logits = self.net(self._preprocess(image))
        acts = {k: v.copy() for k, v in self._acts.items()}
        # Expose both the pre-softmax logits (fc8) and the softmax probabilities.
        # The near-linear fit is strongest at 'prob'; comparing logits vs prob
        # isolates how much of the compression is the softmax (see METHOD.md).
        acts["logits"] = logits.cpu().numpy()
        acts["prob"] = self.torch.softmax(logits, dim=1).cpu().numpy()
        return acts


# --------------------------------------------------------------------------- #
# Offline synthetic front-end (no downloaded weights)
# --------------------------------------------------------------------------- #
@dataclass
class SyntheticFrontEnd(FeatureModel):
    """Weight-free stand-in used to verify the pipeline offline.

    Pipeline: a small bank of radial band-pass filters (a crude spatial-frequency
    front end), each weighted by a band-pass gain, whose energy is passed through
    a compressive nonlinearity ``log(1 + k*energy)`` and pooled. Two properties of
    the phenomenon fall out of this by design and let us check that the analysis
    reads them correctly:

    * the band-pass gain makes the low-contrast response *band-pass* in spatial
      frequency (mid frequencies respond most);
    * the compressive stage turns the (contrast^2) energy signal into a
      roughly log-in-contrast response.

    This is an existence proof for the *analysis*, not a model of any real
    network -- the real phenomenon must be measured on a trained CNN.
    """

    n_scales: int = 6
    gain: float = 200.0

    def __post_init__(self):
        self.layers = ["energy", "output"]
        # A band-pass gain over the radial scales (peaks at mid scale).
        idx = np.arange(self.n_scales, dtype=np.float64)
        peak = (self.n_scales - 1) / 2.0
        self._band_gain = np.exp(-0.5 * ((idx - peak) / 1.2) ** 2) + 0.05

    def _bandpass_energy(self, gray: np.ndarray) -> np.ndarray:
        # FFT-based band-pass energy at several radial scales.
        f = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
        power = np.abs(f) ** 2
        h, w = gray.shape
        cy, cx = h / 2, w / 2
        yy, xx = np.ogrid[:h, :w]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rmax = radius.max()
        energies = []
        edges = np.linspace(0, rmax, self.n_scales + 1)
        for i in range(self.n_scales):
            band = (radius >= edges[i]) & (radius < edges[i + 1])
            energies.append(power[band].sum() / gray.size)
        return np.asarray(energies, dtype=np.float64)

    def represent(self, image: np.ndarray) -> dict[str, np.ndarray]:
        gray = image[..., 0]  # grayscale stimuli: any channel is fine
        energy = self._band_gain * self._bandpass_energy(gray)  # weighted, ~contrast^2
        # Compressive nonlinearity => log-like response to contrast.
        output = np.log1p(self.gain * energy)
        return {"energy": energy, "output": output}
