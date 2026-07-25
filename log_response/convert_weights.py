"""Convert the original Caffe/Keras VGG-19 ImageNet weights to torchvision layout.

Why this exists: where `download.pytorch.org` is blocked (this sandbox), the
Oxford VGG-19 weights are still reachable as Keras HDF5 on
`storage.googleapis.com`. They are also the *reference* checkpoint for
reproducing the original paper, so this is not only a fallback.

Four differences have to be bridged, and the fourth is the delicate one:

1. kernel layout   ``(kh,kw,in,out)`` -> ``(out,in,kh,kw)``
2. input channels  caffe BGR -> RGB
3. fc1 flatten     Keras flattens ``(H,W,C)``, torch flattens ``(C,H,W)``
4. preprocessing   caffe wants ``x*255 - mean_bgr`` (no std division); the repo
   feeds ``(x - mean)/std`` on ``[0,1]`` RGB. Rather than special-casing the
   model's input, the difference is folded into conv1.

**The fold is exact, not an approximation.** Writing the caffe input in terms of
the normalised one, per channel::

    x_caffe = 255*x - caffe_mean
            = 255*(x_tv*std + mean) - caffe_mean
            = x_tv * (255*std) + (255*mean - caffe_mean)
            = a * x_tv + d

and convolution is linear, so ``W ⋆ (a*x_tv + d) = (W*a) ⋆ x_tv + W·d``. The
gain lands in the weights and the offset in the bias, with no residual. That
matters: a *gain* error here would rescale the effective contrast of a grating
and slide the whole contrast-response curve along its own axis, which would
masquerade as a difference between checkpoints. ``fold_preprocessing`` is
therefore a pure function with a dedicated offline test.

Caveat: conv1 pads with zeros, so the constant ``d`` is folded exactly only for
the interior. The 1-pixel border sees ``0`` where the fold assumes ``d`` (|d| <
0.5 against inputs of scale ~128), i.e. 224²->222² is exact and the rim is off
by a part in ~10⁵. ``--verify`` reports the interior.

Usage::

    python -m log_response.convert_weights --out vgg19_caffe.pth --verify
    python -m log_response.run --model vgg19 --weights vgg19_caffe.pth --reps 50

Note on digests: this reproduces the weights behind ``results/vgg19-r50-s0``
tensor-for-tensor (all 38 verified ``torch.equal``), but ``torch.save`` does not
serialise byte-identically across runs, so the file's sha256 will *not* match
the one in that run's ``run.json``. The digest identifies a file, not a
checkpoint -- compare tensors, not hashes, when reproducing.
"""

from __future__ import annotations

import argparse
import os
import urllib.request

import numpy as np

SOURCE_URL = (
    "https://storage.googleapis.com/tensorflow/keras-applications/"
    "vgg19/vgg19_weights_tf_dim_ordering_tf_kernels.h5"
)

# Keras block/conv names in forward order; zips against torchvision's Conv2d list.
KERAS_CONVS = [
    f"block{block}_conv{i}"
    for block, count in [(1, 2), (2, 2), (3, 4), (4, 4), (5, 4)]
    for i in range(1, count + 1)
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float64)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float64)
# Caffe's per-channel mean, written RGB-first to match the swapped kernels.
CAFFE_MEAN_RGB = np.array([123.68, 116.779, 103.939], dtype=np.float64)


def fold_preprocessing(
    weight: np.ndarray,
    bias: np.ndarray,
    mean: np.ndarray = IMAGENET_MEAN,
    std: np.ndarray = IMAGENET_STD,
    caffe_mean: np.ndarray = CAFFE_MEAN_RGB,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold caffe preprocessing into a conv1 kernel already in RGB torch layout.

    ``weight`` is ``(out, in, kh, kw)``. Returns ``(weight, bias)`` such that
    convolving a torchvision-normalised RGB input reproduces what the caffe
    kernel computes from its own preprocessed input. See the module docstring
    for the derivation; ``a`` is the per-channel gain and ``d`` the offset.
    """
    a = 255.0 * np.asarray(std, dtype=np.float64)
    d = 255.0 * np.asarray(mean, dtype=np.float64) - np.asarray(
        caffe_mean, dtype=np.float64
    )
    # bias absorbs W·d summed over input channels and spatial positions
    bias = np.asarray(bias, dtype=np.float64) + (
        weight.sum(axis=(2, 3)) * d[None, :]
    ).sum(axis=1)
    weight = weight * a[None, :, None, None]
    return weight, bias


def convert(h5_path: str) -> dict:
    """Read the Keras HDF5 and return a torchvision-compatible ``state_dict``."""
    import h5py
    import torch
    import torchvision

    with h5py.File(h5_path, "r") as fh:

        def read(layer: str, kind: str) -> np.ndarray:
            group = fh[layer]
            key = next(k for k in group.keys() if f"_{kind}_" in k)
            return np.asarray(group[key], dtype=np.float64)

        net = torchvision.models.vgg19(weights=None)
        conv_idx = [
            i for i, m in enumerate(net.features) if isinstance(m, torch.nn.Conv2d)
        ]
        if len(conv_idx) != len(KERAS_CONVS):
            raise RuntimeError(
                f"expected {len(KERAS_CONVS)} convs, found {len(conv_idx)}"
            )

        state: dict = {}
        for n, (idx, name) in enumerate(zip(conv_idx, KERAS_CONVS)):
            weight = read(name, "W").transpose(3, 2, 0, 1)  # -> (out,in,kh,kw)
            bias = read(name, "b")
            if n == 0:
                weight = weight[:, ::-1, :, :].copy()  # BGR -> RGB input channels
                weight, bias = fold_preprocessing(weight, bias)
            state[f"features.{idx}.weight"] = torch.tensor(weight, dtype=torch.float32)
            state[f"features.{idx}.bias"] = torch.tensor(bias, dtype=torch.float32)

        # fc1: Keras flattens block5_pool as (7,7,512); torch as (512,7,7).
        fc1 = read("fc1", "W").reshape(7, 7, 512, 4096)
        fc1 = fc1.transpose(2, 0, 1, 3).reshape(25088, 4096)
        state["classifier.0.weight"] = torch.tensor(fc1.T, dtype=torch.float32)
        state["classifier.0.bias"] = torch.tensor(read("fc1", "b"), dtype=torch.float32)
        for src, dst in (("fc2", "classifier.3"), ("predictions", "classifier.6")):
            state[f"{dst}.weight"] = torch.tensor(read(src, "W").T, dtype=torch.float32)
            state[f"{dst}.bias"] = torch.tensor(read(src, "b"), dtype=torch.float32)

    net.load_state_dict(state)  # raises on any shape/name mismatch
    return state


def verify(h5_path: str, state: dict) -> float:
    """Check the fold against the caffe path directly. Returns the relative error.

    Runs one random image through (a) the original BGR kernel on caffe-preprocessed
    input and (b) the converted kernel on torchvision-normalised input, and
    compares conv1 over the interior. conv1 is the only layer that touches the
    input, so agreement here rules out an input-gain error anywhere.
    """
    import h5py
    import torch
    import torch.nn.functional as F

    with h5py.File(h5_path, "r") as fh:
        group = fh["block1_conv1"]
        weight = np.asarray(
            group[next(k for k in group if "_W_" in k)], dtype=np.float64
        )
        bias = np.asarray(group[next(k for k in group if "_b_" in k)], dtype=np.float64)

    rng = np.random.default_rng(0)
    image = rng.random((224, 224, 3))  # RGB in [0,1]

    # (a) caffe reference: BGR, [0,255], mean-subtracted, original kernel
    bgr = np.stack([image[..., 2], image[..., 1], image[..., 0]], axis=0) * 255.0
    bgr -= CAFFE_MEAN_RGB[::-1, None, None]
    ref = F.conv2d(
        torch.tensor(bgr)[None],
        torch.tensor(weight.transpose(3, 2, 0, 1)),
        torch.tensor(bias),
        padding=1,
    )

    # (b) converted: torchvision-normalised RGB through the folded kernel
    x = (
        torch.tensor(image).permute(2, 0, 1)
        - torch.tensor(IMAGENET_MEAN)[:, None, None]
    ) / torch.tensor(IMAGENET_STD)[:, None, None]
    got = F.conv2d(
        x[None],
        state["features.0.weight"].double(),
        state["features.0.bias"].double(),
        padding=1,
    )

    interior = (slice(None), slice(None), slice(1, -1), slice(1, -1))
    return float(
        (ref[interior] - got[interior]).abs().max() / ref[interior].abs().max()
    )


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out", required=True, help="path to write the state_dict to")
    p.add_argument(
        "--source",
        default=SOURCE_URL,
        help="Keras VGG-19 .h5 path or URL (default: the Google-hosted mirror)",
    )
    p.add_argument(
        "--cache",
        default="vgg19_keras.h5",
        help="where to keep the downloaded .h5 (reused if already present)",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="check the folded conv1 against the caffe path and fail above 1e-5",
    )
    args = p.parse_args(argv)

    h5_path = args.source
    if not os.path.exists(h5_path):
        h5_path = args.cache
        if not os.path.exists(h5_path):
            print(f"downloading {args.source} -> {h5_path}")
            urllib.request.urlretrieve(args.source, h5_path)

    state = convert(h5_path)
    import torch

    torch.save(state, args.out)
    print(f"wrote {args.out} ({sum(v.numel() for v in state.values())} parameters)")

    if args.verify:
        error = verify(h5_path, state)
        print(f"conv1 vs caffe reference: relative error {error:.3e}")
        if error > 1e-5:
            raise SystemExit(
                f"error: fold is not exact (relative error {error:.3e}); "
                "a gain error here would rescale effective contrast"
            )
        print("fold verified exact (input-gain error ruled out)")


if __name__ == "__main__":
    main()
