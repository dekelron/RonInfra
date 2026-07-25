"""DNN feature extraction and the L1 representation-distance metric.

The quantity measured is the *mean absolute change in DNN representation*
between the gray reference and an input image -- an L1 distance per layer.
Here we:

* run images through a real vision model,
* tap a chosen set of layers with forward hooks,
* flatten each layer's activation and take the mean absolute difference from the
  reference activation.

Three model back-ends are provided:

* ``TorchvisionModel`` -- a real ImageNet-trained CNN (vgg19, resnet50, ...).
  Pretrained weights normally come from torchvision's hub. In network-restricted
  environments that host may be blocked; pass ``weights_path`` to load a local
  ``state_dict`` you fetched elsewhere. Without trained weights the log response
  does not emerge (it is a *consequence of training*), so a warning is emitted.
* ``CLIPModel`` -- a CLIP image tower via ``open_clip``. CLIP has no class
  probabilities, so the terminal layers are rebuilt from the contrastive head:
  the image embedding, similarities against a fixed text prompt set, and their
  softmax (a zero-shot classifier standing in for 'prob'). See the class
  docstring for the caveats this introduces.
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

# ImageNet preprocessing constants (torchvision back-end).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# OpenAI-CLIP preprocessing constants (fallback when the open_clip transform
# does not expose its own mean/std).
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

# Default zero-shot prompt set for CLIPModel's 'prob' layer: a fixed, diverse
# spread of categories (animals, scenes, objects, textures). The 'prob'
# measurement is conditional on this set -- pass your own via ``prompts`` /
# ``--prompts`` to probe sensitivity to the choice.
DEFAULT_PROMPTS: tuple[str, ...] = (
    "a photo of a dog",
    "a photo of a cat",
    "a photo of a bird",
    "a photo of a fish",
    "a photo of a horse",
    "a photo of an elephant",
    "a photo of a butterfly",
    "a photo of a spider",
    "a photo of a snake",
    "a photo of a frog",
    "a photo of a tree",
    "a photo of a flower",
    "a photo of a mushroom",
    "a photo of a mountain",
    "a photo of a beach",
    "a photo of a forest",
    "a photo of a desert",
    "a photo of a waterfall",
    "a photo of a city street",
    "a photo of a bridge",
    "a photo of a castle",
    "a photo of a house",
    "a photo of a skyscraper",
    "a photo of a barn",
    "a photo of a car",
    "a photo of a bicycle",
    "a photo of an airplane",
    "a photo of a boat",
    "a photo of a train",
    "a photo of a chair",
    "a photo of a table",
    "a photo of a lamp",
    "a photo of a clock",
    "a photo of a telephone",
    "a photo of a computer",
    "a photo of a camera",
    "a photo of a book",
    "a photo of a guitar",
    "a photo of a piano",
    "a photo of a violin",
    "a photo of a hammer",
    "a photo of a pair of scissors",
    "a photo of an umbrella",
    "a photo of a backpack",
    "a photo of a shoe",
    "a photo of a hat",
    "a photo of a dress",
    "a photo of a pizza",
    "a photo of an apple",
    "a photo of a banana",
    "a photo of a coffee cup",
    "a photo of a wine bottle",
    "a photo of a cake",
    "a photo of a sandwich",
    "a photo of a person",
    "a photo of a human face",
    "a photo of a hand",
    "a photo of a brick wall",
    "a photo of striped fabric",
    "a photo of a checkered pattern",
    "a photo of the ocean surface",
    "a photo of clouds in the sky",
    "a photo of a campfire",
    "a photo of fresh snow",
)


# The fixed conditioning prompt for HFVLMModel: a generative VLM's forward pass
# needs text, so the measurement is conditional on this instruction.
DEFAULT_INSTRUCTION = "Describe this image."


def parse_hf_spec(spec: str) -> str:
    """Parse an ``hf:MODEL_ID`` spec (HF hub id or local path) into the id."""
    if not spec.startswith("hf:") or len(spec) <= 3:
        raise ValueError(f"not an hf VLM model spec: {spec!r} (expected hf:MODEL_ID)")
    return spec[3:]


def parse_sam_spec(spec: str) -> str:
    """Parse a ``sam[:MODEL_ID]`` spec; bare ``sam`` means facebook/sam-vit-base."""
    if spec == "sam":
        return "facebook/sam-vit-base"
    if not spec.startswith("sam:") or len(spec) <= 4:
        raise ValueError(f"not a SAM model spec: {spec!r} (expected sam[:MODEL_ID])")
    return spec[4:]


def _processor_input_size(processor, attrs: tuple[str, ...], keys: tuple[str, ...], default: int) -> int:
    """Read the square input side from a HF image processor's size attributes.

    Handles plain-dict (transformers 4) and SizeDict (transformers 5) values.
    """
    ip = getattr(processor, "image_processor", None) or processor
    for attr in attrs:
        d = getattr(ip, attr, None)
        if d is None:
            continue
        if isinstance(d, int):
            return d
        for key in keys:
            v = d.get(key) if isinstance(d, dict) else getattr(d, key, None)
            if isinstance(v, int):
                return v
    return default


def parse_clip_spec(spec: str) -> tuple[str, str]:
    """Parse a ``clip[:ARCH[:PRETRAINED]]`` model spec into (arch, tag).

    Examples: ``clip`` -> ('ViT-B-32', 'openai');
    ``clip:ViT-L-14`` -> ('ViT-L-14', 'openai');
    ``clip:ViT-B-32:laion2b_s34b_b79k`` -> ('ViT-B-32', 'laion2b_s34b_b79k').
    """
    parts = spec.split(":")
    if parts[0] != "clip" or len(parts) > 3:
        raise ValueError(f"not a CLIP model spec: {spec!r}")
    arch = parts[1] if len(parts) > 1 and parts[1] else "ViT-B-32"
    tag = parts[2] if len(parts) > 2 and parts[2] else "openai"
    return arch, tag


def load_prompts(path: str) -> list[str]:
    """Load a zero-shot prompt set: one prompt per line, blank lines ignored."""
    with open(path, encoding="utf-8") as fh:
        prompts = [line.strip() for line in fh if line.strip()]
    if len(prompts) < 2:
        raise ValueError(f"prompt file {path!r} needs at least two prompts")
    return prompts


def l1_distance(rep: np.ndarray, ref: np.ndarray) -> float:
    """Mean absolute difference between two flattened representations."""
    return float(np.mean(np.abs(rep.reshape(-1) - ref.reshape(-1))))


class FeatureModel:
    """Interface: map an (H, W, 3) [0,1] image to a dict of layer -> activation."""

    layers: list[str]

    def represent(self, image: np.ndarray) -> dict[str, np.ndarray]:
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Shared torch plumbing
# --------------------------------------------------------------------------- #
class _TorchBackend(FeatureModel):
    """Shared machinery for the torch back-ends: forward hooks on named modules,
    [0,1]-image -> normalised-tensor preprocessing, and the within-layer weight
    scrambling control. Subclasses set ``torch``, ``device``, ``net``, ``arch``
    and ``_norm_mean``/``_norm_std`` before calling ``_init_hooks``/``_register``.
    """

    def _init_hooks(self) -> None:
        self._acts: dict[str, np.ndarray] = {}
        self._hooks: list = []

    def _register(self, names: list[str]) -> None:
        modules = dict(self.net.named_modules())
        for name in names:
            if name not in modules:
                raise KeyError(f"layer {name!r} not found in {self.arch}")

            def hook(_m, _in, out, key=name):
                if isinstance(out, (tuple, list)):  # e.g. blocks returning extras
                    out = out[0]
                # .float() so half/bfloat16 nets (numpy has no bfloat16) work.
                #
                # .clone() is load-bearing, not defensive. On CPU with float32
                # activations .float(), .cpu() and .numpy() are all no-ops that
                # return a view onto the module's own output storage -- and
                # torchvision builds VGG/ResNet with nn.ReLU(inplace=True), so
                # the activation gets overwritten by the next layer before the
                # forward returns. Without this, every pre-activation tap
                # silently reports the post-activation values instead: a conv
                # tap and the ReLU tap after it come back bit-identical, with no
                # negative values anywhere in the "pre-ReLU" surface.
                self._acts[key] = out.detach().clone().float().cpu().numpy()

            self._hooks.append(modules[name].register_forward_hook(hook))

    def close(self) -> None:
        """Remove the forward hooks (call if creating many models per process)."""
        for handle in self._hooks:
            handle.remove()
        self._hooks = []

    def _scramble_within_layers(self, seed: int) -> None:
        # Control: permute each weight tensor's own elements (scramble the
        # learned weights within each layer, for every parameter named
        # '*weight*'). On VGG-19 this collapses the prob-layer R^2 from ~0.98:
        # wiki/Method.md documents 0.60, one measured run gives 0.428 (see
        # results/vgg19-scramble-r50-s0/notes.md) -- unreconciled, so no single
        # number is asserted here.
        torch = self.torch
        gen = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            for name, p in self.net.named_parameters():
                if "weight" in name and p.numel() > 1:
                    flat = p.reshape(-1)
                    perm = torch.randperm(flat.numel(), generator=gen)
                    p.copy_(flat[perm].reshape(p.shape))

    def _preprocess(self, image: np.ndarray):
        torch = self.torch
        x = torch.from_numpy(np.ascontiguousarray(image)).float()  # H,W,3
        x = x.permute(2, 0, 1)  # 3,H,W
        mean = torch.tensor(self._norm_mean).view(3, 1, 1)
        std = torch.tensor(self._norm_std).view(3, 1, 1)
        x = (x - mean) / std
        return x.unsqueeze(0).to(self.device)


# --------------------------------------------------------------------------- #
# Real torchvision CNN back-end
# --------------------------------------------------------------------------- #
class TorchvisionModel(_TorchBackend):
    def __init__(
        self,
        arch: str = "vgg19",
        layers: list[str] | None = None,
        weights_path: str | None = None,
        pretrained: bool = True,
        device: str = "cpu",
        scramble: bool = False,
        scramble_seed: int = 0,
        allow_random_init: bool = False,
    ):
        import torch
        import torchvision

        self.torch = torch
        self.device = torch.device(device)
        self.arch = arch
        self._norm_mean, self._norm_std = IMAGENET_MEAN, IMAGENET_STD
        # Whether this model carries trained weights. False means the measured
        # numbers are meaningless (the log response is a consequence of
        # training), so callers must never persist such a run as a result.
        self.weights_ok = False
        self.weights_source = "random init"

        model_fn = getattr(torchvision.models, arch)
        if pretrained and weights_path is None:
            try:
                self.net = model_fn(weights="IMAGENET1K_V1")
                self.weights_ok = True
                self.weights_source = f"torchvision {arch} IMAGENET1K_V1"
            except Exception as exc:  # weight download blocked, etc.
                if not allow_random_init:
                    raise RuntimeError(
                        f"could not load torchvision pretrained weights for "
                        f"{arch!r} ({exc}). The log response only exists in a "
                        "trained net, so falling back to random init would "
                        "produce plausible-looking but meaningless numbers. "
                        "Pass weights_path with a local state_dict, or set "
                        "allow_random_init=True if you genuinely want an "
                        "untrained control."
                    ) from exc
                warnings.warn(
                    f"could not load torchvision pretrained weights ({exc}); "
                    "falling back to random init -- the log law will NOT appear.",
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
                self.weights_ok = True
                self.weights_source = f"local state_dict: {weights_path}"
            elif not allow_random_init:
                raise RuntimeError(
                    f"{arch!r} requested with pretrained=False and no "
                    "weights_path: the net is randomly initialised and the log "
                    "response will not appear. Set allow_random_init=True to "
                    "measure an untrained control deliberately."
                )

        if scramble:
            self._scramble_within_layers(scramble_seed)

        self.net.eval().to(self.device)

        if layers == ["all"]:
            layers = self._all_layers()
        self.layers = layers or self._default_layers(arch)
        self._init_hooks()
        self._register(self.layers)

    def _all_layers(self) -> list[str]:
        """Every leaf module, so the depth profile is sampled continuously.

        Both sides of each nonlinearity are included -- a Conv2d output is the
        pre-activation and the ReLU that follows it the post-activation -- which
        is the point: it locates *where* along the network the response changes
        shape, rather than sampling three points and interpolating.

        Dropout is skipped: it is the identity in eval mode, so it would
        duplicate the preceding layer's surface exactly.
        """
        import torch.nn as nn

        return [
            name
            for name, module in self.net.named_modules()
            if not list(module.children())
            and not isinstance(module, (nn.Dropout, nn.Dropout2d))
        ]

    def _default_layers(self, arch: str) -> list[str]:
        # A spread from early to end computation. 'logits'/'prob' are appended in
        # represent(); for VGG-19 these tap conv1_1 (pre-ReLU), a mid conv, fc7.
        if arch.startswith("vgg"):
            return ["features.0", "features.19", "classifier.3"]
        if arch.startswith("resnet"):
            return ["layer1", "layer2", "layer3", "layer4"]
        # Generic fallback: tap the last *registered* module. Registration order
        # is not execution order, so for unknown archs pass ``layers`` explicitly
        # if this guess taps the wrong point ('logits'/'prob' are always added).
        return [list(dict(self.net.named_modules()).keys())[-1]]

    def represent(self, image: np.ndarray) -> dict[str, np.ndarray]:
        self._acts = {}
        with self.torch.no_grad():
            logits = self.net(self._preprocess(image))
        acts = {k: v.copy() for k, v in self._acts.items()}
        # Expose both the pre-softmax logits (fc8) and the softmax probabilities.
        # The near-linear fit is strongest at 'prob'; comparing logits vs prob
        # isolates how much of the compression is the softmax (see wiki/Method.md).
        acts["logits"] = logits.cpu().numpy()
        acts["prob"] = self.torch.softmax(logits, dim=1).cpu().numpy()
        return acts


# --------------------------------------------------------------------------- #
# CLIP back-end (open_clip)
# --------------------------------------------------------------------------- #
class CLIPModel(_TorchBackend):
    """CLIP image tower via ``open_clip``, with a zero-shot 'prob' layer.

    CLIP has no class probabilities, so the terminal layers are rebuilt from
    the contrastive head:

    * ``embed``     -- the (unnormalised) image embedding from ``encode_image``;
    * ``zs_logits`` -- logit-scaled cosine similarities of the normalised
      embedding against a fixed set of text prompts;
    * ``prob``      -- their softmax: a zero-shot classifier over ``prompts``.

    Unlike the torchvision back-end, where the 1000 ImageNet classes are part
    of the trained model, 'prob' here is conditional on the chosen prompt set
    (default ``DEFAULT_PROMPTS``, N=64) -- report the prompt set with any
    numbers. The wiki/Method.md total-variation bound scales with the set size:
    ``0 <= D_prob <= 2/N``. Text features are computed once at init, so per-
    image cost is the image tower only.

    ``scramble=True`` permutes every weight tensor in BOTH towers and the text
    features are recomputed from the scrambled text encoder, mirroring the
    within-layer scrambling control of ``TorchvisionModel``.

    ``pretrained_tag`` is an open_clip pretrained tag ('openai',
    'laion2b_s34b_b79k', ...) or a path to a local open_clip checkpoint.
    """

    def __init__(
        self,
        arch: str = "ViT-B-32",
        pretrained_tag: str = "openai",
        layers: list[str] | None = None,
        prompts: list[str] | None = None,
        device: str = "cpu",
        pretrained: bool = True,
        scramble: bool = False,
        scramble_seed: int = 0,
        allow_random_init: bool = False,
    ):
        import torch

        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "CLIPModel requires the open_clip_torch package "
                "(pip install open_clip_torch)"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.arch = arch

        # See TorchvisionModel.weights_ok -- a random-init CLIP tower measures
        # nothing, so the failure is raised rather than warned past by default.
        self.weights_ok = False
        self.weights_source = "random init"

        tag = pretrained_tag if pretrained else ""
        try:
            self.net, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=tag
            )
            self.weights_ok = bool(tag)
            self.weights_source = f"open_clip {arch}:{tag}" if tag else "random init"
        except Exception as exc:
            if not tag:
                raise
            if not allow_random_init:
                raise RuntimeError(
                    f"could not load CLIP pretrained weights {tag!r} ({exc}). "
                    "The log response only exists in a trained net, so falling "
                    "back to random init would produce meaningless numbers. "
                    "Pass a local open_clip checkpoint path as the pretrained "
                    "tag, or set allow_random_init=True for an untrained control."
                ) from exc
            warnings.warn(
                f"could not load CLIP pretrained weights {tag!r} ({exc}); "
                "falling back to random init -- the log law will NOT appear.",
                RuntimeWarning,
            )
            self.net, _, preprocess = open_clip.create_model_and_transforms(
                arch, pretrained=""
            )

        # Use the checkpoint's own normalisation if the transform exposes it.
        self._norm_mean, self._norm_std = CLIP_MEAN, CLIP_STD
        for t in getattr(preprocess, "transforms", []):
            if hasattr(t, "mean") and hasattr(t, "std"):
                self._norm_mean = tuple(float(v) for v in t.mean)
                self._norm_std = tuple(float(v) for v in t.std)

        if scramble:
            self._scramble_within_layers(scramble_seed)

        self.net.eval().to(self.device)

        size = getattr(self.net.visual, "image_size", 224)
        if isinstance(size, (tuple, list)):
            size = size[0]
        self.input_size = int(size)  # gratings must be generated at this size

        self.prompts = list(prompts) if prompts else list(DEFAULT_PROMPTS)
        if len(self.prompts) < 2:
            raise ValueError("need at least two prompts for the zero-shot 'prob' layer")
        tokenizer = open_clip.get_tokenizer(arch)
        with torch.no_grad():
            text = self.net.encode_text(tokenizer(self.prompts).to(self.device))
        self._text_features = text / text.norm(dim=-1, keepdim=True)

        self.layers = layers or self._default_layers()
        self._init_hooks()
        self._register(self.layers)

    def _default_layers(self) -> list[str]:
        modules = dict(self.net.named_modules())
        # ViT towers: first / middle / last transformer block (openai-style and
        # timm-style module naming).
        for prefix in ("visual.transformer.resblocks.", "visual.trunk.blocks."):
            blocks = sorted(
                (n for n in modules if n.startswith(prefix) and n[len(prefix):].isdigit()),
                key=lambda n: int(n[len(prefix):]),
            )
            if blocks:
                return [blocks[0], blocks[len(blocks) // 2], blocks[-1]]
        # ModifiedResNet towers.
        resnet = [f"visual.layer{i}" for i in (1, 2, 3, 4) if f"visual.layer{i}" in modules]
        if resnet:
            return resnet
        return ["visual"]

    def represent(self, image: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        self._acts = {}
        with torch.no_grad():
            embed = self.net.encode_image(self._preprocess(image))
            norm = embed / embed.norm(dim=-1, keepdim=True)
            zs = self.net.logit_scale.exp() * norm @ self._text_features.T
            prob = torch.softmax(zs, dim=-1)
        acts = {k: v.copy() for k, v in self._acts.items()}
        acts["embed"] = embed.cpu().numpy()
        acts["zs_logits"] = zs.cpu().numpy()
        acts["prob"] = prob.cpu().numpy()
        return acts


# --------------------------------------------------------------------------- #
# Generative-VLM back-end (HF transformers)
# --------------------------------------------------------------------------- #
class HFVLMModel(_TorchBackend):
    """Generative VLM (LLaVA, Qwen-VL, ...) via a HF image-text-to-text model.

    A generative VLM's forward pass needs a text prompt, so the measurement is
    **conditional on a fixed instruction** (default ``DEFAULT_INSTRUCTION``),
    rendered through the model's chat template when it has one (image first,
    then the instruction, with the generation prompt appended). Terminal
    layers:

    * ``logits`` -- next-token logits at the final sequence position, i.e. the
      distribution over the *first response token* given grating + instruction;
    * ``prob``   -- their softmax over the LLM vocabulary: the representation
      of "what the model would say next".

    Intermediate taps default to: the last vision-tower block, the multimodal
    projector, and a middle + the last LLM decoder layer (introspected from
    module names; pass ``layers`` explicitly when the guess is wrong). Decoder
    activations cover the whole sequence -- image-token and text-token
    positions together; the text is fixed, so shapes are constant and the
    distance-of-means metric applies unchanged.

    Caveats vs the CNN measurement: 'prob' depends on the instruction and chat
    template (report both alongside any numbers), and the wiki/Method.md TV bound
    becomes ``0 <= D_prob <= 2/V`` for vocab size V. Cost: the full grid is
    ~28k forward passes -- for a 7B model use a GPU (``device='cuda'``,
    ``dtype='float16'``/``'bfloat16'``) and reduce repetitions/frequencies.

    Images are handed to the model's own ``AutoProcessor`` as float arrays in
    [0, 255], so its resize/normalise pipeline runs exactly as in deployment
    but without an 8-bit quantisation step (important at the lowest contrasts).

    ``model_id`` is a HF hub id or a local directory (pass a local path when
    the hub is unreachable). ``model``/``processor`` can be injected directly,
    which is how the offline tiny-model test exercises this class.
    """

    def __init__(
        self,
        model_id: str | None = None,
        layers: list[str] | None = None,
        instruction: str | None = None,
        device: str = "cpu",
        dtype: str | None = None,
        scramble: bool = False,
        scramble_seed: int = 0,
        trust_remote_code: bool = False,
        model=None,
        processor=None,
    ):
        import torch

        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "HFVLMModel requires the transformers package "
                "(pip install transformers pillow)"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        # from_pretrained raises on a failed fetch, so a model that loads here is
        # genuinely pretrained; an injected model (tests) is not.
        self.weights_ok = model is None
        self.weights_source = f"transformers {model_id}" if model is None else "injected"

        if model is None or processor is None:
            if not model_id:
                raise ValueError(
                    "model_id is required unless model and processor are injected"
                )
            if processor is None:
                processor = transformers.AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=trust_remote_code
                )
            if model is None:
                torch_dtype = {
                    None: None,
                    "auto": "auto",
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }[dtype]
                kwargs: dict = {"trust_remote_code": trust_remote_code}
                if torch_dtype is not None:
                    kwargs["dtype"] = torch_dtype
                model = transformers.AutoModelForImageTextToText.from_pretrained(
                    model_id, **kwargs
                )
        self.net = model
        self.processor = processor
        self.arch = model_id or getattr(self.net.config, "model_type", "vlm")

        if scramble:
            self._scramble_within_layers(scramble_seed)
        self.net.eval().to(self.device)

        self.instruction = instruction if instruction is not None else DEFAULT_INSTRUCTION
        self._text = self._build_text()
        self.input_size = self._infer_input_size()

        self.layers = layers or self._default_layers()
        self._init_hooks()
        self._register(self.layers)

    def _build_text(self) -> str:
        """The full conditioning text: chat template if available, else raw."""
        proc = self.processor
        tok = getattr(proc, "tokenizer", None)
        if getattr(proc, "chat_template", None) or getattr(tok, "chat_template", None):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.instruction},
                    ],
                }
            ]
            return proc.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        image_token = str(getattr(proc, "image_token", None) or "<image>")
        return f"{image_token}\n{self.instruction}"

    def _infer_input_size(self) -> int:
        return _processor_input_size(
            self.processor,
            attrs=("crop_size", "size"),
            keys=("height", "shortest_edge", "width"),
            default=336,
        )

    def _default_layers(self) -> list[str]:
        import re

        names = list(dict(self.net.named_modules()))
        # Group repeated blocks: module names ending in '.<index>'.
        groups: dict[str, list[int]] = {}
        for n in names:
            m = re.fullmatch(r"(.+)\.(\d+)", n)
            if m:
                groups.setdefault(m.group(1), []).append(int(m.group(2)))

        picks: list[str] = []
        vision = {p: idx for p, idx in groups.items() if re.search(r"vis", p)}
        if vision:
            prefix, idx = max(vision.items(), key=lambda kv: len(kv[1]))
            picks.append(f"{prefix}.{max(idx)}")  # last vision block

        projectors = [
            n for n in names
            if "projector" in n.rsplit(".", 1)[-1] or n.rsplit(".", 1)[-1] == "merger"
        ]
        if projectors:
            picks.append(min(projectors, key=len))

        llm = {
            p: idx for p, idx in groups.items()
            if p not in vision and (p.endswith("layers") or p.endswith(".h"))
        }
        if not llm:
            llm = {p: idx for p, idx in groups.items() if p not in vision}
        if llm:
            prefix, idx = max(llm.items(), key=lambda kv: len(kv[1]))
            hi = max(idx)
            picks.extend([f"{prefix}.{hi // 2}", f"{prefix}.{hi}"])

        deduped = [p for i, p in enumerate(picks) if p not in picks[:i]]
        return deduped or [names[-1]]

    def _inputs(self, image: np.ndarray) -> dict:
        # Float values in [0, 255]: the processor's usual 1/255 rescale restores
        # the exact [0,1] grating, so no 8-bit quantisation is introduced.
        arr = np.ascontiguousarray(image * 255.0, dtype=np.float32)
        batch = self.processor(text=self._text, images=arr, return_tensors="pt")
        dtype = next(self.net.parameters()).dtype
        out = {}
        for key, value in batch.items():
            if hasattr(value, "to"):
                if value.is_floating_point():
                    value = value.to(self.device, dtype=dtype)
                else:
                    value = value.to(self.device)
            out[key] = value
        return out

    def represent(self, image: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        self._acts = {}
        inputs = self._inputs(image)
        with torch.no_grad():
            try:
                out = self.net(**inputs, use_cache=False)
            except TypeError:  # model without a use_cache kwarg
                out = self.net(**inputs)
        acts = {k: v.copy() for k, v in self._acts.items()}
        logits = out.logits[:, -1, :].detach().float().cpu()
        acts["logits"] = logits.numpy()
        acts["prob"] = torch.softmax(logits, dim=-1).numpy()
        return acts


# --------------------------------------------------------------------------- #
# Segment-Anything back-end (HF transformers, encoder-only by default)
# --------------------------------------------------------------------------- #
class SAMModel(_TorchBackend):
    """Segment Anything (SAM) image encoder, with an optional mask-decoder tap.

    SAM has no classifier and no contrastive head, so there is **no 'prob'
    analogue**: the measurement is encoder-only by default -- transformer
    blocks of the vision encoder plus ``embed``, the final image embedding
    from ``get_image_embeddings`` (a (256, 64, 64) map for SAM-1 at 1024 px).
    The question it answers: does log-contrast compression emerge in a
    representation trained for *segmentation* (vs classification/contrastive/
    generative training in the other back-ends)?

    With ``mask_decoder=True`` the decoder is additionally run with a **fixed
    center-point prompt**, adding two terminal layers: ``mask_logits`` (the
    low-res multimask logits) and ``iou_scores`` (the predicted mask
    qualities). Like the VLM instruction, this output is conditional on the
    chosen prompt -- report that with any numbers.

    Cost: SAM's native input is 1024x1024, so per-image forwards are heavy;
    shrink ``--reps``/``--frequencies`` and use a GPU for the huge variant.
    ``model_id`` is a HF hub id (facebook/sam-vit-base/large/huge) or a local
    directory. ``model``/``processor`` can be injected (used by the offline
    tiny-model test).
    """

    def __init__(
        self,
        model_id: str = "facebook/sam-vit-base",
        layers: list[str] | None = None,
        device: str = "cpu",
        dtype: str | None = None,
        mask_decoder: bool = False,
        scramble: bool = False,
        scramble_seed: int = 0,
        model=None,
        processor=None,
    ):
        import torch

        try:
            import transformers
        except ImportError as exc:
            raise ImportError(
                "SAMModel requires the transformers package "
                "(pip install transformers pillow)"
            ) from exc

        self.torch = torch
        self.device = torch.device(device)
        self.arch = model_id
        self.mask_decoder = mask_decoder
        # As HFVLMModel: from_pretrained raises on failure, injected models are
        # random-config stand-ins used by the offline tests.
        self.weights_ok = model is None
        self.weights_source = f"transformers {model_id}" if model is None else "injected"

        if processor is None:
            processor = transformers.AutoProcessor.from_pretrained(model_id)
        if model is None:
            torch_dtype = {
                None: None,
                "auto": "auto",
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[dtype]
            model_cls = getattr(
                transformers, "AutoModelForMaskGeneration", None
            ) or transformers.SamModel
            kwargs: dict = {}
            if torch_dtype is not None:
                kwargs["dtype"] = torch_dtype
            model = model_cls.from_pretrained(model_id, **kwargs)
        if not hasattr(model, "get_image_embeddings"):
            raise TypeError(
                f"{type(model).__name__} has no get_image_embeddings; "
                "SAMModel supports the SAM-1 family (facebook/sam-vit-*)"
            )
        self.net = model
        self.processor = processor

        if scramble:
            self._scramble_within_layers(scramble_seed)
        self.net.eval().to(self.device)

        self.input_size = _processor_input_size(
            self.processor,
            attrs=("pad_size", "size"),
            keys=("height", "longest_edge", "shortest_edge", "width"),
            default=1024,
        )

        self.layers = layers or self._default_layers()
        self._init_hooks()
        self._register(self.layers)

    def _default_layers(self) -> list[str]:
        import re

        groups: dict[str, list[int]] = {}
        for n in dict(self.net.named_modules()):
            m = re.fullmatch(r"(.+)\.(\d+)", n)
            if m and re.search(r"vis", m.group(1)):
                groups.setdefault(m.group(1), []).append(int(m.group(2)))
        if not groups:
            return ["vision_encoder"]
        prefix, idx = max(groups.items(), key=lambda kv: len(kv[1]))
        picks = [f"{prefix}.{min(idx)}", f"{prefix}.{max(idx) // 2}", f"{prefix}.{max(idx)}"]
        return [p for i, p in enumerate(picks) if p not in picks[:i]]

    def represent(self, image: np.ndarray) -> dict[str, np.ndarray]:
        torch = self.torch
        self._acts = {}
        # Float values in [0, 255]: the processor's 1/255 rescale restores the
        # exact [0,1] grating (no 8-bit quantisation).
        arr = np.ascontiguousarray(image * 255.0, dtype=np.float32)
        kwargs = {}
        if self.mask_decoder:
            center = image.shape[0] / 2.0
            kwargs["input_points"] = [[[center, center]]]  # fixed center-point prompt
        inputs = self.processor(images=arr, return_tensors="pt", **kwargs)
        dtype = next(self.net.parameters()).dtype
        pixel_values = inputs["pixel_values"].to(self.device, dtype=dtype)
        with torch.no_grad():
            embed = self.net.get_image_embeddings(pixel_values)
            out = None
            if self.mask_decoder:
                points = inputs["input_points"].to(self.device, dtype=dtype)
                out = self.net(
                    input_points=points, image_embeddings=embed, multimask_output=True
                )
        acts = {k: v.copy() for k, v in self._acts.items()}
        acts["embed"] = embed.detach().float().cpu().numpy()
        if out is not None:
            acts["mask_logits"] = out.pred_masks.detach().float().cpu().numpy()
            acts["iou_scores"] = out.iou_scores.detach().float().cpu().numpy()
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
        # Weight-free by construction: "are the weights trained?" does not apply,
        # which is distinct from "the weights are untrained" (None vs False).
        self.weights_ok = None
        self.weights_source = "synthetic (weight-free)"
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
        gray = image[..., 0]  # grayscale images: any channel is fine
        energy = self._band_gain * self._bandpass_energy(gray)  # weighted, ~contrast^2
        # Compressive nonlinearity => log-like response to contrast.
        output = np.log1p(self.gain * energy)
        return {"energy": energy, "output": output}
