# RonInfra

[`log_response/`](log_response/) — a runnable experiment measuring the
log-contrast response of ImageNet-trained vision networks: feed sinusoidal
gratings at log-spaced contrasts, and the mean absolute change in the net's
late-layer representation is close to linear in `log(contrast)`. Measured on
**28 architectures** (VGG-19 on two checkpoints, VGG-19-BN, AlexNet,
ResNet/ResNeXt, DenseNet, GoogLeNet, SqueezeNet, the
MobileNet/MNASNet/ShuffleNet/EfficientNet group, RegNet, ConvNeXt,
ViT/Swin/MaxViT, and — via `timm` — FocalNet, PoolFormer, XCiT and the
attention-free, convolution-free gMLP and ResMLP), 90 committed runs.

Twenty-seven of those are ImageNet classifiers; the twenty-eighth is a
**generative VLM** (SmolVLM-256M), where the compression also appears — its
hidden taps sit at the log law with a weight-scrambled control cleanly
separated, so the effect does not depend on the classification objective.

```bash
pip install numpy matplotlib
python -m log_response.run --model synthetic --reps 12 --figures out/   # offline, seconds
```

Docs live in [`wiki/`](wiki/):

| Page | Contents |
|------|----------|
| [Running](wiki/Running.md) | Install, back-ends, flags, and per-environment tips (local / sandbox / GitHub runners). |
| [Results](wiki/Results.md) | Measured numbers. The paper's `prob` R² reproduces at **0.980** on the checkpoint the paper used (Oxford/Caffe); torchvision's `IMAGENET1K_V1` gives a different profile. Neither depth nor rectifiers turn out to be what produces the compression, and λ moves *more* across spatial frequency within one net than between nets. That frequency structure needs trained weights — 3.4–20.8× flatter when scrambled, 6/6 matched pairs — but its *shape* does not generalise across architectures. |
| [Method](wiki/Method.md)   | The exact procedure: inputs, metric, fit, caveats. |
| [1701.04674](wiki/1701.04674-adaptation-as-readout.pdf) | The source paper (PDF): *Adaptation as Readout*, ICLR 2017 submission. |
