# RonInfra

[`log_response/`](log_response/) — a runnable experiment measuring the
log-contrast response of ImageNet-trained vision networks: feed sinusoidal
gratings at log-spaced contrasts, and the mean absolute change in the net's
late-layer representation is close to linear in `log(contrast)`. Measured so
far on VGG-19 (two checkpoints), AlexNet, VGG-19+BN, ResNet-50 and ViT-B/16.

```bash
pip install numpy matplotlib
python -m log_response.run --model synthetic --reps 12 --figures out/   # offline, seconds
```

Docs live in [`wiki/`](wiki/):

| Page | Contents |
|------|----------|
| [Running](wiki/Running.md) | Install, back-ends, flags, and per-environment tips (local / sandbox / GitHub runners). |
| [Results](wiki/Results.md) | Measured numbers. The paper's `prob` R² reproduces at **0.980** on the checkpoint the paper used (Oxford/Caffe); torchvision's `IMAGENET1K_V1` gives a different profile. Across five architectures `prob` λ runs +0.05 to −0.27, and neither depth nor rectifiers turn out to be what produces the compression. |
| [Method](wiki/Method.md)   | The exact procedure: inputs, metric, fit, caveats. |
| [1701.04674](wiki/1701.04674-adaptation-as-readout.pdf) | The source paper (PDF): *Adaptation as Readout*, ICLR 2017 submission. |
