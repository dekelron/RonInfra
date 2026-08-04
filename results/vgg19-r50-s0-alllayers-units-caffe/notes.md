# vgg19-r50-s0-alllayers-units-caffe

`vgg19`, 50 reps/cell, best mean R² 0.976 at `prob`.

## What this run was for

Per-unit surfaces at the head taps. D is an L1 norm, so lambda cannot see a response whose carriers rotate with contrast; this asks whether that is what happens at the fc7 ReLU where lambda drops 1.110 -> 0.231. Also the first read of scale_matched (|z0| against the perturbation scale) at classifier.3, the rectifier input.

## What it showed

Layer surfaces are **byte-identical** to `vgg19-r50-s0-alllayers-caffe` across
all 45 taps (max |Δλ| = 0.000000), so `--unit-taps` changed nothing and this
doubles as a cross-runner reproducibility check.

Per-unit λ fitted at 14 cyc/img; the layer column is the same frequency, not
the 8-frequency median.

| tap | layer λ | median unit | amplitude-weighted mean | unit λ IQR | top 400 | their share of D | `scale_matched` |
|---|---|---|---|---|---|---|---|
| fc6 `classifier.0` | +1.045 | +1.030 | +1.055 | [+0.94, +1.16] | +1.020 | 26% | 0.772 |
| ReLU `.1` | +0.843 | +0.709 | +0.825 | [−0.43, +1.11] | +1.039 | 42% | 0.656 |
| fc7 `.3` | +1.153 | +1.116 | +1.146 | [+0.96, +1.38] | +1.011 | 23% | 0.625 |
| ReLU `.4` | +0.352 | −0.159 | +0.249 | [−1.17, +0.54] | +0.621 | 52% | 0.620 |
| `prob` | +0.116 | −0.035 | −0.023 | [−0.28, +0.23] | +0.110 | 90% | 0.077 |

1. **The norm is faithful here.** Layer λ is within 0.14 of the
   amplitude-weighted mean of the per-unit λ at every tap, within 0.03 at the
   three clean ones. A norm *can* hide a rotating response — that is what
   `--unit-taps` was built for — but at these taps it does not.
2. **The median unit is not the layer.** `classifier.4`'s typical unit reads
   λ −0.159 against the layer's +0.352; a loud minority carries it.
3. **A rectifier fans the exponents out**, it does not translate them: IQR width
   0.22 → 1.54 at `classifier.1`, 0.42 → 1.71 at `classifier.4`.
4. **Carriers rotate, but not where predicted.** Spearman of per-unit shift
   between the lowest and highest contrast: 0.42 / 0.42 / **0.06** / 0.64 /
   0.71. Not a noise artifact — adjacent contrast columns correlate 0.85–0.92
   including the two lowest, and restricting to units in the top quartile at
   both ends still gives 0.19–0.32. `classifier.3` is the most *linear* tap and
   by far the most rotating, which is the opposite of the prediction.
5. **`scale_matched` reads with the wrong sign.** At the two taps that are a
   rectifier's input: fc6 77.2% → its ReLU costs Δλ −0.297; fc7 62.5% → −0.879.
   n = 2, so a flag rather than a result.

Caveats: one seed, `--reps 50`, per-unit λ at one frequency, head taps only.
The conv stack cannot be measured this way — one VGG conv tap would be 1.4 GB.

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-r50-s0-alllayers-units-caffe --notes Per-unit surfaces at the head taps. D is an L1 norm, so lambda cannot see a response whose carriers rotate with contrast; this asks whether that is what happens at the fc7 ReLU where lambda drops 1.110 -> 0.231. Also the first read of scale_matched (|z0| against the perturbation scale) at classifier.3, the rectifier input. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all --unit-taps classifier.0,classifier.1,classifier.3,classifier.4,prob
```

Code: `35e5237476b1`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
