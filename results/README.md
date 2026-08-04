# Results

One directory per run, committed. Each holds `result.npz` (the canonical
`D(freq, contrast)` surfaces), `result.json` (fit summary, diffs readably),
`run.json` (provenance) and `notes.md` (what it was for, what it showed).

> **Runs from 2026-07-27 carry two surfaces.** `surfaces` is the paper's
> distance-of-means `D` and stays the headline; `mean_of_distances` is the other
> order of operations, `mean_r mean_i |a_i(x_r) − gray_i|`. It rides along
> because `D` has population value **zero** at any layer affine in the input
> while `D_mod` does not — so where a layer's two λ disagree, the primary metric
> is reporting its own sampling noise. `result.json` carries it under
> `mean_of_distances`. Earlier runs simply lack it and load unchanged; adding it
> left every committed surface bit-identical.

> **Runs from 2026-08-02 also carry a gate-flip count.** `gate_flips` is
> `G(c,f)`, the fraction of units whose sign the grating flips, and `gate_open`
> the fraction positive at gray — the operating point itself. Not a third
> metric: it was built as the direct test of the perturbation reading of the λ
> profile (`wiki/Method.md`), which held that λ = 1 means no rectifier switches.
> **The first three runs carrying it falsified that** — VGG-19's λ ≈ 1 conv
> stack flips a median 35% of its units — so read `G` as evidence *against*
> gate-freezing, not as support for it. `result.json` carries both under
> `gates`. Earlier runs lack them and load unchanged; no committed surface
> moved.

> **Runs from 2026-07-28 are the first that are not VGG-19.** `alexnet`,
> `vgg19_bn`, `resnet50` and `vit_b_16` — all `--layers all`, each with its
> scrambled control and its reps companion. `prob` λ across the six
> architecture/checkpoint combinations now measured:
>
> | +0.05 | +0.06 | +0.17 | −0.16 | −0.22 | −0.27 |
> |---|---|---|---|---|---|
> | AlexNet | VGG-19 Caffe | VGG-19 IN1K | ViT-B/16 | ResNet-50 | `vgg19_bn` |
>
> **Two of the four scrambled controls are invalid.** `vgg19_bn` and `resnet50`
> carry BatchNorm; `--scramble` permutes γ while the running statistics stay
> put, which decalibrates rather than degrades. Their numbers are recorded and
> explicitly not comparable — see either notes file.

| Run | Model | Reps | Weights | Headline | Notes |
|---|---|---|---|---|---|
| [`vgg19-r50-s0-alllayers-caffe-gates`](vgg19-r50-s0-alllayers-caffe-gates/notes.md) | VGG-19, trained | 50 | converted Caffe | **gate-freezing falsified** | First run carrying `G`. λ ≈ 1 across 33/37 conv taps with **median 35% of units flipping sign**; 36 taps near-linear-but-flipping, 0 near-linear-and-quiet. |
| [`vgg19-r50-s0-alllayers-in1k-gates`](vgg19-r50-s0-alllayers-in1k-gates/notes.md) | VGG-19, trained | 50 | `IMAGENET1K_V1` | `G` does not track λ | Matched to the Caffe run: λ +1.055 → +0.759 while `G` 31.3% → 28.1%, r(Δλ, ΔG) = −0.209. |
| [`vgg19-bn-r50-s0-alllayers-gates`](vgg19-bn-r50-s0-alllayers-gates/notes.md) | VGG-19+BN, trained | 50 | `IMAGENET1K_V1` | r(log `G`, λ) = **−0.003** | Same rectifiers, same flip rates, λ +0.150 against Caffe's +1.055. Separates operating point from switch count. |
| [`resnet50-r250-s0`](resnet50-r250-s0/notes.md) | ResNet-50, trained | 250 | `IMAGENET1K_V1` | peak **0.957 at `layer2.3.relu@2`** | The peak tap is a **reuse slot** — an activation the old hook discarded. `prob` λ −0.223. Residual-stream prediction **falsified**: 0% of deep taps near λ=1. |
| [`resnet50-r50-s0`](resnet50-r50-s0/notes.md) | ResNet-50, trained | 50 | `IMAGENET1K_V1` | 3/160 on the floor | `conv1` + `bn1` (the affine prefix) at 98.5/98.7% noise; ≤5.1% everywhere else. |
| [`resnet50-scramble-r250-s0`](resnet50-scramble-r250-s0/notes.md) | ResNet-50, scrambled | 250 | `IMAGENET1K_V1` | **not comparable** | Same BatchNorm decalibration as `vgg19_bn`: r(logits,prob) 0.673, ratio 1.7e-10, λ-R² 0.692. |
| [`resnet50-scramble-r50-s0`](resnet50-scramble-r50-s0/notes.md) | ResNet-50, scrambled | 50 | `IMAGENET1K_V1` | CI = whole search range | λ moves to +0.028 with interval [−3.00, +4.00]. Confirms the above measures nothing. |
| [`vit-b-16-r250-s0`](vit-b-16-r250-s0/notes.md) | ViT-B/16, trained | 250 | `IMAGENET1K_V1` | `prob` λ **−0.162** | **No ReLU anywhere** — GELU, LayerNorm — and λ still runs +0.93 → −0.62 mid-encoder. The gate-flip reading cannot be the mechanism. |
| [`vit-b-16-r50-s0`](vit-b-16-r50-s0/notes.md) | ViT-B/16, trained | 50 | `IMAGENET1K_V1` | 1/65 on the floor | Only `conv_proj`. The floor does **not** cross the first LayerNorm — LN is not affine in the input, BN in eval is. |
| [`vit-b-16-scramble-r250-s0`](vit-b-16-scramble-r250-s0/notes.md) | ViT-B/16, scrambled | 250 | `IMAGENET1K_V1` | `prob` λ **+0.711**, R² 0.981 | A **clean** control: r(logits,prob) 0.999975, ratio 1/1000. LayerNorm has no running statistics, so the scramble behaves. |
| [`vit-b-16-scramble-r50-s0`](vit-b-16-scramble-r50-s0/notes.md) | ViT-B/16, scrambled | 50 | `IMAGENET1K_V1` | λ +0.714 | Reproducible to three decimals — the contrast with the two BN controls. |
| [`alexnet-r250-s0`](alexnet-r250-s0/notes.md) | AlexNet, trained | 250 | `IMAGENET1K_V1` | `prob` **0.963**, λ **+0.053** | **The log law does not need depth.** 8 weight layers, and `prob` is the peak of 21 taps — the paper's structure, which among VGG runs only Caffe gave. |
| [`alexnet-r50-s0`](alexnet-r50-s0/notes.md) | AlexNet, trained | 50 | `IMAGENET1K_V1` | 1/21 on the floor | Only `features.0` (2.240 ≈ √5); ≤1.8% noise everywhere else. Cleaner than VGG-19. |
| [`alexnet-scramble-r250-s0`](alexnet-scramble-r250-s0/notes.md) | AlexNet, scrambled | 250 | `IMAGENET1K_V1` | `prob` 0.865, λ +0.015 | Control. λ is *closer to log* than trained — only R² (0.889 vs 0.985) separates them. Peak moves to `features.9`. |
| [`alexnet-scramble-r50-s0`](alexnet-scramble-r50-s0/notes.md) | AlexNet, scrambled | 50 | `IMAGENET1K_V1` | `prob` 0.857 | Control companion. Rep-invariant, so the 0.098 gap is a property of the weights. |
| [`vgg19-bn-r250-s0`](vgg19-bn-r250-s0/notes.md) | VGG-19+BN, trained | 250 | `IMAGENET1K_V1` | conv median λ **−0.071** | **BatchNorm moves the whole conv stack to the log law without adding a rectification.** Against Caffe's +1.06 and IN1K's +0.69. `prob` λ −0.268 — past log, saturating. |
| [`vgg19-bn-r50-s0`](vgg19-bn-r50-s0/notes.md) | VGG-19+BN, trained | 50 | `IMAGENET1K_V1` | 5/61 on the floor | `features.1` is a **BatchNorm layer reading pure noise** (99.3%) — the floor is a property of affineness, not of being first. Headline taps ≤0.8%. |
| [`vgg19-bn-scramble-r250-s0`](vgg19-bn-scramble-r250-s0/notes.md) | VGG-19+BN, scrambled | 250 | `IMAGENET1K_V1` | **not comparable** | The scramble decalibrates BN rather than degrading it: r(logits,prob) **0.162**, ratio 1e-10, softmax saturated. λ uninformative (R² 0.613). Do not table it against the others. |
| [`vgg19-bn-scramble-r50-s0`](vgg19-bn-scramble-r50-s0/notes.md) | VGG-19+BN, scrambled | 50 | `IMAGENET1K_V1` | CI spans the search range | Confirms the above: λ moves 1.5 across the rep change and the interval opens to [−3.00, +1.32]. |
| [`alexnet-r50-s1`](alexnet-r50-s1/notes.md) | AlexNet, trained | 50 | `IMAGENET1K_V1` | band contrast **+0.38** | Seed 1 of the per-frequency sweep. Mid band dips 0.38 vs both ends; `prob` λ +0.029 at R² 0.983. Series +0.409 ± 0.021, same sign in all 4. |
| [`alexnet-r50-s2`](alexnet-r50-s2/notes.md) | AlexNet, trained | 50 | `IMAGENET1K_V1` | band contrast **+0.42** | Seed 2 of the per-frequency sweep. Mid band dips 0.42 vs both ends; `prob` λ +0.050 at R² 0.984. Series +0.409 ± 0.021, same sign in all 4. |
| [`vgg19-bn-r50-s1`](vgg19-bn-r50-s1/notes.md) | VGG-19+BN, trained | 50 | `IMAGENET1K_V1` | band contrast **+0.58** | Seed 1 of the per-frequency sweep. Mid band dips 0.58 vs both ends; `prob` λ -0.268 at R² 0.950. Series +0.533 ± 0.036, same sign in all 4. |
| [`vgg19-bn-r50-s2`](vgg19-bn-r50-s2/notes.md) | VGG-19+BN, trained | 50 | `IMAGENET1K_V1` | band contrast **+0.52** | Seed 2 of the per-frequency sweep. Mid band dips 0.52 vs both ends; `prob` λ -0.315 at R² 0.950. Series +0.533 ± 0.036, same sign in all 4. |
| [`resnet50-r50-s1`](resnet50-r50-s1/notes.md) | ResNet-50, trained | 50 | `IMAGENET1K_V1` | band contrast **+0.39** | Seed 1 of the per-frequency sweep. Mid band dips 0.39 vs both ends; `prob` λ -0.230 at R² 0.969. Series +0.404 ± 0.014, same sign in all 4. |
| [`resnet50-r50-s2`](resnet50-r50-s2/notes.md) | ResNet-50, trained | 50 | `IMAGENET1K_V1` | band contrast **+0.39** | Seed 2 of the per-frequency sweep. Mid band dips 0.39 vs both ends; `prob` λ -0.215 at R² 0.964. Series +0.404 ± 0.014, same sign in all 4. |
| [`vit-b-16-r50-s1`](vit-b-16-r50-s1/notes.md) | ViT-B/16, trained | 50 | `IMAGENET1K_V1` | band contrast **-0.28** | Seed 1 of the per-frequency sweep. Mid band peaks 0.28 vs both ends; `prob` λ -0.159 at R² 0.928. Series −0.284 ± 0.022, same sign in all 4. |
| [`vit-b-16-r50-s2`](vit-b-16-r50-s2/notes.md) | ViT-B/16, trained | 50 | `IMAGENET1K_V1` | band contrast **-0.31** | Seed 2 of the per-frequency sweep. Mid band peaks 0.31 vs both ends; `prob` λ -0.169 at R² 0.930. Series −0.284 ± 0.022, same sign in all 4. |
| [`vgg19-r50-s1-alllayers-in1k`](vgg19-r50-s1-alllayers-in1k/notes.md) | VGG-19, trained | 50 | `IMAGENET1K_V1` | band contrast **+0.21** | Seed 1 of the per-frequency sweep. Mid band dips 0.21 vs both ends; `prob` λ +0.147 at R² 0.953. Series +0.198 ± 0.022, same sign in all 4. |
| [`vgg19-r50-s2-alllayers-in1k`](vgg19-r50-s2-alllayers-in1k/notes.md) | VGG-19, trained | 50 | `IMAGENET1K_V1` | band contrast **+0.20** | Seed 2 of the per-frequency sweep. Mid band dips 0.20 vs both ends; `prob` λ +0.167 at R² 0.945. Series +0.198 ± 0.022, same sign in all 4. |
| [`vgg19-r50-s1-alllayers-caffe`](vgg19-r50-s1-alllayers-caffe/notes.md) | VGG-19, trained | 50 | converted Caffe | band contrast **-0.30** | Seed 1 of the per-frequency sweep. Mid band peaks 0.30 vs both ends; `prob` λ +0.079 at R² 0.990. Series −0.335 ± 0.025, same sign in all 4. |
| [`vgg19-r50-s2-alllayers-caffe`](vgg19-r50-s2-alllayers-caffe/notes.md) | VGG-19, trained | 50 | converted Caffe | band contrast **-0.35** | Seed 2 of the per-frequency sweep. Mid band peaks 0.35 vs both ends; `prob` λ +0.069 at R² 0.990. Series −0.335 ± 0.025, same sign in all 4. |
| [`convnext-small-r50-s0`](convnext-small-r50-s0/notes.md) | convnext-small, trained | 50 | `IMAGENET1K_V1` | **monotone** | Architecture screen. **monotone** in frequency (ρ -0.90); λ range 1.40, `prob` λ +0.025 at R² 0.926. |
| [`convnext-small-scramble-r50-s0`](convnext-small-scramble-r50-s0/notes.md) | convnext-small, scrambled | 50 | `IMAGENET1K_V1` | λ range **0.07** | **Valid control** (r(logits,prob) 0.9998). λ range across frequency **0.07** against trained 1.40 — **20.8×** flatter. The frequency structure needs trained weights. |
| [`convnext-tiny-r50-s0`](convnext-tiny-r50-s0/notes.md) | convnext-tiny, trained | 50 | `IMAGENET1K_V1` | **monotone** | Architecture screen. **monotone** in frequency (ρ -0.76); λ range 2.54, `prob` λ -0.481 at R² 0.903. |
| [`convnext-tiny-scramble-r50-s0`](convnext-tiny-scramble-r50-s0/notes.md) | convnext-tiny, scrambled | 50 | `IMAGENET1K_V1` | λ range **0.19** | **Valid control** (r(logits,prob) 0.9998). λ range across frequency **0.19** against trained 2.54 — **13.1×** flatter. The frequency structure needs trained weights. |
| [`densenet121-r50-s0`](densenet121-r50-s0/notes.md) | densenet121, trained | 50 | `IMAGENET1K_V1` | band **-0.17** | Architecture screen. mid band peaks 0.17 (ρ -0.69); λ range 0.80, `prob` λ -0.125 at R² 0.978. |
| [`efficientnet-b0-r50-s0`](efficientnet-b0-r50-s0/notes.md) | efficientnet-b0, trained | 50 | `IMAGENET1K_V1` | band **+0.26** | Architecture screen. mid band dips 0.26 (ρ -0.05); λ range 1.87, `prob` λ -0.096 at R² 0.932. |
| [`googlenet-r50-s0`](googlenet-r50-s0/notes.md) | googlenet, trained | 50 | `IMAGENET1K_V1` | **monotone** | Architecture screen. **monotone** in frequency (ρ -0.76); λ range 0.55, `prob` λ -0.418 at R² 0.973. |
| [`maxvit-t-r50-s0`](maxvit-t-r50-s0/notes.md) | maxvit-t, trained | 50 | `IMAGENET1K_V1` | band **+0.32** | Architecture screen. mid band dips 0.32 (ρ -0.60); λ range 1.32, `prob` λ -0.273 at R² 0.968. |
| [`mnasnet1-0-r50-s0`](mnasnet1-0-r50-s0/notes.md) | mnasnet1-0, trained | 50 | `IMAGENET1K_V1` | band **+0.03** | Architecture screen. mid band dips 0.03 (ρ -0.17); λ range 0.55, `prob` λ +0.019 at R² 0.945. |
| [`mobilenet-v2-r50-s0`](mobilenet-v2-r50-s0/notes.md) | mobilenet-v2, trained | 50 | `IMAGENET1K_V1` | band **+0.45** | Architecture screen. mid band dips 0.45 (ρ -0.02); λ range 0.79, `prob` λ +0.060 at R² 0.963. |
| [`mobilenet-v3-large-r50-s0`](mobilenet-v3-large-r50-s0/notes.md) | mobilenet-v3-large, trained | 50 | `IMAGENET1K_V1` | band **-1.53** | Architecture screen. mid band peaks 1.53 (ρ -0.68); λ range 2.17, `prob` λ -0.818 at R² 0.881. λ **pinned at the −3 bound** at one frequency; excluded from the bands. |
| [`regnet-x-400mf-r50-s0`](regnet-x-400mf-r50-s0/notes.md) | regnet-x-400mf, trained | 50 | `IMAGENET1K_V1` | band **+0.62** | Architecture screen. mid band dips 0.62 (ρ +0.24); λ range 1.44, `prob` λ +0.192 at R² 0.968. |
| [`regnet-y-400mf-r50-s0`](regnet-y-400mf-r50-s0/notes.md) | regnet-y-400mf, trained | 50 | `IMAGENET1K_V1` | band **+0.05** | Architecture screen. mid band dips 0.05 (ρ +0.18); λ range 0.76, `prob` λ +0.148 at R² 0.948. λ **pinned at the −3 bound** at one frequency; excluded from the bands. |
| [`resnext50-32x4d-r50-s0`](resnext50-32x4d-r50-s0/notes.md) | resnext50-32x4d, trained | 50 | `IMAGENET1K_V1` | band **-0.36** | Architecture screen. mid band peaks 0.36 (ρ -0.57); λ range 0.67, `prob` λ -0.271 at R² 0.958. |
| [`shufflenet-v2-x1-0-r50-s0`](shufflenet-v2-x1-0-r50-s0/notes.md) | shufflenet-v2-x1-0, trained | 50 | `IMAGENET1K_V1` | band **+0.50** | Architecture screen. mid band dips 0.50 (ρ +0.17); λ range 0.87, `prob` λ +0.010 at R² 0.977. |
| [`squeezenet1-1-r50-s0`](squeezenet1-1-r50-s0/notes.md) | squeezenet1-1, trained | 50 | `IMAGENET1K_V1` | band **+0.31** | Architecture screen. mid band dips 0.31 (ρ -0.33); λ range 0.89, `prob` λ -0.003 at R² 0.975. |
| [`squeezenet1-1-scramble-r50-s0`](squeezenet1-1-scramble-r50-s0/notes.md) | squeezenet1-1, scrambled | 50 | `IMAGENET1K_V1` | λ range **0.26** | **Valid control** (r(logits,prob) 0.9999). λ range across frequency **0.26** against trained 0.89 — **3.4×** flatter. The frequency structure needs trained weights. |
| [`swin-t-r50-s0`](swin-t-r50-s0/notes.md) | swin-t, trained | 50 | `IMAGENET1K_V1` | **monotone** | Architecture screen. **monotone** in frequency (ρ -0.76); λ range 1.28, `prob` λ +0.109 at R² 0.911. |
| [`swin-t-scramble-r50-s0`](swin-t-scramble-r50-s0/notes.md) | swin-t, scrambled | 50 | `IMAGENET1K_V1` | λ range **0.12** | **Valid control** (r(logits,prob) 0.9918). λ range across frequency **0.12** against trained 1.28 — **10.4×** flatter. The frequency structure needs trained weights. |
| [`swin-v2-t-r50-s0`](swin-v2-t-r50-s0/notes.md) | swin-v2-t, trained | 50 | `IMAGENET1K_V1` | **monotone** | Architecture screen. **monotone** in frequency (ρ -0.81); λ range 2.62, `prob` λ +0.121 at R² 0.940. |
| [`swin-v2-t-scramble-r50-s0`](swin-v2-t-scramble-r50-s0/notes.md) | swin-v2-t, scrambled | 50 | `IMAGENET1K_V1` | λ range **0.22** | **Valid control** (r(logits,prob) 0.9926). λ range across frequency **0.22** against trained 2.62 — **12.2×** flatter. The frequency structure needs trained weights. |
| [`vit-b-32-r50-s0`](vit-b-32-r50-s0/notes.md) | vit-b-32, trained | 50 | `IMAGENET1K_V1` | band **+0.21** | Architecture screen. mid band dips 0.21 (ρ +0.29); λ range 1.05, `prob` λ -0.546 at R² 0.927. |
| [`vit-b-32-scramble-r50-s0`](vit-b-32-scramble-r50-s0/notes.md) | vit-b-32, scrambled | 50 | `IMAGENET1K_V1` | λ range **0.13** | **Valid control** (r(logits,prob) 0.9999). λ range across frequency **0.13** against trained 1.05 — **8.0×** flatter. The frequency structure needs trained weights. |
| [`focalnet-tiny-srf-r50-s0`](focalnet-tiny-srf-r50-s0/notes.md) | focalnet-tiny-srf, trained | 50 | timm `focalnet-tiny-srf` | `prob` λ **+0.420** | Focal modulation in place of attention. λ-R² 0.941, band -0.22 (ρ -0.79). |
| [`focalnet-tiny-srf-scramble-r50-s0`](focalnet-tiny-srf-scramble-r50-s0/notes.md) | focalnet-tiny-srf, scrambled | 50 | timm `focalnet-tiny-srf` | **usable** | r(logits,prob) 0.947 at ratio 1.4e-3. Same reading as PoolFormer: usable, not pristine. |
| [`gmlp-s16-r50-s0`](gmlp-s16-r50-s0/notes.md) | gmlp-s16, trained | 50 | timm `gmlp-s16` | `prob` λ **-0.250** | **mlp only** — no attention and no convolution beyond the patch embedding. λ-R² 0.934, band -0.71 (ρ +0.29). |
| [`gmlp-s16-native-r50-s0`](gmlp-s16-native-r50-s0/notes.md) | gmlp-s16-native, trained | 50 | timm `gmlp-s16-native` | `prob` λ **-0.346** | The gmlp run again under the checkpoint's own normalisation, as a preprocessing sensitivity check. λ-R² 0.936, band -0.05 (ρ +0.60). |
| [`gmlp-s16-scramble-r50-s0`](gmlp-s16-scramble-r50-s0/notes.md) | gmlp-s16, scrambled | 50 | timm `gmlp-s16` | **clean** | r(logits,prob) 0.992 at ratio 8.7e-4 — the softmax stays in its affine regime. |
| [`poolformer-s12-r50-s0`](poolformer-s12-r50-s0/notes.md) | poolformer-s12, trained | 50 | timm `poolformer-s12` | `prob` λ **-0.034** | **pooling** in place of attention (the metaformer control). λ-R² 0.889, band +0.48 (ρ -0.74). |
| [`poolformer-s12-scramble-r50-s0`](poolformer-s12-scramble-r50-s0/notes.md) | poolformer-s12, scrambled | 50 | timm `poolformer-s12` | **usable** | r(logits,prob) 0.968 at ratio 7.1e-4. Logit magnitudes are sane (~2), so the softmax is not saturating, but this is visibly less clean than a LayerNorm control (0.9998). |
| [`resmlp-12-r50-s0`](resmlp-12-r50-s0/notes.md) | resmlp-12, trained | 50 | timm `resmlp-12` | `prob` λ **-0.315** | **mlp only**, with a learned per-channel affine in place of layernorm. λ-R² 0.892, band +0.40 (ρ +0.55). |
| [`resmlp-12-scramble-r50-s0`](resmlp-12-scramble-r50-s0/notes.md) | resmlp-12, scrambled | 50 | timm `resmlp-12` | **BROKEN** | r(logits,prob) **0.590** at ratio 1.3e-4, max |D_logits| **2409** against ~2 for the others, and **42 of 114 taps pinned at the λ = +4 bound** (trained companion: 0). ResMLP replaces LayerNorm with a learned per-channel Affine, which does not renormalise by the input, so nothing absorbs the permuted scales and the logits explode. **Do not table this against any other control.** |
| [`xcit-nano-12-p16-r50-s0`](xcit-nano-12-p16-r50-s0/notes.md) | xcit-nano-12-p16, trained | 50 | timm `xcit-nano-12-p16` | `prob` λ **-0.469** | Cross-covariance attention (over channels, not tokens). λ-R² 0.745, band -0.57 (ρ -0.14). |
| [`vgg19-r250-s0-alllayers-linear`](vgg19-r250-s0-alllayers-linear/notes.md) | VGG-19, trained | 250 | `IMAGENET1K_V1` | 45 taps, linear grid | Grid control. Profile survives: mean \|Δλ\| 0.045, **44/44** steps agree in direction. |
| [`vgg19-scramble-r250-s0-alllayers-linear`](vgg19-scramble-r250-s0-alllayers-linear/notes.md) | VGG-19, scrambled | 250 | `IMAGENET1K_V1` | 45 taps, linear grid | Control for the above. Mean \|Δλ\| 0.024; read against its R² 0.72, which is what makes λ here uninformative. |
| [`vgg19-r250-s0-alllayers-fixed`](vgg19-r250-s0-alllayers-fixed/notes.md) | VGG-19, trained | 250 | `IMAGENET1K_V1` | all 45 taps | Depth profile. conv median +0.69, `prob` +0.165. (Its λ 0.922 at conv1_1 is the noise floor, not a measurement — see `data-r250-s0`.) |
| [`vgg19-scramble-r250-s0-alllayers-fixed`](vgg19-scramble-r250-s0-alllayers-fixed/notes.md) | VGG-19, scrambled | 250 | `IMAGENET1K_V1` | all 45 taps | Control. λ ≈ +0.17 looks log-like; its R² 0.918 and 41% non-monotone cells are what separate it. |
| [`vgg19-r250-s0-alllayers-fixed-caffe`](vgg19-r250-s0-alllayers-fixed-caffe/notes.md) | VGG-19, trained | 250 | converted Caffe | all 45 taps | Conv stack flatly **linear** (λ +1.06, R² 0.999); one ReLU takes it to +0.21, `prob` +0.059. |
| [`vgg19-scramble-r250-s0-alllayers-fixed-caffe`](vgg19-scramble-r250-s0-alllayers-fixed-caffe/notes.md) | VGG-19, scrambled | 250 | converted Caffe | all 45 taps | Control. Runs *away* to λ ≈ **+2.75** (R² 0.972) — supralinear, log-like at 0/45 taps. |
| [`vgg19-scramble-r250-s0-alllayers`](vgg19-scramble-r250-s0-alllayers/notes.md) | VGG-19, scrambled | 250 | `IMAGENET1K_V1` | all 45 taps | **Superseded** — ran before the tap fix; conv taps hold ReLU output. |
| [`vgg19-r250-s0`](vgg19-r250-s0/notes.md) | VGG-19, trained | 250 | `IMAGENET1K_V1` | **0.928** (`classifier.3`) | The documented grid. Disagrees with Method.md on three counts. |
| [`vgg19-scramble-r250-s0`](vgg19-scramble-r250-s0/notes.md) | VGG-19, scrambled | 250 | `IMAGENET1K_V1` | 0.924 (`features.19`) | Control. Exceeds the trained net at the early/middle taps. |
| [`vgg19-r50-s0-in1k`](vgg19-r50-s0-in1k/notes.md) | VGG-19, trained | 50 | `IMAGENET1K_V1` | 0.921 (`classifier.3`) | Separates weight lineage from reps: lineage is the cause. |
| [`vgg19-scramble-r50-s0-in1k`](vgg19-scramble-r50-s0-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 0.924 (`features.19`) | Control, seed 0, and the seed-sweep write-up. |
| [`vgg19-scramble-r50-s1-in1k`](vgg19-scramble-r50-s1-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 0.941 (`features.19`) | Seed 1. `prob` 0.863 — the high end. |
| [`vgg19-scramble-r50-s2-in1k`](vgg19-scramble-r50-s2-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 0.920 (`features.19`) | Seed 2. `prob` 0.704. |
| [`vgg19-scramble-r50-s3-in1k`](vgg19-scramble-r50-s3-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 0.843 (`features.19`) | Seed 3. `prob` 0.693 — the low end. |
| [`vgg19-r50-s0`](vgg19-r50-s0/notes.md) | VGG-19, trained | 50 | converted Caffe | 0.976 (`prob`) | Differs from the `IMAGENET1K_V1` runs because of its checkpoint, not its reps. |
| [`vgg19-scramble-r50-s0`](vgg19-scramble-r50-s0/notes.md) | VGG-19, scrambled | 50 | converted Caffe | 0.428 (`prob`) | Control. Sits 0.33 below both `IMAGENET1K_V1` controls. |
| [`vgg19-r50-s0-alllayers-caffe`](vgg19-r50-s0-alllayers-caffe/notes.md) | VGG-19, trained | 50 | converted Caffe | **1/45 taps on the floor** | The reps companion to the r250 run: only `features.0` falls with reps (2.222 ≈ √5). The flat λ≈1 conv stack is **real**. |
| [`vgg19-scramble-r50-s0-alllayers-caffe`](vgg19-scramble-r50-s0-alllayers-caffe/notes.md) | VGG-19, scrambled | 50 | converted Caffe | 1/45 on the floor | Control. Its supralinear λ ≈ +2.76 is a real measurement too. |
| [`vgg19-r50-s0-alllayers-in1k`](vgg19-r50-s0-alllayers-in1k/notes.md) | VGG-19, trained | 50 | `IMAGENET1K_V1` | 1/45 on the floor | `features.1`/`.2` are the only partial cases (31%, 36%) — and where λ +1.67 vs λ_mod +1.01 exposes it. |
| [`vgg19-scramble-r50-s0-alllayers-in1k`](vgg19-scramble-r50-s0-alllayers-in1k/notes.md) | VGG-19, scrambled | 50 | `IMAGENET1K_V1` | 1/45 on the floor | Control. Max noise fraction outside `features.0/1/2` is 1.8%. |
| [`vgg19-scramble-r50-s0-p1-alllayers-caffe`](vgg19-scramble-r50-s0-p1-alllayers-caffe/notes.md) | VGG-19, scrambled | 50 | converted Caffe | `prob` **0.516** | Permutation sweep, `--seed 0` fixed. The high end — and peaks at `features.0`, so no tap beats the noise floor. |
| [`vgg19-scramble-r50-s0-p2-alllayers-caffe`](vgg19-scramble-r50-s0-p2-alllayers-caffe/notes.md) | VGG-19, scrambled | 50 | converted Caffe | `prob` 0.443 | Permutation 2. λ +3.00. |
| [`vgg19-scramble-r50-s0-p3-alllayers-caffe`](vgg19-scramble-r50-s0-p3-alllayers-caffe/notes.md) | VGG-19, scrambled | 50 | converted Caffe | `prob` **0.422** | Permutation 3, the low end. Sweep spans 0.422–0.516; the paper's 0.60 is outside it. |
| [`data-r250-s0`](data-r250-s0/notes.md) | raw pixels | 250 | none | λ **+0.925**, R² 0.985 | The paper's `data` row, and the metric's **noise floor**. `features.0` reproduces it to 3 decimals on both checkpoints. |
| [`data-r50-s0`](data-r50-s0/notes.md) | raw pixels | 50 | none | D(50)/D(250) = **2.237** | Companion to the above: makes the 1/√reps scaling checkable (√5 = 2.236). |
| [`hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-vlm`](hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-vlm/notes.md) | SmolVLM-256M, trained | **2** | transformers | `text_model.layers.14` λ **+0.020**, R² 0.965 | **The first non-classifier.** A generative VLM's hidden taps sit at the log law; all three intervals contain 0 and none overlaps the control. **Do not quote its `prob`** (λ-R² 0.857, per-frequency +0.03 to +2.77). |
| [`hf-HuggingFaceTB-SmolVLM-256M-Instruct-scramble-r2-s0-vlm`](hf-HuggingFaceTB-SmolVLM-256M-Instruct-scramble-r2-s0-vlm/notes.md) | SmolVLM-256M, scrambled | **2** | transformers | `text_model.layers.14` λ **+0.524**, R² 0.978 | **valid** — zero BatchNorm, no pinned taps, max D_logits 2.93. Separated from trained at all three hidden taps; λ flat across frequency (range 0.22–0.27 vs 0.89–1.03). |
| [`hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks`](hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks/notes.md) | SmolVLM-256M, trained | **2** | transformers | vision **+0.679 → −0.114**, decoder +0.115 → −0.120 | **The VLM depth profile**, 44 taps. The compression is built in the *vision tower* (0.79 over 12 blocks); the 30-layer decoder moves 0.27. λ-R² 0.935–0.988 across all 42 blocks. |
| [`hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks-contrastprompt`](hf-HuggingFaceTB-SmolVLM-256M-Instruct-r2-s0-blocks-contrastprompt/notes.md) | SmolVLM-256M, trained | **2** | transformers | mean \|Δλ\| **0.0000** / **0.0006** | Same run with the instruction `'How much contrast does this pattern have?'`. **The prompt changes nothing** in vision (bit-identical) or decoder (≤0.0020) — causal masking puts image tokens before the prompt. Readout moves but is unresolved. |

> **Fifteen lineage runs landed on 2026-08-04.** One architecture, several
> training runs — the generalisation of the VGG-19 checkpoint result. Torchvision
> ships a second ImageNet checkpoint for **26** architectures, and a timm tag *is*
> a lineage, so `--model resnet50:IMAGENET1K_V2` and `--model timm:resnet50.a1_in1k`
> now reach them. **Read the control row first**: `timm:resnet50.tv_in1k` is
> torchvision's `IMAGENET1K_V1` re-hosted and returns mean |Δλ| **0.000** over 110
> taps, so back-end and preprocessing are excluded and every other difference here
> is the weights. All are **one seed**; the noise scale comes from the older
> three-seed sweeps. The BN nets (everything but the VGGs) cannot take the
> scramble, so those pairs have no untrained control. See `wiki/Results.md`,
> "One architecture, several training runs".

| [`resnet50-r50-s0-v2`](resnet50-r50-s0-v2/notes.md) | ResNet-50, trained | 50 | `IMAGENET1K_V2` | `logits` Δλ **-0.476** | Lineage pair with [`resnet50-r50-s0`](resnet50-r50-s0/notes.md): same architecture and framework, **no conversion**, only the training recipe. `logits` +0.044 → -0.432 (R² 0.95 → 0.92); `prob` λ -0.426 at R² 0.80. |
| [`resnext50-32x4d-r50-s0-v2`](resnext50-32x4d-r50-s0-v2/notes.md) | ResNeXt-50, trained | 50 | `IMAGENET1K_V2` | `logits` Δλ **-0.451** | Lineage pair with [`resnext50-32x4d-r50-s0`](resnext50-32x4d-r50-s0/notes.md): same architecture and framework, **no conversion**, only the training recipe. `logits` -0.031 → -0.482 (R² 0.96 → 0.96); `prob` λ -0.354 at R² 0.96. |
| [`regnet-y-400mf-r50-s0-v2`](regnet-y-400mf-r50-s0-v2/notes.md) | RegNet-Y-400MF, trained | 50 | `IMAGENET1K_V2` | `logits` Δλ **-0.403** | Lineage pair with [`regnet-y-400mf-r50-s0`](regnet-y-400mf-r50-s0/notes.md): same architecture and framework, **no conversion**, only the training recipe. `logits` +0.354 → -0.050 (R² 0.95 → 0.86); `prob` λ -0.296 at R² 0.89. |
| [`mobilenet-v2-r50-s0-v2`](mobilenet-v2-r50-s0-v2/notes.md) | MobileNet-V2, trained | 50 | `IMAGENET1K_V2` | `logits` Δλ **+0.074** | Lineage pair with [`mobilenet-v2-r50-s0`](mobilenet-v2-r50-s0/notes.md): same architecture and framework, **no conversion**, only the training recipe. `logits` +0.057 → +0.131 (R² 0.96 → 0.95); `prob` λ +0.163 at R² 0.92. |
| [`mobilenet-v3-large-r50-s0-v2`](mobilenet-v3-large-r50-s0-v2/notes.md) | MobileNet-V3-L, trained | 50 | `IMAGENET1K_V2` | `logits` Δλ **+0.071** | Lineage pair with [`mobilenet-v3-large-r50-s0`](mobilenet-v3-large-r50-s0/notes.md): same architecture and framework, **no conversion**, only the training recipe. `logits` -0.273 → -0.203 (R² 0.90 → 0.87); `prob` λ -0.585 at R² 0.78. |
| [`timm-resnet50-tv-in1k-r50-s0`](timm-resnet50-tv-in1k-r50-s0/notes.md) | ResNet-50, trained | 50 | timm `.tv_in1k` | mean \|Δλ\| **0.000** | **The back-end control.** Torchvision `IMAGENET1K_V1` re-hosted: identical to [`resnet50-r50-s0`](resnet50-r50-s0/notes.md) at all 110 shared taps, r = +1.000. Excludes library and preprocessing as explanations for the lineage spread. |
| [`timm-resnet50-gluon-in1k-r50-s0`](timm-resnet50-gluon-in1k-r50-s0/notes.md) | ResNet-50, trained | 50 | timm `.gluon_in1k` | `logits` λ **-0.393** | Gluon / Bag of Tricks. One of five lineages of the same 25.6 M parameters; λ spans 0.476 at `logits` across them (11× the 3-seed sd) while the duplicate returns 0.000. `prob` λ -0.313 at R² 0.95. |
| [`timm-resnet50-a3-in1k-r50-s0`](timm-resnet50-a3-in1k-r50-s0/notes.md) | ResNet-50, trained | 50 | timm `.a3_in1k` | `logits` λ **-0.261** | RSB A3, 100 ep, LAMB/BCE. One of five lineages of the same 25.6 M parameters; λ spans 0.476 at `logits` across them (11× the 3-seed sd) while the duplicate returns 0.000. `prob` λ -0.381 at R² 0.95. |
| [`timm-resnet50-a1-in1k-r50-s0`](timm-resnet50-a1-in1k-r50-s0/notes.md) | ResNet-50, trained | 50 | timm `.a1_in1k` | `logits` λ **-0.166** | RSB A1, 600 ep, LAMB/BCE. One of five lineages of the same 25.6 M parameters; λ spans 0.476 at `logits` across them (11× the 3-seed sd) while the duplicate returns 0.000. `prob` λ -0.347 at R² 0.89. |
| [`vgg16-r50-s0`](vgg16-r50-s0/notes.md) | VGG-16, trained | 50 | `IMAGENET1K_V1` | conv-stack median λ **+0.664** | **The VGG-19 lineage result reproduces on VGG-16**: mean |Δλ| 0.353 over 39 taps against VGG-19's 0.328, and Caffe again holds λ ≈ 1 through the conv stack. `prob` λ +0.093 at R² 0.96. |
| [`vgg16-r50-s0-caffe`](vgg16-r50-s0-caffe/notes.md) | VGG-16, trained | 50 | converted Caffe | conv-stack median λ **+1.025** | **The VGG-19 lineage result reproduces on VGG-16**: mean |Δλ| 0.353 over 39 taps against VGG-19's 0.328, and Caffe again holds λ ≈ 1 through the conv stack. `prob` λ +0.135 at R² 0.97. |
| [`vgg16-scramble-r50-s0`](vgg16-scramble-r50-s0/notes.md) | VGG-16, scrambled | 50 | `IMAGENET1K_V1` | `prob` λ **+0.729** | **Valid control** — VGG-16 is BN-free. Against [`vgg16-r50-s0`](vgg16-r50-s0/notes.md)'s +0.093; mean log-R² separates too. |
| [`vgg16-scramble-r50-s0-caffe`](vgg16-scramble-r50-s0-caffe/notes.md) | VGG-16, scrambled | 50 | converted Caffe | `prob` λ **+0.861** | **Valid control** — VGG-16 is BN-free. Against [`vgg16-r50-s0-caffe`](vgg16-r50-s0-caffe/notes.md)'s +0.135; mean log-R² separates too. |
| [`timm-vit-base-patch16-224-orig-in21k-ft-in1k-r50-s0`](timm-vit-base-patch16-224-orig-in21k-ft-in1k-r50-s0/notes.md) | ViT-B/16, trained | 50 | timm `.orig_in21k_ft_in1k` | `prob` λ **-0.096** | The original Dosovitskiy ViT weights (21k → 1k). Half of the tightest pair here: architecture, data and objective all fixed against the AugReg half. |
| [`timm-vit-base-patch16-224-augreg-in21k-ft-in1k-r50-s0`](timm-vit-base-patch16-224-augreg-in21k-ft-in1k-r50-s0/notes.md) | ViT-B/16, trained | 50 | timm `.augreg_in21k_ft_in1k` | mean \|Δλ\| **0.380** | **Largest depth-profile shift of any lineage pair** (189 taps), and the lowest profile correlation (+0.387) — AugReg reshapes rather than shifts. `prob` λ -0.096 → +0.232, **46×** the 3-seed sd. |

> **`logness` was removed on 2026-07-26 and replaced by `λ`.** Prose in any
> `notes.md` written before that date quotes the retired statistic — a race
> between `D = a + b·log c` and `D = a + b·c`, which measured nothing useful
> because *neither* line fits this data (the trained net is convex in `log c` at
> 95% of cells, the scrambled control non-monotone at 41%). `λ` is the exponent
> of `D = a + b·(c^λ − 1)/λ`: **0 is the log law, 1 linear in contrast**, and it
> comes with a confidence interval. Nothing was re-run — `result.npz` holds the
> surfaces, so every directory here re-fits, and each `result.json` now records
> `lambda`, `lambda_ci` and `lambda_r2` where it recorded `logness`,
> `fit_quality` and `logness_r2diff`. **When a note and its `result.json`
> disagree, `result.json` is right.** Read λ against `lambda_r2`, never alone.
> See `wiki/Results.md` for what changed substantively.

> **`result.json` gained per-frequency λ on 2026-07-28, and 16 directories had
> a stale interval repaired.** The layer-level `lambda` is a **median over the
> eight frequencies**, and it discards more variation than the comparisons it
> gets used for — so `per_frequency[]` now carries `lambda`, `lambda_ci` and
> `lambda_r2` beside the log-fit `r2`/`slope`/`intercept` it already held.
>
> The repair: 16 directories carried `lambda_ci` endpoints read off the λ search
> grid rather than bisected, by an intermediate fitter that was never shipped —
> every endpoint an exact multiple of 0.025, and **29 intervals excluding their
> own point estimate**, some collapsed to zero width (`features.1` on the Caffe
> run read λ +1.2045 with CI [+1.200, +1.200]). All 360 affected intervals were
> too *narrow*. Regenerated from the committed surfaces: **λ point estimates are
> unchanged** (worst drift 7e-5 relative, on a λ of 2e-5), `result.npz` is
> byte-identical everywhere, and the numbers in `wiki/Results.md` were already
> the corrected ones because they came from live re-fits.

> **The per-frequency λ is recorded too, since 2026-07-28.** Each
> `per_frequency` entry carries `lambda`, `lambda_ci` and `lambda_r2` next to
> the log fit's `r2`/`slope`/`intercept`. The top-level `lambda` is the *median*
> of those eight, and it discards more than the differences it tends to get
> compared on — within one run λ spans up to 1.75 across frequency against 0.43
> between architectures, so two runs with equal median λ can have quite
> different responses. Again nothing was re-run; the surfaces already held it.

> **Corrected in the same pass: `lambda_ci` in 16 directories was stale.** Those
> files' interval endpoints all sat on a 0.025 grid, because they were written by
> an intermediate fitter and committed alongside — but not regenerated against —
> the bisecting version that shipped in `da87dd0`. The consequence is the one
> `fit.py` warns about at the bisection: **29 intervals excluded their own point
> estimate**, some collapsing to zero width (`features.1` on the Caffe run read
> λ = +1.2045 with a CI of `[+1.200, +1.200]`). All 360 affected intervals were
> too **narrow**, never too wide, so the stale files overstated precision.
> Regenerated from the surfaces: λ point estimates moved by at most 7.4e-5
> relative, no `result.npz` was touched, and the prose was unaffected because
> `wiki/Results.md` had been quoting live re-fits all along.

> **A contributed 23-run architecture screen landed on 2026-07-29** (17 trained
> nets + 6 scrambled controls, `--reps 50`, seed 0, all layers). It was verified
> before merging: all 331 bundled checksums pass, the packaged source snapshot is
> **byte-identical to `baa2fa5`** across every tracked non-results file, all 23
> carry `pretrained_verified: true`, the grids match every other run here, and the
> affine-prefix tap reads λ 0.93–0.97 on each — the noise floor, which is how a
> new architecture's pipeline announces that it works.
>
> Two things to carry from it. **The mid-band dip does not generalise**: 5 of the
> 17 are monotone in frequency with no band shape at all. And **λ pinned at the
> −3 search bound is not a measurement** — two runs have such a cell, and
> dropping it moves RegNet-Y's band contrast from +1.10 to +0.05. Every band
> number in these rows excludes pinned cells.
>
> The six controls are the screen's real contribution: they sit on BN-free nets,
> so all six are valid, and λ's range across frequency is **3.4–20.8× larger
> trained than scrambled in 6/6 pairs**. See `wiki/Results.md`.


> **A 10-run unusual-architecture screen landed on 2026-07-29**, via a new
> `timm:` back-end: gMLP, ResMLP, PoolFormer, FocalNet, XCiT — families that
> replace attention, or drop both attention and convolution. All 384 bundled
> checksums pass, all ten are `pretrained_verified`, all run at 224x224 on the
> repo's exact grids, and each reads λ 0.94–0.96 at its first tap (the noise
> floor). `run.json` records `commit: null` with reason "git not available or
> not a repository" — honest, but these ten cannot be pinned to a code revision
> the way the Actions runs can.
>
> **One of the four controls is broken, and BatchNorm is not why.**
> `resmlp-12-scramble` explodes: max |D_logits| **2409** against ~2 elsewhere,
> r(logits,prob) **0.590**, and **42 of 114 taps pinned at the λ = +4 bound**.
> ResMLP has *no* BatchNorm — it uses a learned per-channel Affine instead of
> LayerNorm, which never renormalises by the input, so nothing absorbs the
> permuted scales. The repo's rule ("valid where there are no BatchNorm running
> statistics") does not catch this. See `wiki/Results.md`.


Note that "best mean R²" is not `prob` for either r250 run — that is the finding,
not a slip. Read the r250 rows with the spacing CV in their notes: the scrambled
column reaches 0.76–0.92 with a CV of 3.5–4.1, so a high R² there is a line
through frequencies that disagree, not an even log ladder.

The r250 pair came from the GitHub path and committed themselves from the job;
the r50 pair were run in the sandbox on converted weights. `run.json`
distinguishes them (`environment.platform`, `weights.source`).

Re-fit and re-plot any run without torch, weights, or network:

```bash
python -m log_response.run --load results/vgg19-r50-s0 --panels out/panels.png
python -m log_response.run --load results/vgg19-r50-s0 --figures out/
```

## Conventions

- **Slug**: `<model>[-<variant>]-r<reps>-s<seed>` — the axes that actually vary.
  It does **not** encode weight lineage, so `vgg19-r50-s0` is ambiguous between
  the converted Caffe checkpoint and `IMAGENET1K_V1`. When they would collide,
  append a lineage tag (`-in1k`, `-caffe`); the workflow's `slug_suffix` input
  does this, and it refuses to start rather than overwrite an existing
  directory. Overwriting is worse than it looks: `save_run_dir` never clobbers
  `notes.md`, so the prose would survive while the numbers underneath changed.
- **Always committed**: the four files above. A surface is
  `n_layers × n_freq × n_contrast` floats — a few KB regardless of `--reps`, so
  even a 28 000-forward run costs ~7 KB. Keep every run.
- **Never committed**: figures (~100× larger than the data behind them, and
  regenerate from the npz), weights, activations.
- **`run.json` is the trust record.** `weights.pretrained_verified` must be
  `true` for any run quoted as a result; `false` means an untrained control and
  `null` means the question does not apply (synthetic back-end). `run.py`
  refuses to save a `false` run unless `--allow-random-init` was passed
  deliberately.

New runs write this layout themselves:

```bash
python -m log_response.run --model vgg19 --weights W --reps 250 \
    --save-run results/vgg19-r250-s0 --notes "full grid"
```

Runs produced on the GitHub-hosted path
([workflow](../.github/workflows/log-response.yml)) commit their own directory
here from the job — the sandbox cannot download Actions artifacts, so CI pushing
the result is what makes it reachable. Their `run.json` carries the runner's
provenance, so they are distinguishable from sandbox runs without a naming
convention for it.
