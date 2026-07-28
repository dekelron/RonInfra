# alexnet-r250-s0

`alexnet`, 250 reps/cell, best mean R² 0.963 at `prob`.

## What this run was for

AlexNet depth profile, 21 taps. Depth control: does the crossover to log need 33 layers, or is it carried by the classifier ReLU + softmax?

## What it showed

**The log response does not need depth.** `prob` returns λ = **+0.053**
[−0.04, +0.16] at R² 0.985, mean R² **0.963** — and `prob` is the **peak of all
21 taps**. AlexNet has 8 weight layers against VGG-19's 19, so whatever produces
the log law, it is not 33 layers of composition.

That is three of the paper's four §5 structural claims, on an architecture the
paper never used:

| claim | AlexNet | paper |
|---|---|---|
| `prob` mean R² | **0.963** | 0.98 |
| `prob` is the peak of all taps | ✓ | ✓ |
| early/middle layers much lower | 0.64–0.88 | ✓ |

Among the VGG-19 runs only the converted Caffe checkpoint did that
([`vgg19-r250-s0-alllayers-fixed-caffe`](../vgg19-r250-s0-alllayers-fixed-caffe/notes.md));
`IMAGENET1K_V1` peaks at `classifier.4` instead. AlexNet reproduces the
*structure* on canonical torchvision weights.

**The conv stack is not flat, and the ReLU sawtooth is here.** Median λ over
`features.1`+ is **+0.864**, declining from +1.27 at `features.1` to +0.41 at
`features.11` — and the decline is carried by the rectifiers, not the
convolutions:

| transition | mean Δλ | negative |
|---|---|---|
| `Conv2d → ReLU` | **−0.218** | 4/5 |
| `ReLU → Conv2d` | **+0.216** | 0/2 |

That is the `IMAGENET1K_V1` VGG-19 pattern (−0.155 / +0.166) reproduced on a
different architecture, and it is absent on the converted Caffe checkpoint
(+0.023 / −0.015). Both nets that show it carry **torchvision** weights and the
one that does not carries the original Oxford/Caffe weights, so the sawtooth
tracks the training recipe rather than the architecture. Stated as a
correlation across three runs, not a mechanism — per rule 4 the Caffe
disagreement stands.

**`features.0` is on the noise floor, as everywhere.** λ **+0.917** at R² 0.986
against the model-free [`data-r250-s0`](../data-r250-s0/notes.md)'s +0.925 —
and the scrambled run gives +0.932, because nothing about the weights can move
a tap whose population D is zero. The reps companion
([`alexnet-r50-s0`](../alexnet-r50-s0/notes.md)) confirms it is the *only* tap
on the floor.

**Caveat that carries over: `prob` is near its ceiling.** Max D_prob = 0.001795,
**89.7%** of the 2/1000 total-variation bound. Prefer `classifier.5`
(λ +0.133, R² 0.982) for a reading with no softmax and no ceiling.

## Reproduce

```
run.py --model alexnet --reps 250 --seed 0 --save-run results/alexnet-r250-s0 --notes AlexNet depth profile, 21 taps. Depth control: does the crossover to log need 33 layers, or is it carried by the classifier ReLU + softmax? --figures out/ --layers all
```

Code: `f9ab3861976a`. Weights: torchvision alexnet IMAGENET1K_V1.
