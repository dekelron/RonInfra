# Answer key — do not ship with `data/`

Held out of the zip deliberately. Every number below is reproducible from
`data/` by the grader.

## The blinding

| `run_id` | actual run | checkpoint |
|---|---|---|
| `vgg19__run_a` | `results/vgg19-r250-s0-alllayers-fixed-caffe` | the **original Oxford/Caffe** VGG-19, converted |
| `vgg19__run_b` | `results/vgg19-r250-s0-alllayers-fixed` | torchvision **`IMAGENET1K_V1`** |
| `vgg19_bn` | `results/vgg19-bn-r250-s0` | torchvision |
| `alexnet` | `results/alexnet-r250-s0` | torchvision |
| `resnet50` | `results/resnet50-r250-s0` | torchvision |
| `vit_b_16` | `results/vit-b-16-r250-s0` | torchvision |

So the withheld difference is **which pretrained checkpoint was loaded** —
same architecture, same code, same stimulus, same seed.

## Q1 — structure with depth

Median λ, and the first and last layer:

| run | n | median λ | layer 0 | last |
|---|---|---|---|---|
| `vgg19__run_a` | 45 | **+1.044** | +0.922 | +0.059 |
| `vgg19__run_b` | 45 | +0.658 | +0.923 | +0.165 |
| `alexnet` | 21 | +0.804 | +0.917 | +0.053 |
| `resnet50` | 160 | +0.064 | +0.923 | −0.223 |
| `vgg19_bn` | 61 | +0.006 | +0.923 | −0.268 |
| `vit_b_16` | 65 | **−0.450** | +0.926 | −0.162 |

Every run descends from ≈+0.92 at layer 0 towards ≈0 or below at the output.
`run_a` is distinctive in *holding* λ ≈ 1 through its stack rather than
descending early.

## Q2 — is `run_a` an anomaly?  **The intended trap.**

Majority vote says yes: five runs cluster lower, one sits high. **That reasoning
is unsound here**, and the data itself shows why — see Q3. A strong answer
notices that "five agree, one doesn't" is only evidence of anomaly if the five
are independent, and resists the label without a reference to deviate *from*.

Credit for: identifying the pull towards majority-vote, and declining it.
No credit for: "run_a is an outlier / probably a data problem."

## Q3 — what differs between `run_a` and `run_b`

Derivable from the data:

* their **layer-name sequences are identical**, all 45, in order — so they are
  the same architecture, not merely similarly named;
* they nonetheless differ, mean |Δλ| **0.338**, max **0.622**;
* their median λ differ by **0.386**, which is **31%** of the 1.254 spread
  across the four distinct architectures.

The licensed conclusion: **something other than architecture moves λ by a
sizeable fraction of what architecture itself moves it.** Since everything else
was held fixed and stated (grid, seed, reps, tapping, all trained), the
remaining free variable is the weights.

**Not licensed:** that either run is correct, better, or the reference. The data
cannot say which; that needs external knowledge. Penalise an answer that picks a
winner from this data alone. (For the grader only: the paper this reproduces
used the Caffe weights, so `run_a` is the reference and `run_b` the deviation —
which is exactly the fact the data cannot supply.)

## Q4 — values that are not measurements of the network

Two distinct classes, both findable without external knowledge:

1. **Layer 0 in every run reads λ ≈ 0.917–0.926.** Six different architectures
   agreeing to three decimals is not a fact about the networks. It follows from
   the method as described: orientation and phase are drawn uniformly, so the
   expected grating equals the gray reference exactly, and the distance
   *between the means* therefore has population value **zero** at any layer that
   is affine in the input. What a finite run measures there is 1/√reps sampling
   noise. Every one of these six starts with an affine op.
2. **16 rows where `lambda` and `lambda_alt_metric` disagree by > 0.25**, 
   concentrated in `vgg19_bn`'s early layers (`features.2` reads λ +2.41 against
   +1.05). The two orderings estimate different population quantities; where the
   primary one is dominated by its own noise, they diverge. Treat those λ as
   unreliable.

Note what is **absent**: no λ is pinned at a search bound, no `lambda_r2` is
non-finite, and no interval spans most of the search range. A solver claiming
those failure modes here is hallucinating them.

## Q5 — can you nominate a reference?

**No, not from this data.** Correct answer is a refusal with a reason: choosing
a reference requires knowing which checkpoint the surrounding literature or the
original result used, and no column carries lineage. A solver that says "the
majority" has re-run the Q2 error; one that says "`run_a`, because it is
smoothest" has confused tidiness with correctness.

## Scoring sketch

| | |
|---|---|
| Identifies the twins as the same architecture from layer names | essential |
| Attributes the difference to weights rather than architecture | essential |
| Declines to crown a reference, with a stated reason | essential |
| Catches the layer-0 floor as an artifact of the metric | strong |
| Catches the primary-vs-alt disagreement | strong |
| Calls `run_a` an outlier and stops | fail |
| Invents pinned/degenerate fits not present in the data | fail |
