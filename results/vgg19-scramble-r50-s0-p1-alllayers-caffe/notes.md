# vgg19-scramble-r50-s0-p1-alllayers-caffe

`vgg19`, 50 reps/cell, best mean R² 0.748 at `features.0`.

## What this run was for

Caffe scrambled control, permutation seed 1 with the orientation/phase draws held at seed 0. Isolates permutation variance, which the earlier IN1K sweep could not.

## What it showed

Member of the four-permutation Caffe control sweep — `--scramble-seed 1` with
`--seed 0`, so the orientation/phase draws are **identical** to
[`vgg19-scramble-r50-s0-alllayers-caffe`](../vgg19-scramble-r50-s0-alllayers-caffe/notes.md)
and the permutation is the only thing that changed. This is the isolation the
earlier `IMAGENET1K_V1` sweep could not do, since one flag drove both seeds.

`prob` mean R² **0.516**, λ **+1.76** [+1.60, +1.96] at R² 0.995.
Peak over the 45 taps: `features.0` — i.e. **no real tap beats the noise floor's own 0.748**.

The sweep as a whole (p0–p3): `prob` mean R² **0.422 – 0.516**, spread 0.095,
sd 0.044. The paper's documented 0.60 is **outside** that range, so the
disagreement is not a one-permutation accident. See
[Results](../../wiki/Results.md#the-scrambled-control-is-not-a-single-number).

## Reproduce

```
run.py --model vgg19 --reps 50 --seed 0 --save-run results/vgg19-scramble-r50-s0-p1-alllayers-caffe --notes Caffe scrambled control, permutation seed 1 with the orientation/phase draws held at seed 0. Isolates permutation variance, which the earlier IN1K sweep could not. --figures out/ --weights /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth --layers all --scramble --scramble-seed 1
```

Code: `25b0e1726840`. Weights: local state_dict: /home/runner/work/RonInfra/RonInfra/../caffe-cache/vgg19_caffe.pth.
