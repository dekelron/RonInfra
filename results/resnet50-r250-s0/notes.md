# resnet50-r250-s0

`resnet50`, 250 reps/cell, best mean R² 0.957 at `layer2.3.relu@2`.

## What this run was for

ResNet-50 depth profile, 160 taps incl. 32 reuse slots. First run possible after the hook reuse fix. Prediction under test: the identity path keeps an affine component alive deep into the net, so lambda ~ 1 should persist AND those taps should fall as 1/sqrt(reps) -- the floor signature VGG-19's conv stack did not show.

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model resnet50 --reps 250 --seed 0 --save-run results/resnet50-r250-s0 --notes ResNet-50 depth profile, 160 taps incl. 32 reuse slots. First run possible after the hook reuse fix. Prediction under test: the identity path keeps an affine component alive deep into the net, so lambda ~ 1 should persist AND those taps should fall as 1/sqrt(reps) -- the floor signature VGG-19's conv stack did not show. --figures out/ --layers all
```

Code: `7bdf878d43c4`. Weights: torchvision resnet50 IMAGENET1K_V1.
