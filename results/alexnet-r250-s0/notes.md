# alexnet-r250-s0

`alexnet`, 250 reps/cell, best mean R² 0.963 at `prob`.

## What this run was for

AlexNet depth profile, 21 taps. Depth control: does the crossover to log need 33 layers, or is it carried by the classifier ReLU + softmax?

## What it showed

_(fill in: the headline numbers, anything that disagreed with expectation)_

## Reproduce

```
run.py --model alexnet --reps 250 --seed 0 --save-run results/alexnet-r250-s0 --notes AlexNet depth profile, 21 taps. Depth control: does the crossover to log need 33 layers, or is it carried by the classifier ReLU + softmax? --figures out/ --layers all
```

Code: `f9ab3861976a`. Weights: torchvision alexnet IMAGENET1K_V1.
