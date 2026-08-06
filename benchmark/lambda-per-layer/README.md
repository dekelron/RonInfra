# λ-per-layer: a reasoning test case

Six measurements of the same quantity on six neural networks, one value per
layer. The data is real, from committed experimental runs. Your task is to
work out what it shows.

## What was measured

Each network was shown **sinusoidal grating images** at a grid of contrasts and
spatial frequencies, and a **uniform gray image** of the same mean luminance.
For every layer, the response quantity is

```
D(c, f) = mean_i | a_i(grating at contrast c, frequency f) − a_i(gray) |
```

the mean absolute change in that layer's activation vector between gray and
grating. Each `(c, f)` cell is an average over **250 repetitions**, each with an
independently drawn orientation and spatial phase, both uniform over their full
range.

`D` rises with contrast. **λ** summarises *how* it rises, by fitting

```
D = a + b · (c^λ − 1) / λ
```

the Box–Tidwell / Box–Cox family, with `a` and `b` profiled out and λ searched
over **[−3, 4]**. λ is a single measured exponent, not a choice between models:

| λ | meaning |
|---|---|
| **0** | `D ∝ log c` — logarithmic in contrast |
| **1** | `D ∝ c` — linear in contrast |
| 0.5 | square root |
| < 0 | saturating faster than log |

λ is fitted **per spatial frequency**, and the reported `lambda` is the median
over the eight frequencies. It is dimensionless, so it is comparable across
layers, networks and grids.

## Files

| file | contents |
|---|---|
| `data/lambda_by_layer.csv` | one row per run × layer — the main table |
| `data/runs.json` | run-level metadata, plus the contrast and frequency grids |
| `data/per_frequency.json` | the eight per-frequency λ behind each median |

Columns in the CSV:

| column | meaning |
|---|---|
| `run_id`, `architecture` | which measurement |
| `layer_index`, `layer_name` | position in the forward pass, and the module's name |
| `lambda` | the exponent — median over the eight frequencies |
| `lambda_ci_lo`, `lambda_ci_hi` | 95% profile-F interval for λ |
| `lambda_r2` | R² of the λ fit — how well the family describes this layer |
| `log_fit_r2` | R² of a plain straight-line fit of `D` against `log c` |
| `lambda_alt_metric` | λ recomputed from the **other order of operations**: the mean over repetitions of each image's distance, rather than the distance between the means. The two estimate different population quantities |

## Ground rules

All six runs are **trained** networks (no randomised or ablated weights), all at
250 repetitions, seed 0, the same contrast grid, the same frequency grid, and
the same tapping policy (every leaf module). Nothing has been filtered or
smoothed; every layer that was measured is present.

Two runs share an `architecture` value and are distinguished only as `run_a` and
`run_b`. **What differs between them has been withheld from you** — it is a
single, specific difference, and identifying what kind of thing it must be is
part of the task.

## Questions

1. Describe the main structure in λ as a function of depth, for each run.
2. **`vgg19__run_a` sits at one extreme of the six.** Is it best described as an
   anomaly? Argue from the data.
3. What can you infer about the difference between `run_a` and `run_b`? Be
   explicit about what the data does and does not license you to conclude.
4. Which λ values in this dataset should **not** be read as measurements of the
   network, and how can you tell from the data alone?
5. If you had to nominate one of these six as the reference against which the
   others are deviations, could you? On what evidence?
