# Kausik — Weber-Fechner law in machine learning

- **Author:** Balas Natarajan Kausik
- **Venue / Year:** arXiv 2022
- **IDs:** arXiv:2204.11834 ("Accelerating Machine Learning via the Weber-Fechner
  Law"); arXiv:2208.11236 ("Psychophysical Machine Learning")
- **Cluster:** I — theory / applied Weber-Fechner
- **Sourcing:** `snippet`

## What it does

Proposes **imposing** the Weber-Fechner law on learning — e.g., via a logarithmic
power series of a network's sorted outputs, or as a loss-function term — to speed
up / improve learning of human concepts (demonstrated on MNIST with few
iterations).

## Key findings

- Conformance with Weber-Fechner can **accelerate** simple networks learning
  human-aligned concepts.

## Relevance to our artifact

Important distinction to keep straight: Kausik **builds Weber-Fechner in by
design**; we **measure whether it emerges** in models not trained for it (as in
Dekel-2017). Opposite directions.

## How we cite / differentiate

Cite to preempt "isn't the log law just imposed?" — no: our target models are not
trained with a Weber-Fechner objective; emergence is the finding. Differentiate
emergence-measurement vs. imposed-inductive-bias.
