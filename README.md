# TODO

- [ ] wrap dataset loading in a function... handle different sizes, formats, cols to drop, etc....

- [ ] wrap preprocessing in a function

- [ ] add ML flow logging .... for each epoch in experiments log to ML flow, also log to ML flow at the end of each experiment

## ML Flow

to start the server using postgres as backend

```bash
mlflow server --backend-store-uri postgresql://uri
```

## Experiments Definition

possible config for experiment runs

```yaml
dataset:
  path:
    train:
    test:
    val:
  cols:
setup:
  params:
    train-batch-size: 1
    val-batch-size:
    epochs:
    lr:
    n_episodes:
    n_support:
    n_query:
    n_experiments:
    n_samples:
    sample_per_class:
  scenario:
    polyak:
      weights: ["euclidean", ......]
      vals: [1, .....]
    no-polyak:
      weights: [eucl, chebyshev, cosine]
      vals: [1, 0, 0]
output:
  train:
  best_models:
```
