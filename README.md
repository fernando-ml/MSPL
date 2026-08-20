# Multi-Space Prototypical Learning (MSPL)

## Run Vanilla

To run the prototypical mode, do:

```bash
python main.py config.yaml
```

To run the baseline mode, do:

```bash
python main.py baseline_config.yaml
```

## Data

Place the raw datasets in the **data** folder. If the parquets don't exist yet, use the combine_data.py to create the parquets for CICIoV and CICEVSE2024. The processed parquets will be placed in `data\parquets`.

## Utils

See `utils\utils.py` for dataset loading and preprocessing functions.

## Experiments Definition

possible config for experiment runs

```yaml
dataset:
  path:
    train:
    test:
    val:
  cols:
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
scenarios:
  - polyak: boolean
    vals: [1, .....]
output:
  train:
  best_models:
```
