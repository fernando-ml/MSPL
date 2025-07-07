# DoS Project

## How to Run

To run an experiment, you need to provide a configuration file and a dataset name as command-line arguments.

**Usage:**
```bash
python main.py <config_file.yaml> <dataset_name>
```

### Example

To run the prototypical mode with the `CICIDS2017` dataset, you can use the `test_config.yaml`:

```bash
python main.py test_config.yaml CICIDS2017
```

To run the baseline mode, you might use `baseline_config.yaml`:

```bash
python main.py baseline_config.yaml CICIDS2017
```

### Available Datasets
The following names are available for `<dataset_name>`:
- `CICEVSE_Network2024`
- `CICEVSE_PowerB2024`
- `CICIoV2024`
- `CICIDS2017`

## Data

Place the raw datasets in the **data** folder. If the parquets don't exist yet, use `combine_data.py` to create the parquets for `CICIoV` and `CICEVSE2024`. The processed parquets will be placed in `data/parquets`.

## Utils

See `utils/utils.py` for dataset loading and preprocessing functions.

## Experiment Configuration

The experiments are defined in a YAML file. Here is an example based on `test_config.yaml`:

```yaml
# Deep Learning Configuration
mode: "prototypical"

# datasets_config is defined in utils/datasets_config.yaml
# The dataset to use is now passed as a command-line argument.

# Deep Learning configuration for prototypical mode
dl_config:
  # Polyak values to test
  polyak_values:
    - true
    - false

  # Weight configurations to test
  weights_configs:
    - ["1", "0", "0", "0"]
    - ["0", "0", "0", "1"]
    - ["1/2", "0", "1/2", "0"]
    - ["0", "1/2", "1/2", "0"]

# Parameters
params:
  train-batch-size: 32
  val-batch-size: 512
  epochs: 3
  lr: 0.005
  n_episodes: 200
  n_support: 40
  n_query: 20
  n_experiments: 1
  n_samples: 200
  sample_per_class: 2

# Output configuration
output:
  train-history: "train_history/dl" # Directory for training outputs
  best_models: "models/saved_models/dl/best_model_" # Directory for saving best models

# Scenarios for non-prototypical modes
scenarios:
  - "logistic_regression"
  - "random_forest"
  - "gradient_boosting"
  - "knn"
```


