# DoS Project: Multi-Space Episodic Training Framework

A machine learning framework for network security focused on Denial of Service (DoS) detection using multi-space episodic training with prototypical networks. This project supports both deep learning (prototypical networks) and traditional machine learning approaches.

## Features

- **Multi-Space Learning**: Uses multiple distance metrics (euclidean, chebyshev, cosine, wasserstein) with configurable weights
- **Experiment Tracking**: Comprehensive integration with Weights & Biases (WandB) for experiment visualization and tracking
- **Dual Approach**: Supports both deep learning models (MLP-based prototypical networks) and traditional ML algorithms
- **Robust Evaluation**: Calculates extensive metrics including balanced accuracy, F1 score, MCC, and AUPRC

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-username/Dos-Project.git
cd Dos-Project
pip install -r requirements.txt
```

## Quick Start

Run an experiment using the command line interface:

```bash
# Run a prototypical network experiment
python main.py --dataset CIC-IDS-2017 --mode prototypical

# Run a traditional ML experiment
python main.py --dataset UNSW-NB15 --mode traditional

# Customize training parameters
python main.py --dataset CIC-IDS-2017 --mode prototypical --epochs 50 --samples 1000
```

## Data Setup

Place the raw datasets in the **data** folder. The processed parquets will be stored in `data/parquets`.

Supported datasets:
- CIC-IDS-2017
- UNSW-NB15 
- CICEVSE_Network2024
- CICIoV

## Configuration

The project uses separate configuration files for different approaches:

1. `config_dl.yaml`: Configuration for deep learning (prototypical network) experiments
2. `config_traditional.yaml`: Configuration for traditional ML experiments

You can override configuration parameters via command line arguments (see `python main.py --help`).

### Configuration Example

```yaml
# Deep Learning Configuration
# Dataset is specified via command line: --dataset DATASET_NAME
mode: "prototypical"

# Deep Learning configuration
dl_config:
  # Polyak values to test
  polyak_values:
    - true
    - false

  # Weight configurations to test
  weights_configs:
    - ["1", "0", "0", "0"]
    - ["0", "1/2", "0", "1/2"]
    - ["0", "1/3", "1/3", "1/3"]
    - ["1/3", "1/3", "1/3", "0"]
    - ["1/4", "1/4", "1/4", "1/4"]

# Parameters
params:
  train-batch-size: 32
  val-batch-size: 512
  epochs: 10
  lr: 0.005
  n_episodes: 200
  n_support: 40
  n_query: 20
  n_experiments: 40
  n_samples: 200
  sample_per_class: 2

# Output configuration
output:
  train-history: "train_history/dl" # Directory for training outputs
  best_models: "models/saved_models/dl/best_model_" # Directory for saving best models
```

## WandB Integration

The project includes comprehensive integration with Weights & Biases for experiment tracking. See [README_WANDB.md](README_WANDB.md) for detailed instructions.

Key logged metrics include:
- Training and validation performance metrics
- Per-class metrics and confusion matrices
- Feature importance for traditional models
- Embedding visualizations for prototypical networks
- Standard deviation across multiple experiment runs

To use WandB:

```bash
# Login to WandB
wandb login

# Run an experiment (metrics will be logged automatically)
python main.py --dataset CIC-IDS-2017 --mode prototypical
```

## Project Structure

```
Dos-Project/
├── data/                # Dataset directory
├── models/              # Model implementations
│   ├── model_components.py      # MLP and neural components
│   ├── prototypical_components.py # Prototypical network components
│   └── saved_models/          # Saved model weights
├── utils/               # Utility functions
│   ├── config_parser.py    # Configuration handling
│   ├── metrics.py      # Metrics calculation functions
│   ├── utils.py        # General utilities
│   └── wandb_logger.py # WandB logging utilities
├── config_dl.yaml      # Deep learning configuration
├── config_traditional.yaml # Traditional ML configuration
├── main.py             # Main entry point
├── trainer.py          # Training pipeline
└── requirements.txt    # Dependencies
```

## MLFlow Support (Future)

Support for MLFlow experiment tracking is planned:

```bash
mlflow server --backend-store-uri postgresql://uri
