import json
import yaml
from pathlib import Path
import torch
import mlflow
from utils.utils import *
from utils.datasets import *
from models.model_components import *
from models.prototypical_components import *
import os
from dotenv import load_dotenv

load_dotenv()

ML_FLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")


def load_config(config_path="config.yaml"):
    """Load and process configuration file."""
    config = yaml.safe_load(open(config_path))
    config_setup_name = 'polyak' if config['polyak'] else 'no-polyak'
    distances_weights = dict(zip(config['setup']['scenario_config']['weights'],
                                 config['setup']['scenario_config']['vals']))

    config_setup_name = config_setup_name + '_' + \
        '_'.join([f"{k}-{v}" for k, v in distances_weights.items()])
    distances_weights = {key: eval(val)
                         for key, val in distances_weights.items()}

    return config, config_setup_name, distances_weights


def setup_mlflow(config, config_setup_name):
    """Initialize MLflow tracking."""
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Dos-Project")

    return mlflow.start_run(run_name=config_setup_name)


def log_config_params(config):
    """Log configuration parameters to MLflow."""
    mlflow.log_params({
        "n_experiments": config['setup']['params']['n_experiments'],
        "n_samples": config['setup']['params']['n_samples'],
        "n_episodes": config['setup']['params']['n_episodes'],
        "n_support": config['setup']['params']['n_support'],
        "n_query": config['setup']['params']['n_query'],
        "epochs": config['setup']['params']['epochs'],
        "learning_rate": config['setup']['params']['lr'],
        "polyak": config['polyak'],
        "polyak_decay": 0.999
    })


def train_experiment(model, optimizer, val_dataloader, episodes, config, experiment_num):
    """Run single training experiment with MLflow logging."""
    experiment_history = multi_space_episodic_training_with_polyak(
        model=model,
        optimizer=optimizer,
        val_dataloader=val_dataloader,
        episodes=episodes,
        epochs=config['setup']['params']['epochs'],
        weights=distances_weights,
        polyak=config['polyak'],
        polyak_decay=0.999
    )

    # Log metrics for each epoch
    for epoch, metrics in enumerate(zip(
        experiment_history["balanced_accuracy"],
        experiment_history["val_loss"],
        experiment_history["val_f1"]
    )):
        mlflow.log_metrics({
            f"balanced_accuracy": metrics[0],
            f"val_loss": metrics[1],
            f"val_f1": metrics[2]
        }, step=epoch)

    return experiment_history


def main():
    # Load configuration
    config, config_setup_name, distances_weights = load_config()

    # Setup paths
    train_history_path = Path(config['output']['train-history'])
    train_history_path.mkdir(parents=True, exist_ok=True)

    # Initialize data
    dm = DatasetManager(config)
    X_train, y_train, val_dataloader = dm.preprocess_data()
    n_features, n_classes = int(X_train.shape[1]), int(y_train.shape[1])

    # Initialize metrics tracking
    best_metrics = {
        'balanced_accuracy': [],
        'val_loss': [],
        'val_f1': []
    }

    # Start MLflow run
    with setup_mlflow(config, config_setup_name) as run:
        log_config_params(config)

        # Run experiments
        for i in range(config['setup']['params']['n_experiments']):
            with mlflow.start_run(run_name=f"exp_{i}", nested=True):
                print(f"\n###############################")
                print(
                    f"Starting experiment {i+1}/{config['setup']['params']['n_experiments']}")
                print("###############################\n")

                # Sample data
                X_train_sampled, y_train_sampled = stratified_sample(
                    datasets=[X_train, y_train],
                    n_samples=config['setup']['params']['n_samples'],
                    sample_per_class=config['setup']['params']['sample_per_class']
                )

                # Create model and optimizer
                model = MLP_MultiLabel(
                    n_features=n_features, n_classes=n_classes)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=config['setup']['params']['lr']
                )

                # Create episodes
                episodes, unique_indices = create_episodes(
                    X_train_sampled,
                    y_train_sampled,
                    n_episodes=config['setup']['params']['n_episodes'],
                    n_support=config['setup']['params']['n_support'],
                    n_query=config['setup']['params']['n_query']
                )
                print(
                    f"Total unique inputs used for training: {len(unique_indices)}")

                # Train and log
                experiment_history = train_experiment(
                    model, optimizer, val_dataloader, episodes, config, i
                )

                # Update best metrics
                best_metrics['balanced_accuracy'].append(
                    max(experiment_history["balanced_accuracy"]))
                best_metrics['val_loss'].append(
                    max(experiment_history["val_loss"]))
                best_metrics['val_f1'].append(
                    max(experiment_history["val_f1"]))

                # Save experiment history
                with open(f'{train_history_path}/exp_{i}.json', 'w') as f:
                    json.dump(experiment_history, f, indent=4)

        # Log final best metrics
        mlflow.log_metrics({
            "best_balanced_accuracy_mean": np.mean(best_metrics['balanced_accuracy']),
            "best_val_loss_mean": np.mean(best_metrics['val_loss']),
            "best_val_f1_mean": np.mean(best_metrics['val_f1']),
            "best_balanced_accuracy_std": np.std(best_metrics['balanced_accuracy']),
            "best_val_loss_std": np.std(best_metrics['val_loss']),
            "best_val_f1_std": np.std(best_metrics['val_f1'])
        })


if __name__ == "__main__":
    main()
