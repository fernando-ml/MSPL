import json
import sklearn.preprocessing
import yaml
import time
from pathlib import Path
import torch
from utils.utils import *
from utils.datasets import *
from models.model_components import *
from models.prototypical_components import *
from models.no_DL_models import *
import wandb

if len(sys.argv) < 3:
    print("Please provide the path to the YAML config file and the dataset name.")
    sys.exit(1)


yaml_path = sys.argv[1]
dataset_name = sys.argv[2]
config = yaml.safe_load(open(yaml_path))
config['selected-dataset'] = dataset_name

if len(sys.argv) == 4:
    log_to_wandb = True
    wand_project = str(sys.argv[3])
else:
    log_to_wandb = False


def get_config_setup_name(scenario):
    """Generate configuration setup name based on scenario parameters."""
    if isinstance(scenario, dict):
        # Handle prototypical mode with weights and polyak
        weights = ["euclidean", "chebyshev", "cosine", "wasserstein"]
        config_setup_name = 'polyak' if scenario.get('polyak', False) else 'no-polyak'
        
        if 'weights' in scenario:
            distances_weights = dict(zip(weights, scenario['weights']))
            config_setup_name = config_setup_name + '_' + \
                '_'.join([f"{k}-{v}" for k, v in distances_weights.items()])
            distances_weights = {key: eval(val) if isinstance(val, str) else val
                               for key, val in distances_weights.items()}
        else:
            distances_weights = None
            
        return config_setup_name, distances_weights
    else:
        # Handle non-DL mode with simple scenario names
        return scenario, None


def generate_scenarios(config):
    """Generate scenarios based on the configuration."""
    scenarios = []
    
    if config['mode'] == 'prototypical':
        # Generate scenarios for prototypical mode
        polyak_values = config['dl_config']['polyak_values']
        weights_configs = config['dl_config']['weights_configs']
        
        for polyak in polyak_values:
            for weights in weights_configs:
                scenario = {
                    'polyak': polyak,
                    'weights': weights
                }
                scenarios.append(scenario)
    else:
        # For non-DL mode, use the scenarios as provided
        scenarios = config['scenarios']
    
    return scenarios

def init_wandb(config: dict,
               run_name: str,
               project_name: str = "DOS-Project",
               experiment_type: str = "prototypical",
               polyak: bool = False) -> wandb.sdk.wandb_run.Run:
    """
    Initialize a wandb run with appropriate configuration.

    Args:
        config (Dict): Configuration dictionary
        run_name (str): Name for this specific run
        project_name (str): Name of the wandb project
        experiment_type (str): Type of experiment (prototypical or traditional)

    Returns:
        wandb.Run: Initialized wandb run
    """
    # Add datetime to run name for better tracking
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    run_name_with_timestamp = f"{run_name}_{timestamp}"

    # Initialize wandb configuration
    wandb_config = {
        "dataset": config.get('selected-dataset', 'unknown'),
        "mode": config.get('mode', experiment_type),
        "experiment_type": experiment_type,
    }

    # Add appropriate hyperparameters based on experiment type
    if experiment_type == "prototypical":
        if 'polyak_values' in config.get('dl_config', {}):
            wandb_config["polyak_values"] = config['dl_config']['polyak_values']
        elif 'polyak' in run_name:
            wandb_config["polyak"] = "yes" if "polyak" in run_name else "no"

        # Add other relevant parameters
        params = config.get('params', {})
        wandb_config.update({
            "epochs": params.get('epochs', 0),
            "learning_rate": params.get('lr', 0),
            "n_episodes": params.get('n_episodes', 0),
            "n_support": params.get('n_support', 0),
            "n_query": params.get('n_query', 0),
            "n_samples": params.get('n_samples', 0),
            "sample_per_class": params.get('sample_per_class', 0),
            "train_batch_size": params.get('train-batch-size', 0),
            "val_batch_size": params.get('val-batch-size', 0),
        })
    else:
        # Traditional ML parameters
        params = config.get('params', {})
        wandb_config.update({
            "n_samples": params.get('n_samples', 0),
            "sample_per_class": params.get('sample_per_class', 0),
            "model_types": config.get('nonDL_config', {}).get('model_types', []),
        })

    # Initialize the run
    run = wandb.init(
        project=project_name,
        name=run_name_with_timestamp,
        config=wandb_config,
        reinit=True,  # Allow reinitializing if needed
        tags=[experiment_type, "polyak" if polyak else "no-polyak"],
    )

    return run


# Generate scenarios based on configuration
scenarios = generate_scenarios(config)

for scenario in scenarios:

    distances_weights = None

    if config['mode'] == 'prototypical':
        config_setup_name, distances_weights = get_config_setup_name(scenario)
    else:
        config_setup_name = scenario
    print(f"{5 * '#'} Starting scenario {config_setup_name} {5 * '#'}")

    use_polyak = 'polyak' in config_setup_name.lower(
    ) and 'no-polyak' not in config_setup_name.lower()

    if log_to_wandb:
        
        if config['mode'] == 'prototypical':
            # Create clean run name for prototypical mode
            run_name = f"{config['selected-dataset']}_{config_setup_name}"
        else:
            # Use scenario name for non-DL mode
            run_name = f"{config['selected-dataset']}_{scenario}"
        
        run = init_wandb(
        config=config,
        run_name=run_name,
        project_name="Local-test" if not wand_project else wand_project,
        polyak=use_polyak,
    )

    dm = DatasetManager(config)
    X_train, y_train, val_dataloader = dm.preprocess_data()
    n_features, n_classes = int(X_train.shape[1]), int(y_train.shape[1])
    train_history_path = config['output']['train-history']
    Path(train_history_path).mkdir(parents=True, exist_ok=True)

    X_val, y_val = dataloader_to_numpy(dataloader=val_dataloader)

    best_metrics = {'balanced_accuracy': [],
                    'val_loss': [],
                    'val_f1': [],
                    'val_AUPRC': [],
                    'val_MCC': [],
                    'n_experiment': [],
                    'training_time': [],
                    }

    print(
        f'results/{config_setup_name.replace("/", "over").replace(".", "point")}.json')

    for i in range(config['params']['n_experiments']):
        print(
            f"\n####################\nStarting experiment {i+1}/{config['params']['n_experiments']}\n#######################\n")

        X_train_sampled, y_train_sampled = stratified_sample(datasets=(X_train, y_train),
                                                             n_samples=config['params']['n_samples'],
                                                             sample_per_class=config['params']['sample_per_class'])
        if config['mode'] == 'prototypical':
            model = MLP_MultiLabel(n_features=n_features, n_classes=n_classes)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=config['params']['lr'])
            episodes, unique_indices = create_episodes(X_train_sampled,
                                                       y_train_sampled,
                                                       n_episodes=config['params']['n_episodes'],
                                                       n_support=config['params']['n_support'],
                                                       n_query=config['params']['n_query'])

            print(
                f"Total unique inputs used for training: {len(unique_indices)}")
            experiment_start_time = time.time()
            experiment_history = multi_space_episodic_training_with_polyak(model=model, optimizer=optimizer,
                                                                           val_dataloader=val_dataloader,
                                                                           episodes=episodes,
                                                                           epochs=config['params']['epochs'],
                                                                           weights=distances_weights,
                                                                           polyak=scenario.get('polyak', False),
                                                                           polyak_decay=0.999,
                                                                     best_model_path=f'{config["output"]["best_models"]}{config_setup_name.replace("/", "over").replace(".", "point")}.pth')
            experiment_time = time.time() - experiment_start_time
            best_metrics['balanced_accuracy'].append(
                max(experiment_history["balanced_accuracy"]))
            best_metrics['val_loss'].append(
                max(experiment_history["val_loss"]))
            best_metrics['val_f1'].append(max(experiment_history["val_f1"]))
            best_metrics['val_AUPRC'].append(
                max(experiment_history["val_AUPRC"]))
            best_metrics['val_MCC'].append(max(experiment_history["val_MCC"]))
            best_metrics['n_experiment'].append(i)
            best_metrics['training_time'].append(experiment_time)

            with open(f'{train_history_path}/exp_{i}.json', 'w') as f:
                json.dump(experiment_history, f, indent=4)

        else:
            model = get_non_dl_model(scenario)
            experiment_start_time = time.time()
            model.fit(X_train_sampled, np.argmax(y_train_sampled, axis=1))
            experiment_time = time.time() - experiment_start_time
            y_pred = model.predict(X_val)
            acc, _ = multi_label_balanced_accuracy(y_val, y_pred)
            best_metrics['balanced_accuracy'].append(acc)
            best_metrics['val_loss'].append(0)
            best_metrics['val_f1'].append(
                f1_score(y_val, y_pred, average='macro'))
            best_metrics['val_AUPRC'].append(average_precision_score(sklearn.preprocessing.label_binarize(y_val, classes=np.arange(n_classes)),
                                                                        sklearn.preprocessing.label_binarize(
                y_pred, classes=np.arange(n_classes)),
                average='macro'))
            best_metrics['val_MCC'].append(matthews_corrcoef(y_val, y_pred))
            best_metrics['n_experiment'].append(i)
            best_metrics['training_time'].append(experiment_time)
            
    if wandb.run is not None:
        table_data_df = pd.DataFrame(best_metrics)
        metrics_table = wandb.Table(dataframe=table_data_df)
        wandb.log({"experiment_summary_table": metrics_table})

        wandb.finish()

    with open(f'results/{config["selected-dataset"]}_{config_setup_name.replace("/", "over").replace(".", "point")}.json', 'w') as f:
        json.dump(best_metrics, f, indent=4)
