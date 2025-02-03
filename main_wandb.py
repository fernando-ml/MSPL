import json
import sklearn.preprocessing
import yaml
from pathlib import Path
import torch
import sys  # Import sys for command line arguments
from utils.utils import *
from utils.datasets import *
from models.model_components import *
from models.prototypical_components import *
from models.no_DL_models import *
import wandb

if len(sys.argv) < 2:
    print("Please provide the path to the YAML config file.")
    sys.exit(1)

yaml_path = sys.argv[1]
config = yaml.safe_load(open(yaml_path))


def get_config_setup_name(scenario):
    weights = ["euclidean", "chebyshev", "cosine", "wasserstein"]
    config_setup_name = 'polyak' if scenario['polyak'] else 'no-polyak'
    distances_weights = dict(zip(weights, scenario['vals']))

    config_setup_name = config_setup_name + '_' + \
        '_'.join([f"{k}-{v}" for k, v in distances_weights.items()])
    distances_weights = {key: eval(val)
                         for key, val in distances_weights.items()}

    return config_setup_name, distances_weights


for scenario in config['scenarios']:

    distances_weights = None

    if config['mode'] == 'prototypical':
        config_setup_name, distances_weights = get_config_setup_name(scenario)
    else:
        config_setup_name = scenario

    # Initialize wandb run for each scenario
    wandb.init(project="Dos-Project",
               name=config_setup_name,
               config={'scenario': scenario})

    print(f"{5 * '#'} Starting scenario {config_setup_name} {5 * '#'}")

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
                    'val_MCC': []
                    }

    print(
        f'results/{config_setup_name.replace("/", "over").replace(".", "point")}.json')

    for i in range(config['params']['n_experiments']):
        print(
            f"\n####################\nStarting experiment {i+1}/{config['params']['n_experiments']}\n#######################\n")

        X_train_sampled, y_train_sampled = stratified_sample(datasets=[X_train, y_train],
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

            experiment_history = multi_space_episodic_training_with_polyak(model=model, optimizer=optimizer,
                                                                           val_dataloader=val_dataloader,
                                                                           episodes=episodes,
                                                                           epochs=config['params']['epochs'],
                                                                           weights=distances_weights,
                                                                           polyak=scenario['polyak'],
                                                                           polyak_decay=0.999,
                                                                           best_model_path=f'{config["output"]["best_models"]}{config_setup_name.replace("/", "over").replace(".", "point")}.pth',
                                                                           wandb_run=wandb
                                                                           )

            best_metrics['balanced_accuracy'].append(
                max(experiment_history["balanced_accuracy"]))
            best_metrics['val_loss'].append(
                min(experiment_history["val_loss"]))
            best_metrics['val_f1'].append(max(experiment_history["val_f1"]))
            best_metrics['val_AUPRC'].append(
                max(experiment_history["val_AUPRC"]))
            best_metrics['val_MCC'].append(max(experiment_history["val_MCC"]))

            with open(f'{train_history_path}/exp_{i}.json', 'w') as f:
                json.dump(experiment_history, f, indent=4)

        else:
            model = get_non_dl_model(scenario)
            model.fit(X_train_sampled, np.argmax(y_train_sampled, axis=1))
            y_pred = model.predict(X_val)
            acc, _ = multi_label_balanced_accuracy(y_val, y_pred)
            f1 = f1_score(y_val, y_pred, average='macro')
            auprc = average_precision_score(sklearn.preprocessing.label_binarize(y_val, classes=np.arange(n_classes)),
                                            sklearn.preprocessing.label_binarize(
                y_pred, classes=np.arange(n_classes)),
                average='macro')
            mcc = matthews_corrcoef(y_val, y_pred)

            best_metrics['balanced_accuracy'].append(acc)
            best_metrics['val_loss'].append(0)
            best_metrics['val_f1'].append(f1)
            best_metrics['val_AUPRC'].append(auprc)
            best_metrics['val_MCC'].append(mcc)

            wandb.log({
                "experiment": i+1,
                "val_balanced_accuracy": acc,
                "val_loss": 0,
                "val_f1": f1,
                "val_AUPRC": auprc,
                "val_MCC": mcc
            })

    wandb.log({
        "best_val_balanced_accuracy_avg": np.mean(best_metrics['balanced_accuracy']),
        "best_val_loss_avg": np.mean(best_metrics['val_loss']),
        "best_val_f1_avg": np.mean(best_metrics['val_f1']),
        "best_val_AUPRC_avg": np.mean(best_metrics['val_AUPRC']),
        "best_val_MCC_avg": np.mean(best_metrics['val_MCC']),
        "best_val_balanced_accuracy_std": np.std(best_metrics['balanced_accuracy']),
        "best_val_loss_std": np.std(best_metrics['val_loss']),
        "best_val_f1_std": np.std(best_metrics['val_f1']),
        "best_val_AUPRC_std": np.std(best_metrics['val_AUPRC']),
        "best_val_MCC_std": np.std(best_metrics['val_MCC']),
    })

    with open(f'results/{config["selected-dataset"]}_{config_setup_name.replace("/", "over").replace(".", "point")}.json', 'w') as f:
        json.dump(best_metrics, f, indent=4)

    wandb.finish()


def multi_space_episodic_training_with_polyak(model, optimizer, val_dataloader, episodes, epochs, weights, polyak, polyak_decay, best_model_path, wandb_run=None):
    history = {
        "val_loss": [],
        "balanced_accuracy": [],
        "val_f1": [],
        "val_AUPRC": [],
        "val_MCC": []
    }
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = episodic_training(model=model,
                                       optimizer=optimizer,
                                       episodes=episodes,
                                       weights=weights,
                                       polyak=polyak,
                                       polyak_decay=polyak_decay)

        val_loss, val_metrics = evaluate_prototypical_model(model=model,
                                                            dataloader=val_dataloader,
                                                            weights=weights)

        history["val_loss"].append(val_loss)
        history["balanced_accuracy"].append(val_metrics["balanced_accuracy"])
        history["val_f1"].append(val_metrics["f1_score"])
        history["val_AUPRC"].append(val_metrics["AUPRC"])
        history["val_MCC"].append(val_metrics["MCC"])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        print(
            f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['balanced_accuracy']:.4f}, Val F1: {val_metrics['f1_score']:.4f}, Val AUPRC: {val_metrics['AUPRC']:.4f}, Val MCC: {val_metrics['MCC']:.4f}")

        if wandb_run:
            wandb_run.log({
                "epoch": epoch+1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_f1": val_metrics["f1_score"],
                "val_AUPRC": val_metrics["AUPRC"],
                "val_MCC": val_metrics["MCC"]
            })

    return history
