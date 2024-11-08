import json
import yaml
from pathlib import Path
import torch
from utils.utils import *
from utils.datasets import *
from models.model_components import *
from models.prototypical_components import *

# TODO: wrap dataset loading in a function... handle different sizes, formats, cols to  drop, etc....

# TODO: wrap preprocessing in a function
config = yaml.safe_load(open("config.yaml"))
config_setup_name = 'polyak' if config['polyak'] else 'no-polyak'
distances_weights = dict(zip(config['setup']['scenario_config']['weights'], 
                                config['setup']['scenario_config']['vals']
                            )
                        )
config_setup_name = config_setup_name + '_' + '_'.join([f"{k}-{v}" for k,v in distances_weights.items()])

dm = DatasetManager(config)
X_train, y_train, val_dataloader = dm.preprocess_data()
n_features, n_classes = int(X_train.shape[1]), int(y_train.shape[1])
train_history_path = config['output']['train-history']
Path(train_history_path).mkdir(parents=True, exist_ok=True)

best_metrics = {'balanced_accuracy': [],
                # 'loss': [],
                'val_loss': [],
                'val_f1': []}

distances_weights = {key: eval(val) for key, val in distances_weights.items()}
# TODO: add ML flow logging .... for each epoch in experiments log to ML flow, also log to ML flow at the end of each experiment
for i in range(config['setup']['params']['n_experiments']):
    print(f"\n###############################\nStarting experiment {i+1}/{config['setup']['params']['n_experiments']}\n###############################\n")
    X_train_sampled, y_train_sampled = stratified_sample(datasets=[X_train, y_train], 
                                                         n_samples=config['setup']['params']['n_samples'], 
                                                         sample_per_class=config['setup']['params']['sample_per_class'])

    model = MLP_MultiLabel(n_features=n_features, n_classes=n_classes)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['setup']['params']['lr'])

    episodes, unique_indices = create_episodes(X_train_sampled, y_train_sampled, 
            n_episodes=config['setup']['params']['n_episodes'], 
            n_support=config['setup']['params']['n_support'], 
            n_query=config['setup']['params']['n_query'])
    print(f"Total unique inputs used for training: {len(unique_indices)}")

    experiment_history = multi_space_episodic_training_with_polyak(model=model, optimizer=optimizer, 
                                                                   val_dataloader=val_dataloader,
                                                                    episodes=episodes, epochs=config['setup']['params']['epochs'], 
                                                                    weights=distances_weights, 
                                        polyak=config['polyak'], polyak_decay=0.999)
    
    best_metrics['balanced_accuracy'].append(max(experiment_history["balanced_accuracy"]))
    best_metrics['val_loss'].append(max(experiment_history["val_loss"]))
    best_metrics['val_f1'].append(max(experiment_history["val_f1"]))
    
    with open(f'{train_history_path}/exp_{i}.json', 'w') as f:
        json.dump(experiment_history, f, indent=4)

with open(f'results/{config_setup_name.replace("/", "over").replace(".", "point")}.json', 'w') as f:
    json.dump(best_metrics, f, indent=4)