import json
import sklearn.preprocessing
import yaml
from pathlib import Path
import torch
from utils.utils import *
from utils.datasets import *
from models.model_components import *
from models.prototypical_components import *
from models.no_DL_models import *

# TODO: wrap dataset loading in a function... handle different sizes, formats, cols to  drop, etc....

# TODO: wrap preprocessing in a function
config = yaml.safe_load(open("config.yaml"))
if config['prototypical']:
    config_setup_name = 'polyak' if config['polyak'] else 'no-polyak'
    distances_weights = dict(zip(config['setup']['prototypical_config']['weights'], 
                                config['setup']['prototypical_config']['vals']
                            )
                        )
    config_setup_name = config_setup_name + '_' + '_'.join([f"{k}-{v}" for k,v in distances_weights.items()])
    distances_weights = {key: eval(val) for key, val in distances_weights.items()}
else:
    config_setup_name = config['setup']['non_dl_model']

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
                'val_MCC': []}
print(f'results/{config_setup_name.replace("/", "over").replace(".", "point")}.json')
# TODO: add ML flow logging .... for each epoch in experiments log to ML flow, also log to ML flow at the end of each experiment
for i in range(config['setup']['params']['n_experiments']):
    print(f"\n###############################\nStarting experiment {i+1}/{config['setup']['params']['n_experiments']}\n###############################\n")
    X_train_sampled, y_train_sampled = stratified_sample(datasets=[X_train, y_train], 
                                                         n_samples=config['setup']['params']['n_samples'], 
                                                         sample_per_class=config['setup']['params']['sample_per_class'])
    if config['prototypical']:
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
                                            polyak=config['polyak'], polyak_decay=0.999,
                                            best_model_path=f'{config["output"]["best_models"]}{config_setup_name.replace("/", "over").replace(".", "point")}.pth')
        best_metrics['balanced_accuracy'].append(max(experiment_history["balanced_accuracy"]))
        best_metrics['val_loss'].append(max(experiment_history["val_loss"]))
        best_metrics['val_f1'].append(max(experiment_history["val_f1"]))
        best_metrics['val_AUPRC'].append(max(experiment_history["val_AUPRC"]))
        best_metrics['val_MCC'].append(max(experiment_history["val_MCC"]))
        
        with open(f'{train_history_path}/exp_{i}.json', 'w') as f:
            json.dump(experiment_history, f, indent=4)
        
    else:
        model = get_non_dl_model(config['setup']['non_dl_model'])
        model.fit(X_train_sampled, np.argmax(y_train_sampled, axis=1))
        y_pred = model.predict(X_val)
        acc, _ = multi_label_balanced_accuracy(y_val, y_pred)
        best_metrics['balanced_accuracy'].append(acc)
        best_metrics['val_loss'].append(0)
        best_metrics['val_f1'].append(f1_score(y_val, y_pred, average='macro'))
        best_metrics['val_AUPRC'].append(average_precision_score(sklearn.preprocessing.label_binarize(y_val, classes=np.arange(n_classes)), 
                                                                 sklearn.preprocessing.label_binarize(y_pred, classes=np.arange(n_classes)), 
                                                                 average='macro'))
        best_metrics['val_MCC'].append(matthews_corrcoef(y_val, y_pred))
        
with open(f'results/{config_setup_name.replace("/", "over").replace(".", "point")}.json', 'w') as f:
    json.dump(best_metrics, f, indent=4)