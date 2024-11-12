from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import json

# Dictionary to store the initialized models with default hyperparameters
non_dl_models = {
    'logistic_regression': LogisticRegression(),
    'random_forest': RandomForestClassifier(),
    'gradient_boosting': GradientBoostingClassifier(),
    'knn': KNeighborsClassifier()
}

# Function to get a non-DL model based on its name
def get_non_dl_model(model_name, params=None):
    if model_name not in non_dl_models:
        raise ValueError(f"Model {model_name} is not recognized. Available models: {list(non_dl_models.keys())}")
    
    # If parameters are provided, create a new instance with the parameters
    if params:
        if model_name == 'logistic_regression':
            return LogisticRegression(**params)
        elif model_name == 'random_forest':
            return RandomForestClassifier(**params)
        elif model_name == 'gradient_boosting':
            return GradientBoostingClassifier(**params)
        elif model_name == 'knn':
            return KNeighborsClassifier(**params)
    else:
        return non_dl_models[model_name]

# Function to register models in a JSON configuration file
def register_non_dl_model_to_json(config_path, model_name):
    with open(config_path, 'r') as file:
        config = json.load(file)
    
    if 'model' not in config:
        config['model'] = {}
    
    config['model']['non_dl_model'] = model_name
    
    with open(config_path, 'w') as file:
        json.dump(config, file, indent=4)