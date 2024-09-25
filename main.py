import json
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import *
from models.model_components import *
from models.prototypical_components import *
from config import *

device = device()

# Read data
if dataset == "CIC":
    train_data, validation_data, _ = pd.read_parquet(train_data_path), pd.read_parquet(val_data_path), pd.read_parquet(test_data_path)
    target_column = target_column_CIC
    try:
        train_data.drop(columns_to_drop_CIC, axis=1, inplace=True)
        validation_data.drop(columns_to_drop_CIC, axis=1, inplace=True)
        # test_data.drop(columns_to_drop, axis=1, inplace=True)
    except:
        pass
    # Convert labels to one-hot encoding for multi-label classification
    y_train_data = pd.get_dummies(train_data[target_column])
    y_validation_data = pd.get_dummies(validation_data[target_column])
    # y_test_data = pd.get_dummies(test_data[target_column])

    X_train_data, y_train_data = train_data.drop(target_column, axis=1), y_train_data
    X_validation_data, y_validation_data = validation_data.drop(target_column, axis=1), y_validation_data
    # X_test_data, y_test_data = test_data.drop(target_column, axis=1), y_test_data

    RFC_pipeline = Pipeline([("Scaler", sklearn.preprocessing.RobustScaler()), ("RFC", RandomForestClassifier())])
    RFC_pipeline.fit(X_train_data, y_train_data)
    ftr_importance_RFC = pd.DataFrame(zip(X_train_data.columns, RFC_pipeline["RFC"].feature_importances_))
    ftr_importance_RFC.columns = ["Column", "Weight"]
    ftr_importance_RFC.sort_values("Weight", ascending=False, inplace=True)
    selected_features = ftr_importance_RFC[ftr_importance_RFC.Weight >= ftr_importance_RFC.Weight.mean()].Column.values

    print(f"# RFC Features: {len(selected_features)}")

elif dataset == "EV":
    from sklearn.model_selection import train_test_split

    EV_data = pd.read_csv(data_power_consumption)
    EV_data.drop(columns_to_drop_power_consumption, axis=1, inplace=True)
    state_columns = pd.get_dummies(EV_data['State'], prefix='State')
    EV_data = pd.concat([EV_data, state_columns], axis=1)
    EV_data.drop('State', axis=1, inplace=True)
    train_data, validation_data = train_test_split(EV_data, test_size=0.5)
    X_train_data, y_train_data = train_data.drop(target_column_power_consumption, axis=1), train_data[target_column_power_consumption]
    X_validation_data, y_validation_data = validation_data.drop(target_column_power_consumption, axis=1), validation_data[target_column_power_consumption]
    
    selected_features = X_train_data.columns

    y_train_data = pd.get_dummies(y_train_data)
    y_validation_data = pd.get_dummies(y_validation_data)
# Test RFC
# RFC_pipeline = Pipeline([("Scaler", sklearn.preprocessing.RobustScaler()), ("RFC", RandomForestClassifier(random_state=random_seed))])
# RFC_pipeline.fit(X_train_data[features_RFC], y_train_data)

# y_test_pred = RFC_pipeline.predict(X_test_data[features_RFC])
# print(classification_report(y_test_data, y_test_pred))
# print(f"Accuracy: {multi_label_balanced_accuracy(y_test_data, y_test_pred)}")

scaler = sklearn.preprocessing.RobustScaler()

X_train_data_scaled = scaler.fit_transform(X_train_data[selected_features].values)
X_val_data_scaled = scaler.transform(X_validation_data[selected_features].values)

X_train = torch.tensor(X_train_data_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_data.values, dtype=torch.float32)
n_features = X_train_data_scaled.shape[1]

train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)

X_val = torch.tensor(X_val_data_scaled, dtype=torch.float32)
y_val = torch.tensor(y_validation_data.values, dtype=torch.float32)
val_dataset = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

best_acc_per_experiment = []
for i in range(n_experiments):
    print(f"\n###############################\nStarting experiment {i+1}/{n_experiments}\n###############################\n")
    X_train_sampled, y_train_sampled = stratified_sample(datasets=[X_train, y_train], n_samples=n_samples, sample_per_class=sample_per_class)

    selected_model = MLP_MultiLabel(n_features=n_features, n_classes=y_train_data.shape[1])

    optimizer = torch.optim.Adam(selected_model.parameters(), lr=learning_rate)

    # criterion = nn.BCELoss()

    episodes, unique_indices = create_episodes(X_train_sampled, y_train_sampled, n_episodes, n_support, n_query)
    print(f"Total unique inputs used for training: {len(unique_indices)}")
    
    experiment_history = episodic_training(model=selected_model, optimizer=optimizer, episodes=episodes, val_dataloader=val_dataloader, 
                                                 epochs=n_epochs, alpha=alpha)
    # experiment_history = new_episodic_training_with_polyak(model=selected_model, optimizer=optimizer, episodes=episodes, val_dataloader=val_dataloader, 
    #                                        epochs=n_epochs, weights=distances_weights)
    best_acc_per_experiment.append(max(experiment_history["balanced_accuracy"]))
    print("Episodic training completed.")
    with open(f'{train_history_path}/exp_{i}.json', 'w') as f:
        json.dump(experiment_history, f, indent=4)

with open(f'results/experiments_{model_type}_accuracy.json', 'w') as f:
    json.dump(best_acc_per_experiment, f, indent=4)