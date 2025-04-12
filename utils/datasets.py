import os
import sys
import numpy as np
import pandas as pd
import yaml
import sklearn
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import TensorDataset, DataLoader


class DatasetManager:
    def __init__(self, config):
        self.selected_dataset = config['selected-dataset']
        self.val_batch_size = config['params']['val-batch-size']
        self.datasets_config = self._load_datasets_config()
        self.current_dataset_config = self._get_current_dataset_config()
        self.loaded_dataset = None
        self.target_column = None

        self.get_data()

    def _load_datasets_config(self):
        """Load dataset configurations from YAML file."""
        # Assuming the yaml file is in utils relative to the project root
        # Adjust the path if necessary based on where main.py is run from
        config_path = os.path.join(os.path.dirname(
            __file__), '..', 'utils', 'datasets_config.yaml')
        if not os.path.exists(config_path):
            # Fallback if running from project root
            config_path = 'utils/datasets_config.yaml'
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Could not find datasets_config.yaml at expected locations.")

        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def _get_current_dataset_config(self):
        """Get configuration for the currently selected dataset."""
        normalized_dataset_name = self.selected_dataset.replace(
            '-', '').replace(' ', '').lower()
        for dataset_config in self.datasets_config:
            normalized_config_name = dataset_config['name'].replace(
                '-', '').replace(' ', '').lower()
            if normalized_config_name == normalized_dataset_name:
                return dataset_config
        raise ValueError(
            f"Dataset '{self.selected_dataset}' not found in configuration")

    def get_data(self):
        self.datasets = yaml.safe_load(open("utils/datasets_config.yaml"))
        for dataset in self.datasets:
            if dataset['name'] == self.selected_dataset:
                if len(dataset.keys()) > 4:
                    self.loaded_dataset = pd.read_parquet(
                        dataset['path_train']), pd.read_parquet(dataset['path_val'])
                    break
                else:
                    self.loaded_dataset = pd.read_parquet(dataset['path'])
                    break
        self.target_column = dataset['target_column']
        if dataset.get('columns-to-drop') is not None:
            self.columns_to_drop = dataset['columns-to-drop']
        else:
            self.columns_to_drop = []

    def preprocess_data(self):

        print(f"Loading dataset: {self.selected_dataset}")
        target_column = self.current_dataset_config['target_column']
        columns_to_drop = self.current_dataset_config.get(
            'columns-to-drop', [])

        # Load data
        if 'path_train' in self.current_dataset_config and 'path_val' in self.current_dataset_config:
            print(f"Loading separate train/val files:")
            print(f"  Train: {self.current_dataset_config['path_train']}")
            print(f"  Val:   {self.current_dataset_config['path_val']}")
            train_df = pd.read_parquet(
                self.current_dataset_config['path_train'])
            val_df = pd.read_parquet(self.current_dataset_config['path_val'])
            train_val_split_done = True
        else:
            print(
                f"Loading single file: {self.current_dataset_config['path']}")
            df = pd.read_parquet(self.current_dataset_config['path'])
            # Split data if loaded from a single file
            print("Splitting data into train/validation (50/50 split, stratified)")
            train_df, val_df = train_test_split(df, test_size=0.5, random_state=42,
                                                stratify=df[target_column])

        # Drop unnecessary columns - specificied in datasets_config.yaml
        train_df.drop(self.columns_to_drop, axis=1, inplace=True)
        val_df.drop(self.columns_to_drop, axis=1, inplace=True)

        # Separate target column
        y_train_data = pd.get_dummies(train_df[self.target_column])
        y_validation_data = pd.get_dummies(val_df[self.target_column])

        # Drop target column from feature sets
        X_train_data = train_df.drop(self.target_column, axis=1)
        X_validation_data = val_df.drop(self.target_column, axis=1)

        # Select text columns
        text_cols = X_train_data.select_dtypes(include='object').columns

        if len(text_cols) > 0:
            # Combine train and validation data to ensure consistent dummy variable creation
            combined_X = pd.concat(
                [X_train_data[text_cols], X_validation_data[text_cols]], axis=0)

            # Create dummy variables for text columns across both train and validation sets
            combined_X_dummies = pd.get_dummies(
                combined_X, prefix='text', drop_first=True)

            # Split back into train and validation sets
            X_train_text_cols = combined_X_dummies.iloc[:X_train_data.shape[0], :]
            X_validation_text_cols = combined_X_dummies.iloc[X_train_data.shape[0]:, :]

            # Drop original text columns from X_train and X_validation
            X_train_data.drop(text_cols, axis=1, inplace=True)
            X_validation_data.drop(text_cols, axis=1, inplace=True)

            # Concatenate the dummy variables back to the original feature sets
            X_train_data = pd.concat([X_train_data, X_train_text_cols], axis=1)
            X_validation_data = pd.concat(
                [X_validation_data, X_validation_text_cols], axis=1)

        scaler = sklearn.preprocessing.StandardScaler()

        X_train_data_scaled = scaler.fit_transform(X_train_data.values)
        X_val_data_scaled = scaler.transform(X_validation_data.values)

        # Convert scaled data to PyTorch tensors
        X_train = torch.tensor(X_train_data_scaled, dtype=torch.float32)
        y_train = torch.tensor(y_train_data.values, dtype=torch.float32)

        # Set number of features based on scaled data
        self.n_features = X_train.shape[1]

        X_val = torch.tensor(X_val_data_scaled, dtype=torch.float32)
        y_val = torch.tensor(y_validation_data.values, dtype=torch.float32)

        # Create a DataLoader for the validation set
        val_dataset = TensorDataset(X_val, y_val)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=self.val_batch_size,
                                    shuffle=False)

        return X_train, y_train, val_dataloader

    def get_n_features(self):
        return self.n_features


def dataloader_to_numpy(dataloader):
    no_dl_data, no_dl_labels = [], []
    for batch_data, batch_labels in dataloader:
        no_dl_data.append(batch_data)
        no_dl_labels.append(batch_labels)
    all_data = torch.cat(no_dl_data, dim=0)
    all_labels = torch.cat(no_dl_labels, dim=0)
    X_val = all_data.numpy()
    y_val = all_labels.numpy()
    y_val = np.argmax(y_val, axis=1)
    return X_val, y_val


def read_csv_files_in_folder(folder_path):
    """
    Reads CSV files from a folder and concatenates them into a single pandas DataFrame.

    Parameters:
    folder_path (str): Path to the folder containing the CSV files

    Returns:
    pandas.DataFrame: Concatenated DataFrame from all CSV files in the folder
    """
    # Get a list of all CSV files in the folder
    csv_files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Read each CSV file and append to a list of DataFrames
    dfs = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        dfs.append(df)

    # Concatenate the DataFrames into a single DataFrame
    combined_df = pd.concat(dfs, ignore_index=True)

    return combined_df
