"""
Dual-Space (DOS) Project - Data Loading Module

This module provides data loading and preprocessing functionality for different datasets
used in the DOS project, supporting both prototypical networks and traditional ML approaches.

NOTE: This version incorporates the preprocessing logic (text handling, one-hot encoding)
from the original datasets.py's DatasetManager.
"""

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from utils.utils import stratified_sample, get_device
import yaml
import sklearn  # Added import

# CustomDataset remains useful for the Prototypical Network training structure


class CustomDataset(Dataset):
    """Custom PyTorch Dataset for prototypical network training."""

    def __init__(self, features, labels):
        """
        Initialize CustomDataset.

        Args:
            features (torch.Tensor): Input features
            labels (torch.Tensor): Target labels (expected to be one-hot encoded)
        """
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class DatasetManager:
    """
    Manages dataset loading, preprocessing, and preparation for model training.

    This class handles loading datasets from parquet files, preprocessing them
    (including text feature handling and one-hot encoding of targets),
    and creating appropriate data structures.
    """

    def __init__(self, config):
        """
        Initialize the DatasetManager.

        Args:
            config (dict): Configuration dictionary containing dataset parameters
        """
        self.config = config
        self.device = get_device()
        self.dataset_name = config['selected-dataset']
        self.datasets_config = self._load_datasets_config()
        self.current_dataset_config = self._get_current_dataset_config()
        self.n_features = None  # Will be set after preprocessing
        self.scaler = StandardScaler()  # Initialize scaler

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
        normalized_dataset_name = self.dataset_name.replace(
            '-', '').replace(' ', '').lower()
        for dataset_config in self.datasets_config:
            normalized_config_name = dataset_config['name'].replace(
                '-', '').replace(' ', '').lower()
            if normalized_config_name == normalized_dataset_name:
                return dataset_config
        raise ValueError(
            f"Dataset '{self.dataset_name}' not found in configuration")

    def preprocess_data(self):
        """
        Load and preprocess the selected dataset according to the logic
        similar to the original datasets.py.

        Returns:
            tuple: (X_train, y_train, val_dataloader) preprocessed data.
                   y_train is one-hot encoded.
        """
        print(f"Loading dataset: {self.dataset_name}")
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
            print("Splitting data into train/validation (80/20 split, stratified)")
            train_df, val_df = train_test_split(df, test_size=0.2, random_state=42,
                                                stratify=df[target_column])
            train_val_split_done = False  # Scaler needs fitting only on train

        # --- Preprocessing Steps (derived from datasets.py) ---

        # 1. Drop specified columns
        print(f"Dropping specified columns: {columns_to_drop}")
        # Ensure columns exist before dropping
        train_cols_to_drop = [
            col for col in columns_to_drop if col in train_df.columns]
        val_cols_to_drop = [
            col for col in columns_to_drop if col in val_df.columns]
        train_df = train_df.drop(columns=train_cols_to_drop, errors='ignore')
        val_df = val_df.drop(columns=val_cols_to_drop, errors='ignore')

        # 2. Separate target variable
        print(f"Separating target column: {target_column}")
        y_train_series = train_df[target_column]
        y_val_series = val_df[target_column]

        # Drop target column from feature sets
        X_train_df = train_df.drop(columns=[target_column])
        X_val_df = val_df.drop(columns=[target_column])

        # 3. Handle Text Columns using One-Hot Encoding
        print("Processing text columns (if any) using one-hot encoding...")
        text_cols = X_train_df.select_dtypes(include='object').columns
        if len(text_cols) > 0:
            print(f"Found text columns: {list(text_cols)}")
            # Combine train and validation text columns to ensure consistent encoding
            combined_text_X = pd.concat(
                [X_train_df[text_cols], X_val_df[text_cols]], axis=0)
            combined_text_X_dummies = pd.get_dummies(
                combined_text_X, prefix='text', drop_first=True)  # Match datasets.py

            # Split back into train and validation dummy sets
            X_train_text_dummies = combined_text_X_dummies.iloc[:X_train_df.shape[0], :]
            X_val_text_dummies = combined_text_X_dummies.iloc[X_train_df.shape[0]:, :]

            # Drop original text columns
            X_train_df = X_train_df.drop(columns=text_cols)
            X_val_df = X_val_df.drop(columns=text_cols)

            # Align columns after potential dummy creation differences (important!)
            # Get columns present in both after dummy creation
            common_dummy_cols = X_train_text_dummies.columns.intersection(
                X_val_text_dummies.columns)
            X_train_text_dummies = X_train_text_dummies[common_dummy_cols]
            X_val_text_dummies = X_val_text_dummies[common_dummy_cols]

            # Add missing columns created during get_dummies (fill with 0)
            missing_cols_val = list(
                set(X_train_text_dummies.columns) - set(X_val_text_dummies.columns))
            for col in missing_cols_val:
                X_val_text_dummies[col] = 0
            # Ensure same order
            X_val_text_dummies = X_val_text_dummies[X_train_text_dummies.columns]

            missing_cols_train = list(
                set(X_val_text_dummies.columns) - set(X_train_text_dummies.columns))
            for col in missing_cols_train:
                X_train_text_dummies[col] = 0
            # Ensure same order
            X_train_text_dummies = X_train_text_dummies[X_val_text_dummies.columns]

            # Concatenate numeric and dummy features
            X_train_df = pd.concat([X_train_df, X_train_text_dummies], axis=1)
            X_val_df = pd.concat([X_val_df, X_val_text_dummies], axis=1)
        else:
            print("No text columns found.")

        # Ensure columns are exactly the same in train and val features after processing
        train_cols = set(X_train_df.columns)
        val_cols = set(X_val_df.columns)

        if train_cols != val_cols:
            print("Warning: Train and Validation columns differ after processing!")
            print("Train only:", train_cols - val_cols)
            print("Val only:", val_cols - train_cols)
            # Attempt to align - keep only common columns
            common_cols = list(train_cols.intersection(val_cols))
            print(f"Aligning to {len(common_cols)} common columns.")
            X_train_df = X_train_df[common_cols]
            X_val_df = X_val_df[common_cols]

        # 4. Scale Numerical Features
        print("Scaling features using StandardScaler...")
        # Fit scaler ONLY on training data
        self.scaler.fit(X_train_df.values)
        X_train_scaled = self.scaler.transform(X_train_df.values)
        X_val_scaled = self.scaler.transform(X_val_df.values)

        # Set number of features
        self.n_features = X_train_scaled.shape[1]
        print(f"Number of features after preprocessing: {self.n_features}")

        # 5. One-Hot Encode Target Variable
        print("One-hot encoding target variable...")
        # Use pd.get_dummies on the combined series to ensure all classes are represented
        combined_y = pd.concat([y_train_series, y_val_series], axis=0)
        y_dummies = pd.get_dummies(combined_y)

        # Split back into train/validation target dummies
        y_train_dummies = y_dummies.iloc[:len(y_train_series)]
        y_val_dummies = y_dummies.iloc[len(y_train_series):]

        # 6. Convert to PyTorch Tensors
        print("Converting data to PyTorch tensors...")
        X_train = torch.tensor(
            X_train_scaled, dtype=torch.float32).to(self.device)
        # One-hot encoded labels should be float32 for many loss functions (like BCEWithLogitsLoss)
        # or long if using CrossEntropyLoss with integer targets.
        # datasets.py used float32, so we follow that. Check if your loss function requires this.
        y_train = torch.tensor(y_train_dummies.values,
                               dtype=torch.float32).to(self.device)

        X_val = torch.tensor(X_val_scaled, dtype=torch.float32).to(self.device)
        y_val = torch.tensor(y_val_dummies.values,
                             dtype=torch.float32).to(self.device)

        print(f"Original train shape: {X_train.shape}, {y_train.shape}")

        # 7. Apply Stratified Sampling (Optional, based on config)
        if 'n_samples' in self.config['params'] and 'sample_per_class' in self.config['params']:
            n_samples = self.config['params']['n_samples']
            sample_per_class = self.config['params']['sample_per_class']

            # Need integer labels for stratified sampling function
            # Convert one-hot y_train back to class indices temporarily
            y_train_indices = torch.argmax(y_train, dim=1)

            print(
                f"Applying stratified sampling: {n_samples} samples, {sample_per_class} per class")
            # Pass index labels to stratified_sample
            X_train_sampled, y_train_indices_sampled = stratified_sample((X_train, y_train_indices),
                                                                         n_samples,
                                                                         sample_per_class)

            # Convert sampled indices back to one-hot encoding if needed, or keep indices
            # For prototypical networks, integer indices are usually preferred.
            # Let's assume the trainer expects integer labels for sampling's output
            # *** Adjust this if your trainer specifically needs one-hot labels AFTER sampling ***
            X_train = X_train_sampled
            # IMPORTANT: If trainer needs one-hot, convert y_train_indices_sampled back here.
            # Otherwise, return the integer labels from sampling.
            # Let's return integer labels for compatibility with typical proto-net inputs.
            # Revert y_train to integer format after sampling
            y_train = y_train_indices_sampled.to(
                self.device)  # Keep on correct device
            print(
                f"Sampled train shape: {X_train.shape}, {y_train.shape} (Integer Labels)")
            # Note: The traceback error occurred *before* sampling in the original flow.
            # The one-hot encoding fixed the `y.shape[1]` issue *if* the trainer was correctly expecting that.
            # If the trainer *actually* expects integer labels (common for CrossEntropyLoss/NLLLoss),
            # then the original dataloader.py label format was correct, and the bug was in the trainer's
            # `create_episodes` function assuming `y.shape[1]`.
            # Given we're merging datasets.py logic, we initially create one-hot, but sampling
            # usually works better with/outputs integer labels. We return integer labels post-sampling.

        # 8. Create Validation DataLoader
        val_batch_size = self.config['params'].get(
            'val-batch-size', 512)  # Use configured bs
        # Use TensorDataset for validation loader as features/labels are tensors now
        # Make sure y_val matches the format expected by evaluation (one-hot or integer)
        # Using one-hot y_val here as created above.
        val_dataset = TensorDataset(X_val, y_val)
        val_dataloader = DataLoader(
            val_dataset, batch_size=val_batch_size, shuffle=False)

        print(f"Data preprocessing complete.")
        print(
            f"Final X_train shape: {X_train.shape}, y_train shape: {y_train.shape} (Note label format change if sampled)")
        print(
            f"Validation X_val shape: {X_val.shape}, y_val shape: {y_val.shape} (One-Hot Labels)")

        # Return training tensors and validation dataloader
        # The train dataloader is created separately using get_train_dataloader
        return X_train, y_train, val_dataloader

    def get_train_dataloader(self, X_train, y_train):
        """
        Create a DataLoader for training data using CustomDataset.

        Args:
            X_train (torch.Tensor): Training features
            y_train (torch.Tensor): Training labels (format depends on whether sampling was applied)

        Returns:
            torch.utils.data.DataLoader: DataLoader for training
        """
        train_dataset = CustomDataset(
            X_train, y_train)  # CustomDataset handles tensor inputs
        train_batch_size = self.config['params'].get('train-batch-size', 32)
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=train_batch_size,
                                      shuffle=True)
        print(f"Train DataLoader created with batch size {train_batch_size}")
        return train_dataloader

    def get_n_features(self):
        """Returns the number of features after preprocessing."""
        if self.n_features is None:
            print("Warning: n_features not set. Call preprocess_data first.")
            # Optionally, run preprocessing if not done? Or raise error?
            # self.preprocess_data() # Be careful about side effects
            return None
        return self.n_features
