"""
Dual-Space (DOS) Project - Data Loading Module

This module provides data loading and preprocessing functionality for different datasets
used in the DOS project, supporting both prototypical networks and traditional ML approaches.
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


class CustomDataset(Dataset):
    """Custom PyTorch Dataset for prototypical network training."""
    
    def __init__(self, features, labels):
        """
        Initialize CustomDataset.
        
        Args:
            features (torch.Tensor): Input features
            labels (torch.Tensor): Target labels
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
    
    This class handles loading datasets from parquet files, preprocessing them,
    and creating appropriate data structures for both prototypical networks and
    traditional machine learning approaches.
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
        
    def _load_datasets_config(self):
        """
        Load dataset configurations from YAML file.
        
        Returns:
            list: List of dataset configurations
        """
        with open('utils/datasets_config.yaml', 'r') as f:
            return yaml.safe_load(f)
    
    def _get_current_dataset_config(self):
        """
        Get configuration for currently selected dataset.
        
        Returns:
            dict: Configuration for current dataset
            
        Raises:
            ValueError: If selected dataset is not found in configuration
        """
        # Normalize dataset names to make matching more flexible
        # Remove dashes, spaces, and convert to lowercase for comparison
        normalized_dataset_name = self.dataset_name.replace('-', '').replace(' ', '').lower()
        
        for dataset_config in self.datasets_config:
            # Normalize the config dataset name the same way
            normalized_config_name = dataset_config['name'].replace('-', '').replace(' ', '').lower()
            
            if normalized_config_name == normalized_dataset_name:
                return dataset_config
        
        raise ValueError(f"Dataset '{self.dataset_name}' not found in configuration")
    
    def preprocess_data(self):
        """
        Load and preprocess the selected dataset.
        
        Returns:
            tuple: (X_train, y_train, val_dataloader) preprocessed data
        """
        print(f"Loading dataset: {self.dataset_name}")
        
        # Load data based on dataset type
        if 'path_train' in self.current_dataset_config and 'path_val' in self.current_dataset_config:
            # Separate train and validation files
            X_train, y_train = self._load_separate_train_val_files(is_train=True)
            X_val, y_val = self._load_separate_train_val_files(is_train=False)
        else:
            # Single file with train/val split
            X_train, y_train, X_val, y_val = self._load_single_file()
        
        # Apply sampling if configured
        if 'n_samples' in self.config['params'] and 'sample_per_class' in self.config['params']:
            n_samples = self.config['params']['n_samples']
            sample_per_class = self.config['params']['sample_per_class']
            
            print(f"Applying stratified sampling: {n_samples} samples, {sample_per_class} per class")
            X_train, y_train = stratified_sample((X_train, y_train), 
                                                n_samples, 
                                                sample_per_class)
        
        # Create validation dataloader
        val_dataset = TensorDataset(X_val, y_val)
        val_batch_size = self.config['params'].get('val-batch-size', 512)
        val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)
        
        print(f"Data preprocessing complete - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        return X_train, y_train, val_dataloader
    
    def _load_separate_train_val_files(self, is_train=True):
        """
        Load data from separate train and validation files.
        
        Args:
            is_train (bool): If True, load training data, else validation data
            
        Returns:
            tuple: (X, y) data tensors
        """
        file_key = 'path_train' if is_train else 'path_val'
        file_path = self.current_dataset_config[file_key]
        
        print(f"Loading data from: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Drop specified columns if any
        if 'columns-to-drop' in self.current_dataset_config:
            columns_to_drop = self.current_dataset_config['columns-to-drop']
            # Only drop columns that actually exist in the dataframe
            columns_to_drop = [col for col in columns_to_drop if col in df.columns]
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Get target column
        target_column = self.current_dataset_config['target_column']
        
        # Convert to PyTorch tensors
        X, y = self._prepare_tensors(df, target_column)
        
        return X, y
    
    def _load_single_file(self):
        """
        Load data from a single file and split into train/validation sets.
        
        Returns:
            tuple: (X_train, y_train, X_val, y_val) data tensors
        """
        file_path = self.current_dataset_config['path']
        print(f"Loading data from: {file_path}")
        df = pd.read_parquet(file_path)
        
        # Drop specified columns if any
        if 'columns-to-drop' in self.current_dataset_config:
            columns_to_drop = self.current_dataset_config['columns-to-drop']
            # Only drop columns that actually exist in the dataframe
            columns_to_drop = [col for col in columns_to_drop if col in df.columns]
            df = df.drop(columns=columns_to_drop, errors='ignore')
        
        # Get target column
        target_column = self.current_dataset_config['target_column']
        
        # Split data
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, 
                                          stratify=df[target_column])
        
        # Convert to PyTorch tensors
        X_train, y_train = self._prepare_tensors(train_df, target_column)
        X_val, y_val = self._prepare_tensors(val_df, target_column)
        
        return X_train, y_train, X_val, y_val
    
    def _prepare_tensors(self, df, target_column):
        """
        Prepare features and labels as PyTorch tensors.
        
        Args:
            df (pd.DataFrame): DataFrame containing data
            target_column (str): Name of the target column
            
        Returns:
            tuple: (X, y) PyTorch tensors
        """
        # Extract features and targets
        y = df[target_column].values
        X = df.drop(columns=[target_column]).values
        
        # Scale features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Convert to PyTorch tensors
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Handle different label formats
        if isinstance(y[0], str):
            # Convert string labels to integers
            unique_labels = sorted(set(y))
            label_map = {label: i for i, label in enumerate(unique_labels)}
            y = np.array([label_map[label] for label in y])
        
        y = torch.tensor(y, dtype=torch.long).to(self.device)
        
        return X, y
    
    def get_train_dataloader(self, X_train, y_train):
        """
        Create a DataLoader for training data.
        
        Args:
            X_train (torch.Tensor): Training features
            y_train (torch.Tensor): Training labels
            
        Returns:
            torch.utils.data.DataLoader: DataLoader for training
        """
        train_dataset = CustomDataset(X_train, y_train)
        train_batch_size = self.config['params'].get('train-batch-size', 32)
        train_dataloader = DataLoader(train_dataset, 
                                     batch_size=train_batch_size, 
                                     shuffle=True)
        return train_dataloader
