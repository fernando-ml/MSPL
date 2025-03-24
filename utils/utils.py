"""
Utility functions for the DOS (Dual-Space) project.

This module contains common utilities for data processing, device selection,
and other helper functions used throughout the project.
"""

import torch
import numpy as np
import sys

def get_device():   
    """ 
    Returns the device to be used for PyTorch tensor operations.
    
    Checks in order:
    1. Metal Performance Shader (MPS)
    2. CUDA
    3. CPU

    Returns:
        str: The device to use ("mps", "cuda", or "cpu")
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

def stratified_sample(datasets, n_samples, sample_per_class, random_state=None):
    """
    Perform stratified sampling on the given datasets.

    This function ensures that each class is represented by at least 'sample_per_class' instances,
    and then randomly samples the remaining instances to reach the total 'n_samples'.

    Args:
        datasets (tuple): A tuple containing features (X) and labels (y).
        n_samples (int): The total number of samples to return.
        sample_per_class (int): The minimum number of samples to include for each class.
        random_state (int, optional): Seed for random number generation. Defaults to None.

    Returns:
        tuple: A tuple containing the sampled features (X_sampled) and labels (y_sampled).

    Raises:
        ValueError: If n_samples is less than the product of unique labels and sample_per_class.
    """
    X, y = datasets
    
    if random_state is not None:
        torch.manual_seed(random_state)
        np.random.seed(random_state)
    
    # Get the unique labels
    if len(y.shape) > 1:  # One-hot encoded
        n_classes = y.shape[1]
        indices_per_class = []
        
        for class_idx in range(n_classes):
            # Get indices where this class is present
            class_indices = torch.where(y[:, class_idx] == 1)[0]
            if len(class_indices) > 0:
                # Sample from available indices
                n_to_sample = min(sample_per_class, len(class_indices))
                sampled_indices = class_indices[torch.randperm(len(class_indices))[:n_to_sample]]
                indices_per_class.extend(sampled_indices.tolist())
    else:  # Single label
        unique_labels = torch.unique(y)
        indices_per_class = []
        
        for label in unique_labels:
            label_indices = torch.where(y == label)[0]
            if len(label_indices) > 0:
                n_to_sample = min(sample_per_class, len(label_indices))
                sampled_indices = label_indices[torch.randperm(len(label_indices))[:n_to_sample]]
                indices_per_class.extend(sampled_indices.tolist())
    
    # Remove duplicates and convert to tensor
    indices_per_class = torch.tensor(list(set(indices_per_class)))
    
    # Calculate the remaining samples needed
    remaining_samples = n_samples - len(indices_per_class)
    
    if remaining_samples < 0:
        # We already have more than n_samples due to class balancing
        # Just select a random subset
        indices_per_class = indices_per_class[torch.randperm(len(indices_per_class))[:n_samples]]
    else:
        # Get all indices that were not sampled initially
        remaining_indices = list(set(range(len(X))) - set(indices_per_class.tolist()))
        
        if len(remaining_indices) > 0:
            # Randomly sample the remaining indices
            remaining_indices_sample = np.random.choice(
                remaining_indices, 
                min(remaining_samples, len(remaining_indices)), 
                replace=False
            )
            
            # Combine the initial and random samples indices
            final_indices = torch.cat([indices_per_class, torch.tensor(remaining_indices_sample)])
        else:
            # If no remaining indices, just use what we have
            final_indices = indices_per_class
            
        # Shuffle the final indices
        final_indices = final_indices[torch.randperm(len(final_indices))]
    
    # Sample the data using the final indices
    X_sampled = X[final_indices, :]
    y_sampled = y[final_indices]
    
    return X_sampled, y_sampled

def dataloader_to_numpy(dataloader):
    """
    Convert a PyTorch DataLoader to NumPy arrays.
    
    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader to convert
        
    Returns:
        tuple: (features, labels) as NumPy arrays
    """
    all_features = []
    all_labels = []
    
    for features, labels in dataloader:
        all_features.append(features.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    
    return np.vstack(all_features), np.vstack(all_labels)

def print_section(title, char='#', length=40):
    """
    Print a section header with decoration.
    
    Args:
        title (str): The title text
        char (str): Character for decoration
        length (int): Length of the decoration line
    """
    print(f"\n{char * length}")
    print(f"{title}")
    print(f"{char * length}\n")
