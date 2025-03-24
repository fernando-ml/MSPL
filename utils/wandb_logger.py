"""
Weights & Biases (wandb) Logger for the DOS project.

This module provides functions to initialize wandb, log metrics,
and visualize experiment results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
from typing import Dict, List, Union, Tuple, Any, Optional


def init_wandb(config: Dict, 
              run_name: str, 
              project_name: str = "DOS-Project",
              experiment_type: str = "prototypical") -> wandb.sdk.wandb_run.Run:
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
        name=run_name,
        config=wandb_config,
        reinit=True,  # Allow reinitializing if needed
    )
    
    return run


def log_metrics(metrics: Dict[str, Union[float, List[float]]], 
                step: Optional[int] = None, 
                prefix: str = "") -> None:
    """
    Log metrics to wandb.
    
    Args:
        metrics (Dict): Dictionary of metrics to log
        step (int, optional): Step number for this set of metrics
        prefix (str, optional): Prefix to add to metric names
    """
    # Process metrics to ensure they're in the right format for wandb
    log_dict = {}
    
    for key, value in metrics.items():
        # If we have a list or array, we may want to log mean, std, min, max
        if isinstance(value, (list, np.ndarray)) and len(value) > 1:
            value_array = np.array(value)
            log_dict[f"{prefix}{key}/mean"] = float(np.mean(value_array))
            log_dict[f"{prefix}{key}/std"] = float(np.std(value_array))
            log_dict[f"{prefix}{key}/min"] = float(np.min(value_array))
            log_dict[f"{prefix}{key}/max"] = float(np.max(value_array))
        else:
            # If it's a single value or empty, just log it as is
            if isinstance(value, (list, np.ndarray)) and len(value) == 1:
                value = value[0]
            log_dict[f"{prefix}{key}"] = value
    
    # Log to wandb
    wandb.log(log_dict, step=step)


def log_confusion_matrix(y_true: np.ndarray, 
                         y_pred: np.ndarray, 
                         class_names: Optional[List[str]] = None,
                         title: str = "Confusion Matrix") -> None:
    """
    Log confusion matrix visualization to wandb.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        class_names (List[str], optional): List of class names
        title (str): Title for the plot
    """
    # For multi-label, we need to handle differently
    if len(y_true.shape) > 1 and y_true.shape[1] > 1:
        # This is multi-label, log each class individually
        for i in range(y_true.shape[1]):
            class_name = class_names[i] if class_names else f"Class {i}"
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            # Create a figure and axes
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.title(f"{title} - {class_name}")
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            # Log to wandb
            wandb.log({f"confusion_matrix/{class_name}": wandb.Image(fig)})
            plt.close(fig)
    else:
        # Single-label classification
        cm = confusion_matrix(y_true, y_pred)
        
        # Create a figure and axes
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Use class names if provided
        if class_names:
            tick_marks = np.arange(len(class_names))
            plt.xticks(tick_marks + 0.5, class_names, rotation=45, ha='right')
            plt.yticks(tick_marks + 0.5, class_names, rotation=0)
        
        # Log to wandb
        wandb.log({"confusion_matrix": wandb.Image(fig)})
        plt.close(fig)


def log_per_class_metrics(metrics_dict: Dict[str, np.ndarray], 
                         class_names: Optional[List[str]] = None,
                         prefix: str = "") -> None:
    """
    Log per-class metrics to wandb.
    
    Args:
        metrics_dict (Dict): Dictionary of per-class metrics
        class_names (List[str], optional): List of class names
        prefix (str, optional): Prefix to add to metric names
    """
    # Convert metrics to a DataFrame for easier visualization
    metrics_df = pd.DataFrame(metrics_dict)
    
    if class_names:
        metrics_df.index = class_names
    
    # Create a table for wandb
    table = wandb.Table(dataframe=metrics_df.reset_index().rename(columns={'index': 'class'}))
    
    # Log the table
    wandb.log({f"{prefix}per_class_metrics": table})
    
    # Also log a heatmap visualization
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(metrics_df, annot=True, cmap='viridis', ax=ax)
    plt.title(f"{prefix}Per-Class Metrics")
    plt.tight_layout()
    
    wandb.log({f"{prefix}per_class_metrics_heatmap": wandb.Image(fig)})
    plt.close(fig)


def log_embeddings(features: np.ndarray, 
                  labels: np.ndarray,
                  method: str = 'tsne',
                  class_names: Optional[List[str]] = None,
                  name: str = "embeddings") -> None:
    """
    Log embeddings visualization using t-SNE or PCA.
    
    Args:
        features (np.ndarray): Feature vectors
        labels (np.ndarray): Labels for each feature vector
        method (str): Dimensionality reduction method ('tsne' or 'pca')
        class_names (List[str], optional): List of class names
        name (str): Name for the plot
    """
    # Convert labels to 1D if needed
    if len(labels.shape) > 1:
        # If multi-label, use argmax to get primary class
        labels = np.argmax(labels, axis=1)
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
    else:  # PCA as default fallback
        reducer = PCA(n_components=2, random_state=42)
    
    # Reduce dimensionality
    reduced_features = reducer.fit_transform(features)
    
    # Create DataFrame for plotting
    df = pd.DataFrame(reduced_features, columns=['x', 'y'])
    df['label'] = labels
    
    if class_names:
        df['class'] = df['label'].apply(lambda x: class_names[x] if x < len(class_names) else f"Class {x}")
    else:
        df['class'] = df['label'].apply(lambda x: f"Class {x}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for label, group in df.groupby('label'):
        class_name = class_names[label] if class_names and label < len(class_names) else f"Class {label}"
        ax.scatter(group['x'], group['y'], label=class_name, alpha=0.8)
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_title(f"{method.upper()} Visualization of {name}")
    plt.tight_layout()
    
    # Log to wandb
    wandb.log({f"{name}_{method}": wandb.Image(fig)})
    plt.close(fig)


def log_distance_distributions(prototypes: torch.Tensor, 
                              query_embeddings: torch.Tensor,
                              query_labels: torch.Tensor,
                              distance_names: List[str]) -> None:
    """
    Log distributions of distances between query embeddings and prototypes.
    
    Args:
        prototypes (torch.Tensor): Class prototypes
        query_embeddings (torch.Tensor): Query embeddings
        query_labels (torch.Tensor): Query labels
        distance_names (List[str]): Names of distance metrics
    """
    for distance_name in distance_names:
        # Create a figure for this distance metric
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Get distances from query embeddings to prototypes
        # Note: This is pseudocode - actual distance calculation will depend on your implementation
        distances = calculate_distances(query_embeddings, prototypes, distance_name)
        
        # Get distances to correct and incorrect prototypes
        same_class_distances = []
        diff_class_distances = []
        
        for i in range(len(query_labels)):
            true_class = query_labels[i].argmax().item() if len(query_labels[i].shape) > 0 else query_labels[i].item()
            
            for j in range(prototypes.shape[0]):
                if j == true_class:
                    same_class_distances.append(distances[i, j].item())
                else:
                    diff_class_distances.append(distances[i, j].item())
        
        # Plot distributions
        sns.histplot(same_class_distances, color='green', alpha=0.5, label='Same Class', ax=ax)
        sns.histplot(diff_class_distances, color='red', alpha=0.5, label='Different Class', ax=ax)
        
        ax.set_title(f"{distance_name} Distance Distributions")
        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        ax.legend()
        
        # Log to wandb
        wandb.log({f"distance_distributions/{distance_name}": wandb.Image(fig)})
        plt.close(fig)


def calculate_distances(embeddings: torch.Tensor, 
                       prototypes: torch.Tensor, 
                       distance_type: str) -> torch.Tensor:
    """
    Calculate distances between embeddings and prototypes.
    
    Args:
        embeddings (torch.Tensor): Query embeddings [n_queries, embedding_dim]
        prototypes (torch.Tensor): Class prototypes [n_classes, embedding_dim]
        distance_type (str): Type of distance to calculate
        
    Returns:
        torch.Tensor: Distance matrix [n_queries, n_classes]
    """
    # Based on the distance type, calculate accordingly
    if distance_type.lower() == 'euclidean':
        # Euclidean distance
        n_queries = embeddings.shape[0]
        n_classes = prototypes.shape[0]
        distances = torch.zeros(n_queries, n_classes, device=embeddings.device)
        
        for i in range(n_queries):
            for j in range(n_classes):
                distances[i, j] = torch.sqrt(torch.sum((embeddings[i] - prototypes[j])**2))
                
    elif distance_type.lower() == 'cosine':
        # Cosine distance
        # Normalize embeddings and prototypes
        embeddings_norm = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        prototypes_norm = torch.nn.functional.normalize(prototypes, p=2, dim=1)
        
        # Calculate similarity
        similarity = torch.mm(embeddings_norm, prototypes_norm.t())
        
        # Convert to distance (1 - similarity)
        distances = 1 - similarity
        
    elif distance_type.lower() == 'chebyshev':
        # Chebyshev distance
        n_queries = embeddings.shape[0]
        n_classes = prototypes.shape[0]
        distances = torch.zeros(n_queries, n_classes, device=embeddings.device)
        
        for i in range(n_queries):
            for j in range(n_classes):
                distances[i, j] = torch.max(torch.abs(embeddings[i] - prototypes[j]))
                
    # You can implement more distance types as needed
    else:
        # Default to Euclidean if unknown
        distances = torch.cdist(embeddings, prototypes, p=2)
        
    return distances
