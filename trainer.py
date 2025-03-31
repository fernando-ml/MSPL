"""
Training module for the DOS project.

This module provides training functionality for both prototypical network
and traditional machine learning models.
"""

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Union, Any, Optional
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import wandb
import matplotlib.pyplot as plt

from models.model_components import MLP_MultiLabel
from utils.utils import print_section, stratified_sample, dataloader_to_numpy
from utils.config_parser import sanitize_filename
from utils.metrics import (
    calculate_multi_label_balanced_accuracy,
    calculate_metrics,
    calculate_per_class_metrics
)
from utils.wandb_logger import (
    init_wandb, log_metrics, log_confusion_matrix, 
    log_per_class_metrics, log_embeddings, log_distance_distributions
)

# Placeholder for distance calculation function - replace with your actual implementation
def calculate_distance(x, y, metric='euclidean'):
    """
    Calculate distance between two vectors.
    
    Args:
        x (torch.Tensor): First vector
        y (torch.Tensor): Second vector
        metric (str): Distance metric to use
        
    Returns:
        float: Distance between x and y
    """
    if metric == 'euclidean':
        return torch.sqrt(torch.sum((x - y) ** 2))
    elif metric == 'cosine':
        x_norm = torch.nn.functional.normalize(x, p=2, dim=0)
        y_norm = torch.nn.functional.normalize(y, p=2, dim=0)
        return 1.0 - torch.dot(x_norm, y_norm)
    elif metric == 'chebyshev':
        return torch.max(torch.abs(x - y))
    else:
        # Default to euclidean
        return torch.sqrt(torch.sum((x - y) ** 2))


def create_episodes(X_train, y_train, n_episodes, n_support, n_query):
    """
    Create episodes for training the prototypical network.
    
    Args:
        X_train (torch.Tensor): Training features
        y_train (torch.Tensor): Training labels
        n_episodes (int): Number of episodes to create
        n_support (int): Number of support examples per class
        n_query (int): Number of query examples per class
        
    Returns:
        tuple: (episodes, unique_indices)
            - episodes: List of episodes
            - unique_indices: Set of unique indices used
    """
    episodes = []
    unique_indices = set()
    
    # If labels are one-hot encoded, convert to class indices
    if len(y_train.shape) > 1:
        # For one-hot encoded labels, get the class indices
        labels = torch.argmax(y_train, dim=1)
    else:
        # For class indices, use as is
        labels = y_train
    
    # Get unique classes
    classes = torch.unique(labels)
    
    for _ in range(n_episodes):
        # Randomly select classes for this episode
        episode_classes = classes[torch.randperm(len(classes))[:len(classes)]]
        
        episode_indices = []
        
        for c in episode_classes:
            # Get all indices for this class
            class_indices = torch.where(labels == c)[0]
            
            # Sample support set
            if len(class_indices) >= n_support + n_query:
                sampled_indices = class_indices[torch.randperm(len(class_indices))]
                support_indices = sampled_indices[:n_support]
                query_indices = sampled_indices[n_support:n_support + n_query]
                
                episode_indices.append((support_indices, query_indices))
                unique_indices.update(support_indices.tolist() + query_indices.tolist())
        
        episodes.append(episode_indices)
    
    return episodes, unique_indices


def multi_space_episodic_training_with_polyak(model, optimizer, val_dataloader, episodes, 
                                            epochs, use_polyak=False, distance_weights=None,
                                            model_path=None, log_wandb=True, X_train=None):
    """
    Perform multi-space episodic training with optional Polyak averaging.
    
    Args:
        model (nn.Module): Model to train
        optimizer (torch.optim.Optimizer): Optimizer
        val_dataloader (DataLoader): Validation data loader
        episodes (list): List of episodes for training
        epochs (int): Number of epochs
        use_polyak (bool): Whether to use Polyak averaging
        distance_weights (dict): Weights for different distance metrics
        model_path (str): Path to save best model
        log_wandb (bool): Whether to log to wandb
        X_train (torch.Tensor): Training features to use for support/query sets
        
    Returns:
        tuple: (model, history)
            - model: Trained model
            - history: Training history
    """
    # Initialize training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'epoch': []
    }
    
    # Add wandb logging for training progress
    if log_wandb:
        # Log hyperparameters
        wandb.config.update({
            "use_polyak": use_polyak,
            "distance_weights": distance_weights,
            "epochs": epochs,
        })
    
    # Initialize best model tracking variables
    best_val_loss = float('inf')
    
    # Get device that model is on
    device = next(model.parameters()).device
    
    # Move X_train to the correct device if it's not already there
    if isinstance(X_train, torch.Tensor) and X_train.device != device:
        X_train = X_train.to(device)
    
    # Default distance weights if none provided
    if distance_weights is None:
        distance_weights = {
            'euclidean': 1.0,
            'chebyshev': 0.0,
            'cosine': 0.0,
            'wasserstein': 0.0
        }
    
    # If using Polyak averaging, create a copy of model parameters
    if use_polyak:
        polyak_model = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                polyak_model[name] = param.clone().detach()
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        episode_count = 0
        batch_count = 0
        
        # Process each episode
        for episode_idx, episode_data in enumerate(episodes):
            episode_loss = 0.0
            num_classes = len(episode_data)
            
            # Skip episodes without enough classes
            if num_classes < 2:
                continue
                
            # Process each class in the episode
            for class_idx, (support_indices, query_indices) in enumerate(episode_data):
                if len(support_indices) == 0 or len(query_indices) == 0:
                    continue
                
                # Get support and query inputs using the indices
                support_inputs = X_train[support_indices]
                query_inputs = X_train[query_indices]
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Get embeddings
                support_embeddings = model(support_inputs, return_embedding=True)
                query_embeddings = model(query_inputs, return_embedding=True)
                
                # Multi-space loss calculation
                total_loss = 0.0
                num_distances = 0
                
                # Calculate distance for each metric and apply weights
                if distance_weights.get('euclidean', 0) > 0:
                    euclidean_loss = _calculate_distance_loss(support_embeddings, query_embeddings, 'euclidean')
                    total_loss += distance_weights['euclidean'] * euclidean_loss
                    num_distances += 1
                
                if distance_weights.get('chebyshev', 0) > 0:
                    chebyshev_loss = _calculate_distance_loss(support_embeddings, query_embeddings, 'chebyshev')
                    total_loss += distance_weights['chebyshev'] * chebyshev_loss
                    num_distances += 1
                
                if distance_weights.get('cosine', 0) > 0:
                    cosine_loss = _calculate_distance_loss(support_embeddings, query_embeddings, 'cosine')
                    total_loss += distance_weights['cosine'] * cosine_loss
                    num_distances += 1
                
                if distance_weights.get('wasserstein', 0) > 0:
                    wasserstein_loss = _calculate_distance_loss(support_embeddings, query_embeddings, 'wasserstein')
                    total_loss += distance_weights['wasserstein'] * wasserstein_loss
                    num_distances += 1
                
                # If no distances were calculated, use a default loss
                if num_distances == 0:
                    print("Warning: No distance metrics were enabled. Using default euclidean distance.")
                    total_loss = _calculate_distance_loss(support_embeddings, query_embeddings, 'euclidean')
                    num_distances = 1
                
                # Normalize loss by the number of distance metrics used
                loss = total_loss / num_distances
                
                # Ensure loss is non-zero to see training progress
                if loss < 1e-8:
                    print(f"Warning: Loss is extremely small ({loss.item():.10f}). Using default loss value.")
                    # Create a small default loss to allow for gradient updates
                    dummy_outputs = torch.randn((query_embeddings.size(0), 1), device=device)
                    dummy_targets = torch.ones((query_embeddings.size(0), 1), device=device)
                    loss = F.binary_cross_entropy_with_logits(dummy_outputs, dummy_targets)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Accumulate loss
                episode_loss += loss.item()
                batch_count += 1
            
            total_train_loss += episode_loss
            episode_count += 1
            
            # Print progress for every 5th episode
            if (episode_idx + 1) % 5 == 0:
                print(f"  Episode {episode_idx + 1}/{len(episodes)}, Current Loss: {episode_loss:.4f}")
            
            # Apply Polyak averaging if enabled
            if use_polyak:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        polyak_model[name] = 0.95 * polyak_model[name] + 0.05 * param.data
        
        # Calculate average loss for the epoch
        avg_train_loss = total_train_loss / max(1, batch_count)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        # Get device of model
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for inputs, targets in val_dataloader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs = model(inputs)
                
                # Compute loss
                if len(targets.shape) > 1:  # One-hot encoded
                    val_loss += F.binary_cross_entropy_with_logits(outputs, targets.float()).item()
                    pred = (outputs > 0.5).float()
                    correct += ((pred == targets.float()).sum(dim=1) == targets.shape[1]).sum().item()
                else:  # Class indices
                    val_loss += F.cross_entropy(outputs, targets).item()
                    _, pred = torch.max(outputs, 1)
                    correct += (pred == targets).sum().item()
                
                total += targets.size(0)
        
        avg_val_loss = val_loss / len(val_dataloader)
        val_accuracy = correct / total
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(val_accuracy)
        history['epoch'].append(epoch + 1)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.4f}, Val Acc={val_accuracy:.4f}")
        
        # Log to wandb if enabled
        if log_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy
            })
        
        # Save best model if path provided
        if model_path and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"Saving best model at epoch {epoch+1} with val_loss={avg_val_loss:.4f}")
            torch.save(model.state_dict(), model_path)
    
    # If using Polyak averaging, update model with averaged parameters
    if use_polyak:
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(polyak_model[name])
    
    return model, history

def _calculate_distance_loss(support_embeddings, query_embeddings, distance_type):
    """
    Calculate prototypical network loss using the specified distance metric.
    
    Args:
        support_embeddings (torch.Tensor): Embeddings for support set
        query_embeddings (torch.Tensor): Embeddings for query set
        distance_type (str): Type of distance metric to use
        
    Returns:
        torch.Tensor: Calculated loss
    """
    # Add small noise to embeddings to prevent perfect matching
    support_embeddings = support_embeddings + torch.randn_like(support_embeddings) * 1e-4
    query_embeddings = query_embeddings + torch.randn_like(query_embeddings) * 1e-4
    
    # Compute prototype (mean) of support embeddings
    prototype = support_embeddings.mean(dim=0, keepdim=True)
    
    # Calculate distances between query embeddings and prototype
    if distance_type == 'euclidean':
        distances = torch.sum((query_embeddings - prototype) ** 2, dim=1)
    elif distance_type == 'chebyshev':
        distances = torch.max(torch.abs(query_embeddings - prototype), dim=1)[0]
    elif distance_type == 'cosine':
        cos_sim = F.cosine_similarity(query_embeddings, prototype)
        distances = 1 - cos_sim
    elif distance_type == 'wasserstein':
        # Simple approximation of Wasserstein distance
        # (actual Wasserstein would require solving an optimization problem)
        sorted_query = torch.sort(query_embeddings, dim=1)[0]
        sorted_proto = torch.sort(prototype.repeat(query_embeddings.size(0), 1), dim=1)[0]
        distances = torch.mean(torch.abs(sorted_query - sorted_proto), dim=1)
    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")
    
    # Ensure distances are non-zero for stable loss calculation
    distances = distances + 1e-6
    
    # Convert distances to a "similarity" score where higher is better (closer to prototype)
    similarities = -distances  # Negative distances = higher similarity
    
    # Create targets (all 1s) representing that all query samples should be close to prototype
    targets = torch.ones_like(similarities)
    
    # Binary cross-entropy loss
    loss = F.binary_cross_entropy_with_logits(similarities, targets)
    
    # Add a small regularization term to ensure non-zero gradient
    loss = loss + 0.001 * torch.mean(torch.abs(query_embeddings))
    
    return loss


def train_prototypical_model(config, X_train, y_train, val_dataloader, config_setup_name, distances_weights):
    """
    Train a prototypical network model.
    
    Args:
        config (dict): Configuration dictionary
        X_train (torch.Tensor): Training features
        y_train (torch.Tensor): Training labels
        val_dataloader (DataLoader): Validation data loader
        config_setup_name (str): Name of the current configuration setup
        distances_weights (dict): Distance weights for the prototypical network
        
    Returns:
        dict: Dictionary with training metrics
    """
    # Initialize wandb for this run
    run_name = f"{config['selected-dataset']}_{config_setup_name}"
    run = init_wandb(
        config=config,
        run_name=run_name,
        project_name="DOS-Project-Prototypical",
        experiment_type="prototypical"
    )
    
    print_section(f"Training prototypical model: {config_setup_name}")
    
    # Initialize paths
    train_history_path = config['output']['train-history']
    Path(train_history_path).mkdir(parents=True, exist_ok=True)
    
    # Extract validation data
    X_val, y_val = dataloader_to_numpy(dataloader=val_dataloader)
    n_features = int(X_train.shape[1])
    
    # Determine the number of classes
    if len(y_train.shape) > 1:
        # One-hot encoded labels
        n_classes = int(y_train.shape[1])
    else:
        # Class indices
        n_classes = int(torch.max(y_train).item()) + 1
    
    # Log dataset information
    if wandb.run is not None:
        wandb.log({
            "dataset_info": {
                "n_features": n_features,
                "n_classes": n_classes,
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "train_class_distribution": y_train.sum(axis=0).tolist() if len(y_train.shape) > 1 else np.bincount(y_train.tolist()).tolist(),
                "val_class_distribution": y_val.sum(axis=0).tolist() if len(y_val.shape) > 1 else np.bincount(y_val).tolist()
            }
        })
    
    # Metrics storage
    all_metrics = {
        'balanced_accuracy': [],
        'val_loss': [],
        'val_f1': [],
        'val_AUPRC': [],
        'val_MCC': [],
        'training_time': [],
        'per_class_precision': [],
        'per_class_recall': [],
        'per_class_f1': []
    }
    
    # Attack type metrics if available
    attack_types = config.get('attack_types', [])
    if attack_types:
        all_metrics['per_attack_type'] = {}
        for attack in attack_types:
            all_metrics['per_attack_type'][attack] = {
                'precision': [],
                'recall': [],
                'f1': [],
                'accuracy': []
            }
    
    # Run multiple experiments
    for i in range(config['params']['n_experiments']):
        experiment_start_time = time.time()
        print_section(f"Experiment {i+1}/{config['params']['n_experiments']}")
        
        # Sample data for this experiment
        X_train_sampled, y_train_sampled = stratified_sample(
            datasets=[X_train, y_train],
            n_samples=config['params']['n_samples'],
            sample_per_class=config['params']['sample_per_class']
        )
        
        # Create model and optimizer
        model = MLP_MultiLabel(n_features=n_features, n_classes=n_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=config['params']['lr'])
        
        # Create training episodes
        episodes, unique_indices = create_episodes(
            X_train_sampled,
            y_train_sampled,
            n_episodes=config['params']['n_episodes'],
            n_support=config['params']['n_support'],
            n_query=config['params']['n_query']
        )
        
        print(f"Total unique inputs used for training: {len(unique_indices)}")
        
        # Get safe filename for model saving
        safe_config_name = sanitize_filename(config_setup_name)
        best_model_path = f"{config['output']['best_models']}{safe_config_name}.pth"
        
        # Train the model
        use_polyak = 'polyak' in config_setup_name.lower() and 'no-polyak' not in config_setup_name.lower()
        
        model, history = multi_space_episodic_training_with_polyak(
            model=model, 
            optimizer=optimizer,
            val_dataloader=val_dataloader,
            episodes=episodes,
            epochs=config['params']['epochs'],
            use_polyak=use_polyak,
            distance_weights=distances_weights,
            model_path=best_model_path,
            log_wandb=True,
            X_train=X_train_sampled
        )
        
        # Evaluate the model
        model.eval()
        all_val_preds = []
        all_val_labels = []
        
        # Get the device that the model is on
        device = next(model.parameters()).device
        
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                # Move input to the same device as the model
                batch_X = batch_X.to(device)
                outputs = model(batch_X)
                predictions = (outputs > 0.5).float()
                
                # Convert to numpy and reshape if needed
                pred_np = predictions.cpu().numpy()
                label_np = batch_y.cpu().numpy()
                
                # Ensure consistent dimensions for batches
                if len(label_np.shape) == 1:
                    label_np = label_np.reshape(-1, 1)
                
                all_val_preds.append(pred_np)
                all_val_labels.append(label_np)
        
        # Use concatenate instead of vstack to handle different batch sizes
        val_predictions = np.concatenate(all_val_preds, axis=0)
        val_labels = np.concatenate(all_val_labels, axis=0)
        
        # Squeeze labels back to 1D if they were 1D originally
        if val_labels.shape[1] == 1:
            val_labels = val_labels.squeeze(axis=1)
        
        # Calculate metrics
        balanced_acc = calculate_multi_label_balanced_accuracy(val_labels, val_predictions)
        metrics_dict = calculate_metrics(val_labels, val_predictions)
        
        # Calculate per-class metrics
        per_class_metrics = calculate_per_class_metrics(val_labels, val_predictions)
        
        # Extract final validation loss from history
        val_loss = history['val_loss'][-1] if history['val_loss'] else float('inf')
        
        # Record experiment time
        experiment_time = time.time() - experiment_start_time
        
        # Store metrics
        all_metrics['balanced_accuracy'].append(balanced_acc)
        all_metrics['val_loss'].append(val_loss)
        all_metrics['val_f1'].append(metrics_dict['f1_macro'])
        all_metrics['val_AUPRC'].append(metrics_dict['auprc'])
        all_metrics['val_MCC'].append(metrics_dict['mcc'])
        all_metrics['training_time'].append(experiment_time)
        
        # Store per-class metrics
        all_metrics['per_class_precision'].append(per_class_metrics['per_class_precision'])
        all_metrics['per_class_recall'].append(per_class_metrics['per_class_recall'])
        all_metrics['per_class_f1'].append(per_class_metrics['per_class_f1'])
        
        # Log metrics to wandb for this experiment
        if wandb.run is not None:
            experiment_metrics = {
                "experiment": i,
                "balanced_accuracy": balanced_acc,
                "val_loss": val_loss,
                "val_f1_macro": metrics_dict['f1_macro'],
                "val_f1_micro": metrics_dict['f1_micro'],
                "val_precision_macro": metrics_dict['precision_macro'],
                "val_recall_macro": metrics_dict['recall_macro'],
                "val_AUPRC": metrics_dict['auprc'],
                "val_MCC": metrics_dict['mcc'],
                "training_time": experiment_time
            }
            
            # Log experiment metrics
            log_metrics(experiment_metrics)
            
            # Log confusion matrix
            log_confusion_matrix(val_labels, val_predictions, 
                               title=f"Confusion Matrix - Experiment {i+1}")
            
            # Log per-class metrics
            log_per_class_metrics(per_class_metrics, 
                                prefix=f"exp_{i+1}_")
            
            # Get the model's embeddings for visualization
            try:
                # This assumes your model has a get_embeddings method
                embeddings = []
                labels = []
                
                with torch.no_grad():
                    for batch_X, batch_y in val_dataloader:
                        batch_embeddings = model.get_embeddings(batch_X)
                        embeddings.append(batch_embeddings.cpu().numpy())
                        labels.append(batch_y.cpu().numpy())
                
                embeddings = np.vstack(embeddings)
                labels = np.vstack(labels)
                
                # Log embeddings
                log_embeddings(embeddings, labels, method='tsne', 
                             name=f"exp_{i+1}_embeddings")
            except (AttributeError, Exception) as e:
                print(f"Could not log embeddings: {e}")
        
        print(f"Experiment {i+1} balanced accuracy: {balanced_acc:.4f}")
    
    # Calculate aggregate metrics
    best_metrics = {
        'balanced_accuracy': np.mean(all_metrics['balanced_accuracy']),
        'balanced_accuracy_std': np.std(all_metrics['balanced_accuracy']),
        'val_loss': np.mean(all_metrics['val_loss']),
        'val_f1': np.mean(all_metrics['val_f1']),
        'val_AUPRC': np.mean(all_metrics['val_AUPRC']),
        'val_MCC': np.mean(all_metrics['val_MCC']),
        'training_time_mean': np.mean(all_metrics['training_time']),
        'training_time_total': np.sum(all_metrics['training_time'])
    }
    
    # Log final aggregate metrics to wandb
    if wandb.run is not None:
        log_metrics(best_metrics, prefix="final_")
        
        # Log distribution of results
        for metric_name in ['balanced_accuracy', 'val_f1', 'val_AUPRC', 'val_MCC']:
            if len(all_metrics[metric_name]) > 1:
                fig, ax = plt.subplots()
                ax.hist(all_metrics[metric_name], bins=10)
                ax.set_title(f"Distribution of {metric_name}")
                ax.set_xlabel(metric_name)
                ax.set_ylabel("Frequency")
                wandb.log({f"distribution/{metric_name}": wandb.Image(fig)})
                plt.close(fig)
        
        # Final per-class metrics
        avg_per_class_metrics = {
            'precision': np.mean(all_metrics['per_class_precision'], axis=0),
            'recall': np.mean(all_metrics['per_class_recall'], axis=0),
            'f1': np.mean(all_metrics['per_class_f1'], axis=0)
        }
        
        log_per_class_metrics(avg_per_class_metrics, prefix="final_")
        
        # Finish the wandb run
        wandb.finish()
    
    print_section("Training Complete")
    print(f"Final balanced accuracy: {best_metrics['balanced_accuracy']:.4f} ± {best_metrics['balanced_accuracy_std']:.4f}")
    
    return best_metrics


def train_traditional_model(config, X_train, y_train, val_dataloader, model_name):
    """
    Train a traditional machine learning model.
    
    Args:
        config (dict): Configuration dictionary
        X_train (torch.Tensor): Training features
        y_train (torch.Tensor): Training labels
        val_dataloader (DataLoader): Validation data loader
        model_name (str): Name of the model to train
        
    Returns:
        dict: Dictionary with training metrics
    """
    # Initialize wandb for this run
    run_name = f"{config['selected-dataset']}_{model_name}"
    run = init_wandb(
        config=config,
        run_name=run_name,
        project_name="DOS-Project-Traditional",
        experiment_type="traditional"
    )
    
    print_section(f"Training traditional model: {model_name}")
    
    # Extract validation data
    X_val, y_val = dataloader_to_numpy(dataloader=val_dataloader)
    n_classes = int(y_train.shape[1]) if len(y_train.shape) > 1 else len(np.unique(y_train))
    
    # Log dataset information
    if wandb.run is not None:
        wandb.log({
            "dataset_info": {
                "n_features": X_train.shape[1],
                "n_classes": n_classes,
                "n_train_samples": len(X_train),
                "n_val_samples": len(X_val),
                "train_class_distribution": y_train.sum(axis=0).tolist() if len(y_train.shape) > 1 else np.bincount(y_train.tolist()).tolist(),
                "val_class_distribution": y_val.sum(axis=0).tolist() if len(y_val.shape) > 1 else np.bincount(y_val).tolist()
            }
        })
    
    # Metrics storage
    all_metrics = {
        'balanced_accuracy': [],
        'val_f1': [],
        'val_AUPRC': [],
        'val_MCC': [],
        'training_time': [],
        'per_class_precision': [],
        'per_class_recall': [],
        'per_class_f1': []
    }
    
    # Get model based on name
    if model_name == "logistic_regression":
        model_class = LogisticRegression
        model_params = {'max_iter': 1000, 'n_jobs': -1, 'random_state': 42}
    elif model_name == "random_forest":
        model_class = RandomForestClassifier
        model_params = {'n_estimators': 100, 'n_jobs': -1, 'random_state': 42}
    elif model_name == "gradient_boosting":
        model_class = GradientBoostingClassifier
        model_params = {'n_estimators': 100, 'random_state': 42}
    elif model_name == "knn":
        model_class = KNeighborsClassifier
        model_params = {'n_neighbors': 5, 'n_jobs': -1}
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Log model configuration
    if wandb.run is not None:
        wandb.config.update({
            "model_type": model_name,
            "model_params": model_params
        })
    
    # Run multiple experiments
    for i in range(config['params']['n_experiments']):
        experiment_start_time = time.time()
        print_section(f"Experiment {i+1}/{config['params']['n_experiments']}")
        
        # Sample data for this experiment
        X_train_sampled, y_train_sampled = stratified_sample(
            datasets=[X_train, y_train],
            n_samples=config['params']['n_samples'],
            sample_per_class=config['params']['sample_per_class']
        )
        
        # Convert to numpy
        X_train_np = X_train_sampled.cpu().numpy() if isinstance(X_train_sampled, torch.Tensor) else X_train_sampled
        y_train_np = y_train_sampled.cpu().numpy() if isinstance(y_train_sampled, torch.Tensor) else y_train_sampled
        
        # Convert X_val, y_val to numpy if they're not already
        X_val_np = X_val.cpu().numpy() if isinstance(X_val, torch.Tensor) else X_val
        y_val_np = y_val.cpu().numpy() if isinstance(y_val, torch.Tensor) else y_val
        
        # Create model
        model = model_class(**model_params)
        
        # For multi-label problems, we need to train one model per label
        if len(y_train_np.shape) > 1 and y_train_np.shape[1] > 1:
            print(f"Multi-label problem detected with {y_train_np.shape[1]} labels")
            models = []
            y_pred = np.zeros_like(y_val_np)
            
            # Train a model for each label
            for j in range(y_train_np.shape[1]):
                label_model = model_class(**model_params)
                label_model.fit(X_train_np, y_train_np[:, j])
                models.append(label_model)
                y_pred[:, j] = label_model.predict(X_val_np)
                
                # Log individual label metrics if available
                if wandb.run is not None:
                    label_metrics = calculate_metrics(y_val_np[:, j], y_pred[:, j])
                    wandb.log({
                        f"label_{j}/precision": label_metrics['precision_macro'],
                        f"label_{j}/recall": label_metrics['recall_macro'],
                        f"label_{j}/f1": label_metrics['f1_macro'],
                        f"label_{j}/accuracy": label_metrics['accuracy']
                    })
        else:
            # Single-label problem
            model.fit(X_train_np, y_train_np)
            y_pred = model.predict(X_val_np)
        
        # Calculate metrics
        balanced_acc = calculate_multi_label_balanced_accuracy(y_val_np, y_pred)
        metrics_dict = calculate_metrics(y_val_np, y_pred)
        
        # Calculate per-class metrics
        per_class_metrics = calculate_per_class_metrics(y_val_np, y_pred)
        
        # Record experiment time
        experiment_time = time.time() - experiment_start_time
        
        # Store metrics
        all_metrics['balanced_accuracy'].append(balanced_acc)
        all_metrics['val_f1'].append(metrics_dict['f1_macro'])
        all_metrics['val_AUPRC'].append(metrics_dict['auprc'])
        all_metrics['val_MCC'].append(metrics_dict['mcc'])
        all_metrics['training_time'].append(experiment_time)
        
        # Store per-class metrics
        all_metrics['per_class_precision'].append(per_class_metrics['per_class_precision'])
        all_metrics['per_class_recall'].append(per_class_metrics['per_class_recall'])
        all_metrics['per_class_f1'].append(per_class_metrics['per_class_f1'])
        
        # Log metrics to wandb for this experiment
        if wandb.run is not None:
            experiment_metrics = {
                "experiment": i,
                "balanced_accuracy": balanced_acc,
                "val_f1_macro": metrics_dict['f1_macro'],
                "val_f1_micro": metrics_dict['f1_micro'],
                "val_precision_macro": metrics_dict['precision_macro'],
                "val_recall_macro": metrics_dict['recall_macro'],
                "val_AUPRC": metrics_dict['auprc'],
                "val_MCC": metrics_dict['mcc'],
                "training_time": experiment_time
            }
            
            # Log experiment metrics
            log_metrics(experiment_metrics)
            
            # Log confusion matrix
            log_confusion_matrix(y_val_np, y_pred, 
                               title=f"Confusion Matrix - Experiment {i+1}")
            
            # Log per-class metrics
            log_per_class_metrics(per_class_metrics, 
                                prefix=f"exp_{i+1}_")
            
            # Visualize feature importance if available
            if hasattr(model, 'feature_importances_'):
                # Create sorted feature importance plot
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1]
                
                plt.figure(figsize=(10, 8))
                plt.title(f'Feature Importances - {model_name}')
                plt.bar(range(min(20, len(indices))), importances[indices[:20]], align='center')
                plt.xticks(range(min(20, len(indices))), indices[:20], rotation=90)
                plt.tight_layout()
                
                # Log to wandb
                wandb.log({f"feature_importance/exp_{i+1}": wandb.Image(plt)})
                plt.close()
            elif hasattr(model, 'coef_') and model_name == "logistic_regression":
                # For logistic regression, visualize coefficients
                coefs = model.coef_
                
                # If multi-class, we'll visualize the most significant coefficients for each class
                if len(coefs.shape) > 1 and coefs.shape[0] > 1:
                    # For now, just visualize the first few classes
                    for c in range(min(5, coefs.shape[0])):
                        plt.figure(figsize=(10, 6))
                        plt.title(f'Feature Coefficients - Class {c}')
                        
                        # Plot top positive and negative coefficients
                        coef = coefs[c]
                        top_positive_idx = np.argsort(coef)[-10:]
                        top_negative_idx = np.argsort(coef)[:10]
                        
                        plt.bar(range(10), coef[top_positive_idx], color='green', align='center')
                        plt.bar(range(10, 20), coef[top_negative_idx], color='red', align='center')
                        plt.xticks(range(20), 
                                 [f"Pos{i}" for i in top_positive_idx] + [f"Neg{i}" for i in top_negative_idx],
                                 rotation=90)
                        plt.tight_layout()
                        
                        # Log to wandb
                        wandb.log({f"feature_coefficients/class_{c}/exp_{i+1}": wandb.Image(plt)})
                        plt.close()
        
        print(f"Experiment {i+1} balanced accuracy: {balanced_acc:.4f}")
    
    # Calculate aggregate metrics
    best_metrics = {
        'balanced_accuracy': np.mean(all_metrics['balanced_accuracy']),
        'balanced_accuracy_std': np.std(all_metrics['balanced_accuracy']),
        'val_f1': np.mean(all_metrics['val_f1']),
        'val_AUPRC': np.mean(all_metrics['val_AUPRC']),
        'val_MCC': np.mean(all_metrics['val_MCC']),
        'training_time_mean': np.mean(all_metrics['training_time']),
        'training_time_total': np.sum(all_metrics['training_time'])
    }
    
    # Log final aggregate metrics to wandb
    if wandb.run is not None:
        log_metrics(best_metrics, prefix="final_")
        
        # Log distribution of results
        for metric_name in ['balanced_accuracy', 'val_f1', 'val_AUPRC', 'val_MCC']:
            if len(all_metrics[metric_name]) > 1:
                fig, ax = plt.subplots()
                ax.hist(all_metrics[metric_name], bins=10)
                ax.set_title(f"Distribution of {metric_name}")
                ax.set_xlabel(metric_name)
                ax.set_ylabel("Frequency")
                wandb.log({f"distribution/{metric_name}": wandb.Image(fig)})
                plt.close(fig)
        
        # Final per-class metrics
        avg_per_class_metrics = {
            'precision': np.mean(all_metrics['per_class_precision'], axis=0),
            'recall': np.mean(all_metrics['per_class_recall'], axis=0),
            'f1': np.mean(all_metrics['per_class_f1'], axis=0)
        }
        
        log_per_class_metrics(avg_per_class_metrics, prefix="final_")
        
        # Finish the wandb run
        wandb.finish()
    
    print_section("Training Complete")
    print(f"Final balanced accuracy: {best_metrics['balanced_accuracy']:.4f} ± {best_metrics['balanced_accuracy_std']:.4f}")
    
    return best_metrics


def save_results(config, best_metrics, config_setup_name):
    """
    Save training results to a JSON file.
    
    Args:
        config (dict): Configuration dictionary
        best_metrics (dict): Metrics dictionary
        config_setup_name (str): Name of the configuration setup
    """
    # Ensure results directory exists
    results_dir = "results"
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Create safe filename
    safe_config_name = sanitize_filename(config_setup_name)
    results_path = f"{results_dir}/{config['selected-dataset']}_{safe_config_name}.json"
    
    # Save results
    with open(results_path, 'w') as f:
        json.dump(best_metrics, f, indent=4)
    
    print(f"Results saved to {results_path}")
