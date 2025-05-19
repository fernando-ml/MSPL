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
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef, average_precision_score
import random
import wandb
import matplotlib.pyplot as plt

from models.model_components import MLP_MultiLabel
from utils.utils import print_section, stratified_sample, dataloader_to_numpy
from utils.datasets import dataloader_to_numpy as other_dataloader_to_numpy
from utils.config_parser import sanitize_filename
from utils.metrics import (
    calculate_multi_label_balanced_accuracy,
    calculate_metrics,
    calculate_per_class_metrics
)
from utils.wandb_logger import (
    init_wandb, log_metrics, log_confusion_matrix,
    log_per_class_metrics,
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


def multi_label_balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy for multi-label classification.

    :param y_true: True labels, either one-hot encoded or single class labels (n_samples, n_classes) or (n_samples,)
    :param y_pred: Predicted labels, either one-hot encoded or single class labels (n_samples, n_classes) or (n_samples,)
    :return: Balanced accuracy score and list of per-class accuracies
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Determine the number of classes from y_true if possible, otherwise use y_pred
    n_classes = y_true.shape[1] if len(y_true.shape) > 1 else len(
        np.unique(np.concatenate((y_true, y_pred))))

    # Handle case where labels are not one-hot encoded
    if len(y_true.shape) == 1:
        y_true_one_hot = np.zeros((y_true.size, n_classes))
        y_true_one_hot[np.arange(y_true.size), y_true] = 1
        y_true = y_true_one_hot

    if len(y_pred.shape) == 1:
        y_pred_one_hot = np.zeros((y_pred.size, n_classes))
        y_pred_one_hot[np.arange(y_pred.size), y_pred] = 1
        y_pred = y_pred_one_hot

    class_accuracies = []

    for i in range(n_classes):
        true_pos = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        true_neg = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        false_pos = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        false_neg = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))

        sensitivity = true_pos / \
            (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / \
            (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0

        class_accuracies.append((sensitivity + specificity) / 2)

    return np.mean(class_accuracies), class_accuracies


def create_episodes(X, y, n_episodes, n_support, n_query):
    episodes = []
    class_indices = {label: [] for label in range(y.shape[1])}
    all_indices = set()

    # separate indices for each class
    for idx, labels in enumerate(y):
        for label in range(y.shape[1]):
            if labels[label] == 1:
                class_indices[label].append(idx)

    for _ in range(n_episodes):
        support_indices = []
        query_indices = []

        for class_label in range(y.shape[1]):
            if len(class_indices[class_label]) >= n_support + n_query:
                selected_indices = random.sample(
                    class_indices[class_label], n_support + n_query)
            elif len(class_indices[class_label]) > 0:
                # if not enough samples, repeat the indices until we reach the required number
                selected_indices = random.sample(class_indices[class_label] * (
                    (n_support + n_query) // len(class_indices[class_label]) + 1), n_support + n_query)
            else:
                # skip this class if there are no samples available
                continue

            all_indices.update(selected_indices)
            support_indices.extend(selected_indices[:n_support])
            query_indices.extend(selected_indices[n_support:])

        if support_indices and query_indices:
            support_set = (X[support_indices], y[support_indices])
            query_set = (X[query_indices], y[query_indices])
            episodes.append((support_set, query_set))

    return episodes, all_indices


# Distance functions
def cosine_similarity(x1, x2):
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    return torch.mm(x1_norm, x2_norm.transpose(0, 1))


def chebyshev_distance(x1, x2):
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    abs_diff = torch.abs(diff)
    distances = torch.max(abs_diff, dim=-1).values
    return distances


def wasserstein_distance_torch(x1, x2):
    x1_sorted, _ = torch.sort(x1, dim=-1)
    x2_sorted, _ = torch.sort(x2, dim=-1)
    return torch.mean(torch.abs(x1_sorted - x2_sorted), dim=-1)


def z_score_normalize_with_clipping(embeddings, clip_range=3.0):
    mean = embeddings.mean(dim=0, keepdim=True)
    std = embeddings.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-8)  # avoid division by zero

    z_scores = (embeddings - mean) / std
    clipped_z_scores = torch.clamp(z_scores, min=-clip_range, max=clip_range)

    return clipped_z_scores


def compute_prototypes(embeddings, labels):
    prototypes = []
    for i in range(labels.shape[1]):
        class_embeddings = embeddings[labels[:, i] == 1]
        if len(class_embeddings) > 0:
            prototypes.append(class_embeddings.mean(0))
        else:
            prototypes.append(torch.zeros(
                embeddings.shape[1], device=embeddings.device))
    return torch.stack(prototypes)


def calculate_distances(query_embeddings, prototypes, weights):
    """
    Calculate distances between query embeddings and prototypes.
    """
    euclidean_distances = torch.cdist(query_embeddings, prototypes)
    cosine_distances = 1 - cosine_similarity(query_embeddings, prototypes)
    chebyshev_distances = chebyshev_distance(query_embeddings, prototypes)
    wasserstein_distances = wasserstein_distance_torch(
        query_embeddings.unsqueeze(1), prototypes.unsqueeze(0))
    # Normalize distances
    normalized_euclidean_distances = z_score_normalize_with_clipping(
        euclidean_distances)
    normalized_cosine_distances = z_score_normalize_with_clipping(
        cosine_distances)
    normalized_chebyshev_distances = z_score_normalize_with_clipping(
        chebyshev_distances)
    normalized_wasserstein_distances = z_score_normalize_with_clipping(
        wasserstein_distances)

    combined_distances = (weights['euclidean'] * normalized_euclidean_distances) +\
        (weights['cosine'] * normalized_cosine_distances) +\
        (weights['chebyshev'] * normalized_chebyshev_distances) +\
        (weights['wasserstein'] *
         normalized_wasserstein_distances)
    return combined_distances


def improved_hybrid_prototype_loss(query_embeddings, prototypes, query_labels, weights):
    combined_distances = calculate_distances(
        query_embeddings, prototypes, weights)
    predictions = torch.sigmoid(-combined_distances)
    criterion = nn.BCELoss()
    return criterion(predictions, query_labels)


def new_validate_with_prototypes(model, dataloader, weights, alpha=1):
    model.eval()
    all_predictions = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            embeddings = model(inputs, return_embedding=True)
            prototypes = compute_prototypes(embeddings, targets)

            combined_distances = calculate_distances(
                embeddings, prototypes, weights)
            predictions = torch.sigmoid(-combined_distances)

            loss = improved_hybrid_prototype_loss(
                embeddings, prototypes, targets, weights)
            total_loss += loss.item()

            all_predictions.append(predictions)
            all_labels.append(targets)

    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)

    # convert predictions to binary (0 or 1) using 0.5 as threshold
    max_indices = torch.argmax(all_predictions, dim=1, keepdim=True)
    binary_predictions = torch.zeros_like(all_predictions)
    binary_predictions.scatter_(1, max_indices, 1)

    # Calculate metrics
    f1 = f1_score(all_labels.cpu().numpy(),
                  binary_predictions.cpu().numpy(), average='micro')
    mcc = matthews_corrcoef(np.argmax(all_labels.cpu().numpy(), axis=1), np.argmax(
        binary_predictions.cpu().numpy(), axis=1))
    auprc = average_precision_score(all_labels.cpu().numpy(
    ), binary_predictions.cpu().numpy(), average="micro")

    balanced_acc, classes_acc = multi_label_balanced_accuracy(
        all_labels.cpu().numpy(), binary_predictions.cpu().numpy())

    # calculate average loss
    avg_loss = total_loss / len(dataloader)

    return balanced_acc, avg_loss, f1, mcc, auprc


def multi_space_episodic_training_with_polyak(model, optimizer, val_dataloader, episodes, epochs, use_polyak=False, distance_weights=None, model_path=None, log_wandb=True, X_train=None):
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
        'epoch': [],
        'val_MCC': [],
        'val_AUPRC': [],
        "val_f1": [],
    }

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
        ema_model = type(model)(model.input_layer.in_features,
                                model.output_layer.out_features)
        ema_model.load_state_dict(model.state_dict())

    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        # Process each episode
        for episode_idx, episode_data in enumerate(episodes):
            model.train()  # va en episodio
            if use_polyak:
                ema_model.eval()

            support_set, query_set = episode_data
            support_embeddings = model(support_set[0], return_embedding=True)
            query_embeddings = model(query_set[0], return_embedding=True)

            prototypes = compute_prototypes(support_embeddings, support_set[1])
            loss = improved_hybrid_prototype_loss(
                query_embeddings, prototypes, query_set[1], distance_weights)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)  # Clip gradients
            optimizer.step()

            total_loss += loss.item()

            # Apply Polyak averaging if enabled
            if use_polyak:
                with torch.no_grad():
                    polyak_decay = 0.999
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(polyak_decay).add_(
                            param.data, alpha=1 - polyak_decay)
                    wandb.config.update({
                        "polyak_decay": polyak_decay,
                    })

        # Calculate average loss for the epoch
        avg_train_loss = total_loss / len(episodes)

        if use_polyak:
            # Use the EMA model for validation
            balanced_acc, avg_val_loss, val_f1, mcc, auprc = new_validate_with_prototypes(
                ema_model, val_dataloader, distance_weights)
        else:
            balanced_acc, avg_val_loss, val_f1, mcc, auprc = new_validate_with_prototypes(
                model, val_dataloader, distance_weights)

        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(balanced_acc)
        history['epoch'].append(epoch + 1)
        history["val_f1"].append(val_f1)
        history["val_MCC"].append(mcc)
        history["val_AUPRC"].append(auprc)

        # Print epoch results
        print(f"Epoch {epoch+1}/{epochs}: Train Loss={avg_train_loss:.6f}, Val Loss={avg_val_loss:.4f}, Val Acc={balanced_acc:.4f}")

        # Log to wandb if enabled
        # if log_wandb:
        #     wandb.log({
        #         "epoch": epoch + 1,
        #         "train_loss": avg_train_loss,
        #         "val_loss": avg_val_loss,
        #         "val_accuracy": balanced_acc
        #     })

        # Save best model if path provided
        if model_path and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(
                f"Saving best model at epoch {epoch+1} with val_loss={avg_val_loss:.4f}")
            torch.save(model.state_dict(), model_path)

    # If using Polyak averaging, update model with averaged parameters
    if use_polyak:
        model.load_state_dict(ema_model.state_dict())

    return model, history


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
    # Train the model
    use_polyak = 'polyak' in config_setup_name.lower(
    ) and 'no-polyak' not in config_setup_name.lower()

    run = init_wandb(
        config=config,
        run_name=run_name,
        project_name="DOS-Erdos",
        polyak=use_polyak,
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
        run.summary["dataset_info"] = {
            "n_features": n_features,
            "n_classes": n_classes,
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "train_class_distribution": y_train.sum(axis=0).tolist() if len(y_train.shape) > 1 else np.bincount(y_train.tolist()).tolist(),
            "val_class_distribution": y_val.sum(axis=0).tolist() if len(y_val.shape) > 1 else np.bincount(y_val).tolist()
        }

    # Metrics storage
    all_metrics = {
        'n_experiment': [],
        'val_accuracy': [],
        'val_loss': [],
        'val_f1': [],
        'val_AUPRC': [],
        'val_MCC': [],
        'training_time': [],
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
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config['params']['lr'])

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

        # Record experiment time
        experiment_time = time.time() - experiment_start_time

        # Store metrics
        all_metrics['val_accuracy'].append(max(history['val_accuracy']))
        all_metrics['val_loss'].append(max(history['val_loss']))
        all_metrics['val_f1'].append(max(history['val_f1']))
        all_metrics['val_AUPRC'].append(max(history['val_AUPRC']))
        all_metrics['val_MCC'].append(max(history['val_MCC']))
        all_metrics['training_time'].append(experiment_time)
        all_metrics['n_experiment'].append(i)

        print(
            f"Experiment {i+1} val accuracy: {max(history['val_accuracy']):.4f}")

    # Calculate aggregate metrics
    best_metrics = {
        'val_balanced_accuracy': np.mean(all_metrics['val_accuracy']),
        'val_balanced_accuracy_std': np.std(all_metrics['val_accuracy']),
        'val_loss': np.mean(all_metrics['val_loss']),
        'val_f1': np.mean(all_metrics['val_f1']),
        'val_AUPRC': np.mean(all_metrics['val_AUPRC']),
        'val_MCC': np.mean(all_metrics['val_MCC']),
        'training_time_mean': np.mean(all_metrics['training_time']),
        'training_time_total': np.sum(all_metrics['training_time'])
    }

    # Log final aggregate metrics to wandb
    if wandb.run is not None:
        table_data_df = pd.DataFrame(all_metrics)
        metrics_table = wandb.Table(dataframe=table_data_df)
        wandb.log({"experiment_summary_table": metrics_table})

        # Find the best row (e.g., row with highest accuracy)
        # For highest accuracy
        best_row_index = table_data_df['val_loss'].idxmin()
        best_row = table_data_df.iloc[best_row_index].to_dict()
        best_row_with_prefix = {f"best_{k}": v for k, v in best_row.items()}
        for key, value in best_row_with_prefix.items():
            wandb.run.summary[key] = value

        wandb.finish()

    print_section("Training Complete")
    print(
        f"Final balanced accuracy: {best_metrics['val_balanced_accuracy']:.4f} ± {best_metrics['val_balanced_accuracy_std']:.4f}")

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
    X_val, y_val = other_dataloader_to_numpy(dataloader=val_dataloader)
    n_features = int(X_train.shape[1])
    n_classes = int(y_train.shape[1]) if len(
        y_train.shape) > 1 else len(np.unique(y_train))

    # Log dataset information
    if wandb.run is not None:
        run.summary["dataset_info"] = {
            "n_features": n_features,
            "n_classes": n_classes,
            "n_train_samples": len(X_train),
            "n_val_samples": len(X_val),
            "train_class_distribution": y_train.sum(axis=0).tolist() if len(y_train.shape) > 1 else np.bincount(y_train.tolist()).tolist(),
            "val_class_distribution": y_val.sum(axis=0).tolist() if len(y_val.shape) > 1 else np.bincount(y_val).tolist()
        }

    # Metrics storage
    all_metrics = {
        'n_experiment': [],
        'val_accuracy': [],
        'val_f1': [],
        'val_AUPRC': [],
        'val_MCC': [],
        'training_time': [],
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

        # Create model
        model = model_class(**model_params)

        # Single-label problem
        model.fit(X_train_sampled, np.argmax(y_train_sampled, axis=1))
        y_pred = model.predict(X_val)

        # Calculate metrics
        balanced_acc = calculate_multi_label_balanced_accuracy(
            y_val, y_pred)
        metrics_dict = calculate_metrics(y_val, y_pred)

        # Record experiment time
        experiment_time = time.time() - experiment_start_time

        # Store metrics
        all_metrics['n_experiment'].append(i)
        all_metrics['val_accuracy'].append(metrics_dict['balanced_accuracy'])
        all_metrics['val_f1'].append(metrics_dict['f1_macro'])
        all_metrics['val_AUPRC'].append(metrics_dict['auprc'])
        all_metrics['val_MCC'].append(metrics_dict['mcc'])
        all_metrics['training_time'].append(experiment_time)

        print(f"Experiment {i+1} balanced accuracy: {balanced_acc:.4f}")

    # Calculate aggregate metrics
    best_metrics = {
        'val_balanced_accuracy': np.mean(all_metrics['val_accuracy']),
        'val_balanced_accuracy_std': np.std(all_metrics['val_accuracy']),
        'val_f1': np.mean(all_metrics['val_f1']),
        'val_AUPRC': np.mean(all_metrics['val_AUPRC']),
        'val_MCC': np.mean(all_metrics['val_MCC']),
        'training_time_mean': np.mean(all_metrics['training_time']),
        'training_time_total': np.sum(all_metrics['training_time'])
    }

    # Log final aggregate metrics to wandb
    if wandb.run is not None:
        table_data_df = pd.DataFrame(all_metrics)
        metrics_table = wandb.Table(dataframe=table_data_df)
        wandb.log({"experiment_summary_table": metrics_table})

        wandb.finish()

    print_section("Training Complete")
    print(
        f"Final balanced accuracy: {best_metrics['val_balanced_accuracy']:.4f} ± {best_metrics['val_balanced_accuracy_std']:.4f}")

    return best_metrics


def save_results(config, best_metrics, config_setup_name):
    import json
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
