"""
Metrics module containing evaluation functions for model assessment.

This module provides various metrics for evaluating model performance,
particularly for multi-label classification tasks.
"""

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, confusion_matrix
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Union, Tuple, Any, Optional


def calculate_multi_label_balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy for multi-label classification.
    
    Args:
        y_true (np.ndarray): True labels, either one-hot encoded (n_samples, n_classes) 
                             or single class labels (n_samples,)
        y_pred (np.ndarray): Predicted labels, either one-hot encoded (n_samples, n_classes) 
                             or single class labels (n_samples,)
    
    Returns:
        float: Balanced accuracy score
    """
    return multi_label_balanced_accuracy(y_true, y_pred)[0]


def multi_label_balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy for multi-label classification.
    
    Args:
        y_true (np.ndarray): True labels, either one-hot encoded (n_samples, n_classes) 
                             or single class labels (n_samples,)
        y_pred (np.ndarray): Predicted labels, either one-hot encoded (n_samples, n_classes) 
                             or single class labels (n_samples,)
    
    Returns:
        tuple: (float, list) Balanced accuracy score and list of per-class accuracies
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Determine the number of classes from y_true if possible, otherwise use y_pred
    n_classes = y_true.shape[1] if len(y_true.shape) > 1 else len(np.unique(np.concatenate((y_true, y_pred))))

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

        sensitivity = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0
        specificity = true_neg / (true_neg + false_pos) if (true_neg + false_pos) > 0 else 0

        class_accuracies.append((sensitivity + specificity) / 2)

    return np.mean(class_accuracies), class_accuracies


def calculate_metrics(y_true: np.ndarray, 
                     y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for classification tasks.
    
    Args:
        y_true: True labels (multi-label or single-label)
        y_pred: Predicted labels (multi-label or single-label)
        y_prob: Optional probability predictions for AUPRC calculation
        
    Returns:
        Dictionary of metrics including accuracy, precision, recall, F1, MCC, AUPRC
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Handle multi-label vs single-label
    multi_label = len(y_true.shape) > 1 and y_true.shape[1] > 1
    
    if multi_label:
        # For multi-label, we'll calculate metrics in two ways:
        # 1. Treat each label as binary classification and average (macro)
        # 2. For metrics requiring single label, use argmax
        y_true_single = np.argmax(y_true, axis=1)
        y_pred_single = np.argmax(y_pred, axis=1)
    else:
        # For single-label
        y_true_single = y_true
        y_pred_single = y_pred
    
    # Calculate metrics
    metrics = {}
    
    # Balanced accuracy
    metrics['balanced_accuracy'], class_accuracies = multi_label_balanced_accuracy(y_true, y_pred)
    
    # F1, precision, recall (macro and micro)
    metrics['f1_macro'] = f1_score(y_true_single, y_pred_single, average='macro')
    metrics['f1_micro'] = f1_score(y_true_single, y_pred_single, average='micro')
    metrics['precision_macro'] = precision_score(y_true_single, y_pred_single, average='macro')
    metrics['precision_micro'] = precision_score(y_true_single, y_pred_single, average='micro')
    metrics['recall_macro'] = recall_score(y_true_single, y_pred_single, average='macro')
    metrics['recall_micro'] = recall_score(y_true_single, y_pred_single, average='micro')
    
    # Matthews Correlation Coefficient
    metrics['mcc'] = matthews_corrcoef(y_true_single, y_pred_single)
    
    # Simple accuracy
    metrics['accuracy'] = np.mean(y_true_single == y_pred_single)
    
    # Add AUPRC if probabilities are provided, else use predictions
    n_classes = y_true.shape[1] if len(y_true.shape) > 1 else len(np.unique(np.concatenate((y_true_single, y_pred_single))))
    
    # Prepare binary versions for AUPRC
    if len(y_true.shape) == 1:
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
    else:
        y_true_bin = y_true
    
    if y_prob is not None:
        # Use probabilities if provided
        y_score = y_prob
    else:
        # Use predictions otherwise
        if len(y_pred.shape) == 1:
            y_score = label_binarize(y_pred, classes=np.arange(n_classes))
        else:
            y_score = y_pred
    
    try:
        metrics['auprc'] = average_precision_score(y_true_bin, y_score, average='macro')
    except (ValueError, Exception) as e:
        # Fallback if there's an issue with calculation
        metrics['auprc'] = 0.0
        print(f"Error calculating AUPRC: {e}")
    
    return metrics


def calculate_per_class_metrics(y_true: np.ndarray, 
                               y_pred: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate per-class metrics for each class in the dataset.
    
    Args:
        y_true: True labels (multi-label or single-label)
        y_pred: Predicted labels (multi-label or single-label)
        
    Returns:
        Dictionary with keys 'precision', 'recall', 'f1', 'accuracy'
        each containing an array of per-class metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Determine if multi-label and get number of classes
    multi_label = len(y_true.shape) > 1 and y_true.shape[1] > 1
    
    if multi_label:
        n_classes = y_true.shape[1]
    else:
        # For single-label, determine number of classes
        unique_classes = np.unique(np.concatenate((y_true, y_pred)))
        n_classes = len(unique_classes)
        
        # Convert to one-hot for per-class metrics
        y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        y_pred_bin = label_binarize(y_pred, classes=np.arange(n_classes))
    
    # Initialize arrays for per-class metrics
    precision = np.zeros(n_classes)
    recall = np.zeros(n_classes)
    f1 = np.zeros(n_classes)
    accuracy = np.zeros(n_classes)
    
    # Calculate metrics for each class
    for i in range(n_classes):
        if multi_label:
            y_true_class = y_true[:, i]
            y_pred_class = y_pred[:, i]
        else:
            y_true_class = y_true_bin[:, i]
            y_pred_class = y_pred_bin[:, i]
        
        # Calculate confusion matrix values
        tp = np.sum((y_true_class == 1) & (y_pred_class == 1))
        tn = np.sum((y_true_class == 0) & (y_pred_class == 0))
        fp = np.sum((y_true_class == 0) & (y_pred_class == 1))
        fn = np.sum((y_true_class == 1) & (y_pred_class == 0))
        
        # Calculate metrics
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
        accuracy[i] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def calculate_all_metrics(y_true, y_pred, y_prob=None):
    """
    Calculate multiple classification metrics.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
        y_prob (np.ndarray, optional): Prediction probabilities for AUPRC calculation
        
    Returns:
        dict: Dictionary containing multiple metrics
    """
    if len(y_true.shape) > 1:
        # For multi-label data
        y_true_single = np.argmax(y_true, axis=1)
        y_pred_single = np.argmax(y_pred, axis=1)
    else:
        y_true_single = y_true
        y_pred_single = y_pred
        
    balanced_acc, class_acc = multi_label_balanced_accuracy(y_true, y_pred)
    
    metrics = {
        "balanced_accuracy": balanced_acc,
        "class_accuracies": class_acc,
        "f1": f1_score(y_true_single, y_pred_single, average='macro'),
        "precision": precision_score(y_true_single, y_pred_single, average='macro'),
        "recall": recall_score(y_true_single, y_pred_single, average='macro'),
        "mcc": matthews_corrcoef(y_true_single, y_pred_single)
    }
    
    # Add AUPRC if probabilities are provided
    if y_prob is not None:
        n_classes = y_true.shape[1] if len(y_true.shape) > 1 else len(np.unique(y_true))
        
        if len(y_true.shape) == 1:
            y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))
        else:
            y_true_bin = y_true
            
        if len(y_prob.shape) == 1:
            y_prob_bin = label_binarize(y_prob, classes=np.arange(n_classes))
        else:
            y_prob_bin = y_prob
            
        metrics["auprc"] = average_precision_score(y_true_bin, y_prob_bin, average='macro')
    
    return metrics
