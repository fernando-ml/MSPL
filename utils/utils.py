import torch
import numpy as np
import yaml

def device():   
    """ 
    Returns the device to be used for PyTorch tensor operations.
    Checks in order:
    1. Metal Performance Shader (MPS)
    2. CUDA
    3. CPU

    Returns:
        str: The device to use
    """
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device

def stratified_sample(datasets: tuple, n_samples: int, sample_per_class: int, random_state: int = None) -> tuple:
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
    unique_labels = y.unique()

    # Sample 'sample_per_class' instances from each label
    indices_per_class = []
    for label in unique_labels:
        label_indices = torch.where(y == label)[0]
        sampled_indices = label_indices[torch.randperm(len(label_indices))[:sample_per_class]]
        indices_per_class.extend(sampled_indices.tolist())
    
    # Combine the sampled indices into a tensor
    indices_per_class = torch.tensor(indices_per_class)

    # Calculate the remaining samples needed
    remaining_samples = n_samples - len(indices_per_class)

    # Get all indices that were not sampled initially
    remaining_indices = list(set(range(len(X))) - set(indices_per_class.tolist()))

    # Randomly sample the remaining indices
    remaining_indices_sample = np.random.choice(remaining_indices, remaining_samples, replace=False)

    # Combine the initial and random samples indices
    final_indices = torch.cat([indices_per_class, torch.tensor(remaining_indices_sample)])

    # Shuffle the final indices
    final_indices = final_indices[torch.randperm(len(final_indices))]

    # Sample the data using the final indices
    X_sampled = X[final_indices, :]
    y_sampled = y[final_indices]

    return X_sampled, y_sampled

def multi_label_balanced_accuracy(y_true, y_pred):
    """
    Calculate balanced accuracy for multi-label classification.
    
    :param y_true: True labels (n_samples, n_classes)
    :param y_pred: Predicted labels (n_samples, n_classes)
    :return: Balanced accuracy score
    """
    y_true = np.array(y_true)  # Convert to numpy array
    y_pred = np.array(y_pred)  # Convert to numpy array
    
    n_classes = y_true.shape[1]
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