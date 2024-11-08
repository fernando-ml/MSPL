import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import multi_label_balanced_accuracy
from models.model_components import SaveBestModelCallback
import random
from sklearn.metrics import  precision_score, recall_score, f1_score

from config import *

def z_score_normalize_with_clipping(embeddings, clip_range=3.0):
    mean = embeddings.mean(dim=0, keepdim=True)
    std = embeddings.std(dim=0, keepdim=True)
    std = torch.clamp(std, min=1e-8)  # avoid division by zero
    
    z_scores = (embeddings - mean) / std
    clipped_z_scores = torch.clamp(z_scores, min=-clip_range, max=clip_range)
    
    return clipped_z_scores

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
                selected_indices = random.sample(class_indices[class_label], n_support + n_query)
            elif len(class_indices[class_label]) > 0:
                # if not enough samples, repeat the indices until we reach the required number
                selected_indices = random.sample(class_indices[class_label] * ((n_support + n_query) // len(class_indices[class_label]) + 1), n_support + n_query)
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

def compute_prototypes(embeddings, labels):
    prototypes = []
    for i in range(labels.shape[1]):
        class_embeddings = embeddings[labels[:, i] == 1]
        if len(class_embeddings) > 0:
            prototypes.append(class_embeddings.mean(0))
        else:
            prototypes.append(torch.zeros(embeddings.shape[1], device=embeddings.device))
    return torch.stack(prototypes)

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

def calculate_distances(query_embeddings, prototypes, weights):
    """
    Calculate distances between query embeddings and prototypes.
    """
    euclidean_distances = torch.cdist(query_embeddings, prototypes)
    cosine_distances = 1 - cosine_similarity(query_embeddings, prototypes)
    chebyshev_distances = chebyshev_distance(query_embeddings, prototypes)
    wasserstein_distances = wasserstein_distance_torch(query_embeddings.unsqueeze(1), prototypes.unsqueeze(0))
    # Normalize distances
    normalized_euclidean_distances = z_score_normalize_with_clipping(euclidean_distances)
    normalized_cosine_distances = z_score_normalize_with_clipping(cosine_distances)
    normalized_chebyshev_distances = z_score_normalize_with_clipping(chebyshev_distances)
    normalized_wasserstein_distances = z_score_normalize_with_clipping(wasserstein_distances)

    combined_distances = (weights['euclidean'] * normalized_euclidean_distances) +\
                        (weights['cosine'] * normalized_cosine_distances) +\
                        (weights['chebyshev'] * normalized_chebyshev_distances) +\
                        (weights['wasserstein'] * normalized_wasserstein_distances)
    return combined_distances

def improved_hybrid_prototype_loss(query_embeddings, prototypes, query_labels, weights):
    combined_distances = calculate_distances(query_embeddings, prototypes, weights)
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

            combined_distances = calculate_distances(embeddings, prototypes, weights)
            predictions = torch.sigmoid(-combined_distances)

            loss = improved_hybrid_prototype_loss(embeddings, prototypes, targets, weights)
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
    f1 = f1_score(all_labels.cpu().numpy(), binary_predictions.cpu().numpy(), average='micro')
    precision = precision_score(all_labels.cpu().numpy(), binary_predictions.cpu().numpy(), average='micro')
    recall = recall_score(all_labels.cpu().numpy(), binary_predictions.cpu().numpy(), average='micro')
    
    balanced_acc, classes_acc = multi_label_balanced_accuracy(all_labels.cpu().numpy(), binary_predictions.cpu().numpy())

    # calculate average loss
    avg_loss = total_loss / len(dataloader)

    return balanced_acc, avg_loss, f1, precision, recall, binary_predictions, classes_acc

def multi_space_episodic_training_with_polyak(model, optimizer, epochs, episodes, val_dataloader, weights, polyak, polyak_decay=0.999):
    history = {
        "epochs": [], "loss": [], "val_loss": [], "balanced_accuracy": [],
        "val_f1": [], "val_precision": [], "val_recall": [], 'val_classes_acc': []
    }
    save_best_model_cb = SaveBestModelCallback(save_path=best_model_path, target='val_f1', mode='max')
    
    if polyak:
        ema_model = type(model)(model.input_layer.in_features, model.output_layer.out_features)
        ema_model.load_state_dict(model.state_dict())
    
    for epoch in range(epochs):
        total_loss = 0
        for episode_index, episode in enumerate(episodes):
            model.train()
            if polyak:
                ema_model.eval()

            support_set, query_set = episode
            support_embeddings = model(support_set[0], return_embedding=True)
            query_embeddings = model(query_set[0], return_embedding=True)

            prototypes = compute_prototypes(support_embeddings, support_set[1])

            loss = improved_hybrid_prototype_loss(query_embeddings, prototypes, query_set[1], weights)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # Clip gradients
            optimizer.step()

            total_loss += loss.item()

            if polyak:
                # update the exponential moving average (EMA) model.
                with torch.no_grad():
                    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                        ema_param.data.mul_(polyak_decay).add_(param.data, alpha=1 - polyak_decay)

        avg_train_loss = total_loss / len(episodes)
        if polyak:
            # Use the EMA model for validation
            balanced_acc, val_loss, val_f1, val_precision, val_recall, val_predictions, classes_acc = new_validate_with_prototypes(ema_model, val_dataloader, weights)
            save_best_model_cb(ema_model, balanced_acc, epoch)
        else:
            balanced_acc, val_loss, val_f1, val_precision, val_recall, val_predictions, classes_acc = new_validate_with_prototypes(model, val_dataloader, weights)
            save_best_model_cb(model, val_f1, epoch)
        
        print(f"Epoch {epoch}")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Balanced Accuracy: {balanced_acc:.4f}")
        
        history["epochs"].append(epoch)
        history["loss"].append(avg_train_loss)
        history["val_loss"].append(val_loss)
        history["balanced_accuracy"].append(balanced_acc)
        history["val_f1"].append(val_f1)
        history["val_precision"].append(val_precision)
        history["val_recall"].append(val_recall)
        history["val_classes_acc"].append(classes_acc)
    if polyak:
        save_best_model_cb.load_best_model(model=ema_model)
    else:
        save_best_model_cb.load_best_model(model=model)
    return history