import torch
from torch import nn
from torch.nn import functional as F
from evaluation import *
import random
def cosine_similarity(x1, x2):
    # Normalize input vectors to compute cosine similarity
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    return torch.mm(x1_norm, x2_norm.transpose(0, 1))

def hybrid_prototype_loss(query_embeddings, prototypes, query_labels, alpha=1):
    """
    Calculate the Prototype Loss.
    """
    # Compute distances and apply softmax
    euclidean_distances = F.normalize(torch.cdist(query_embeddings, prototypes), p=2, dim=1)
    cosine_distances = 1 - cosine_similarity(query_embeddings, prototypes)
    combined_distances = (alpha * euclidean_distances) + ((1 - alpha) * cosine_distances)
    predictions = F.log_softmax(-combined_distances, dim=1)
    # Cross-entropy loss expects class indices
    criterion = nn.CrossEntropyLoss()
    return criterion(predictions, query_labels.long())

def create_episodes(X, y, n_episodes, n_support, n_query):
    episodes = []
    class_indices = {0: [], 1: []}
    for idx, label in enumerate(y):
        class_indices[label.item()].append(idx)
    
    for _ in range(n_episodes):
        support_indices = []
        query_indices = []
        
        for class_label in [0, 1]:
            selected_indices = random.sample(class_indices[class_label], n_support + n_query)
            support_indices.extend(selected_indices[:n_support])
            query_indices.extend(selected_indices[n_support:])
        
        support_set = (X[support_indices], y[support_indices])
        query_set = (X[query_indices], y[query_indices])
        episodes.append((support_set, query_set))
    return episodes

def compute_prototypes(embeddings, labels):
    """
    Compute class prototypes from embeddings
    """
    classes = torch.unique(labels)
    prototypes = torch.stack([embeddings[labels == c].mean(0) for c in classes])
    return prototypes

def validate_with_prototypes(model, dataloader, alpha=0):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            emb = model(inputs, return_embedding=True)
            embeddings.append(emb)
            labels.append(targets)

    embeddings = torch.cat(embeddings)
    labels = torch.cat(labels)

    # Compute prototypes
    prototypes = compute_prototypes(embeddings, labels)

    # Classify based on nearest prototype
    distances_prot = F.normalize(torch.cdist(embeddings, prototypes), p=2, dim=1)
    cosine_distances = 1 - cosine_similarity(embeddings, prototypes)
    combined_distances = (alpha * distances_prot) + ((1 - alpha) * cosine_distances)
    predictions = torch.argmin(combined_distances, dim=1)
    val_loss = hybrid_prototype_loss(embeddings, prototypes, labels)
    accuracy = accuracy_score(labels.numpy(), predictions.numpy())
    return accuracy, val_loss