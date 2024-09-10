import torch
import torch.nn.functional as F

def chebyshev_distance(x1, x2):
    # Compute the difference between each pair of vectors
    diff = x1.unsqueeze(1) - x2.unsqueeze(0)
    
    # Take the absolute value of the differences
    abs_diff = torch.abs(diff)
    
    # Compute the maximum along the feature dimension
    distances = torch.max(abs_diff, dim=-1).values
    
    return distances

# Example usage
x1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float)
x2 = torch.tensor([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=torch.float)

distances = chebyshev_distance(x1, x2)
print("Chebyshev distances:")
print(distances)

# For comparison, let's also compute cosine similarity
def cosine_similarity(x1, x2):
    x1_norm = F.normalize(x1, p=2, dim=1)
    x2_norm = F.normalize(x2, p=2, dim=1)
    return torch.mm(x1_norm, x2_norm.transpose(0, 1))

similarities = cosine_similarity(x1, x2)
print("\nCosine similarities:")
print(similarities)