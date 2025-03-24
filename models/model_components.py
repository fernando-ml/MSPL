"""
Neural network model components for the DOS project.

This module contains neural network models and components used for
classification tasks in the DOS project.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_model import BaseModel


class Attention(nn.Module):
    """
    Attention module for focusing on important features.
    
    Applies soft-attention by projecting embeddings through softmax(tanh(dim_projections)).
    """
    
    def __init__(self, feature_dim, step_dim):
        """
        Initialize the attention module.
        
        Args:
            feature_dim (int): Dimension of input features
            step_dim (int): Dimension for projection
        """
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x):
        """
        Apply attention to input features.
        
        Args:
            x (torch.Tensor): Input features [batch_size, feature_dim]
            
        Returns:
            tuple: (weighted_input, attention_weights)
        """
        projection = self.proj(x)
        projection = torch.tanh(projection)
        # Compute attention weights (2-part softmax for numerical stability)
        att = torch.exp(projection) 
        att = att / (torch.sum(att, dim=1, keepdim=True) + 1e-10)
        weighted_input = x * att
        return weighted_input, att


class MLP_MultiLabel(BaseModel):
    """
    Multi-layer perceptron for multi-label classification.
    
    A simple feedforward neural network with attention mechanism.
    """
    
    def __init__(self, n_features, n_classes, dropout_rate=0.5):
        """
        Initialize the MLP model.
        
        Args:
            n_features (int): Number of input features
            n_classes (int): Number of output classes
            dropout_rate (float): Dropout rate for regularization
        """
        super(MLP_MultiLabel, self).__init__()
        self.dropout_rate = dropout_rate
        
        # Network architecture
        self.input_layer = nn.Linear(n_features, 64)
        self.hidden_layer1 = nn.Linear(64, 32)
        self.attention = Attention(32, n_features)
        self.hidden_layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, n_classes)

    def forward(self, model_input, return_embedding=False):
        """
        Forward pass through the network.
        
        Args:
            model_input (torch.Tensor): Input features [batch_size, n_features]
            return_embedding (bool): Whether to return the embedding before final layer
            
        Returns:
            torch.Tensor: Network output or embedding if return_embedding is True
        """
        # Input layer with ReLU activation and dropout
        x = F.relu(self.input_layer(model_input))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # First hidden layer with ReLU activation and dropout
        x = F.relu(self.hidden_layer1(x))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Second hidden layer
        x = self.hidden_layer2(x)
        embedding = F.relu(x)
        
        # Return embedding if requested
        if return_embedding:
            return embedding
            
        # Output layer
        x = self.output_layer(embedding)
        return x