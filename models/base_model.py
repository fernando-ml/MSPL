"""
Base model module containing abstract classes and shared model functionality.

This module provides base classes for models used in the DOS project,
ensuring consistent interfaces and shared functionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    """
    Base class for all neural network models in the project.
    
    Provides common functionality and ensures consistent interfaces.
    """
    
    def __init__(self):
        super(BaseModel, self).__init__()
        
    def get_name(self):
        """
        Get the model name for reporting and saving.
        
        Returns:
            str: Model name
        """
        return self.__class__.__name__
    
    def count_parameters(self):
        """
        Count the number of trainable parameters in the model.
        
        Returns:
            int: Number of trainable parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def summary(self):
        """
        Print a summary of the model architecture.
        
        Returns:
            str: Model summary
        """
        summary_str = f"Model: {self.get_name()}\n"
        summary_str += f"Trainable parameters: {self.count_parameters():,}\n"
        summary_str += "Architecture:\n"
        for name, module in self.named_children():
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            summary_str += f"  {name}: {module} ({params:,} parameters)\n"
        return summary_str


class EarlyStopping:
    """
    Early stopping handler to prevent overfitting during training.
    
    Monitors validation loss and signals when to stop training.
    """
    
    def __init__(self, patience=5, delta=0):
        """
        Initialize early stopping handler.
        
        Args:
            patience (int): Number of epochs to wait after last improvement
            delta (float): Minimum change to qualify as improvement
        """
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        """
        Check if training should stop based on validation loss.
        
        Args:
            val_loss (float): Current validation loss
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        
        return self.early_stop


class SaveBestModelCallback:
    """
    Callback to save the best model during training based on a metric.
    
    Monitors a specified metric and saves the model when it improves.
    """
    
    def __init__(self, save_path, target='loss', mode='min'):
        """
        Initialize the callback.
        
        Args:
            save_path (str): Path to save the best model
            target (str): Target metric to monitor
            mode (str): 'min' if lower is better, 'max' if higher is better
        """
        self.best_value = None
        self.save_path = save_path
        self.target = target
        self.mode = mode
        self.best_epoch = 0

    def __call__(self, model, current_value, current_epoch):
        """
        Check and save model if it's the best so far.
        
        Args:
            model (nn.Module): Model to save
            current_value (float): Current value of the monitored metric
            current_epoch (int): Current epoch number
        """
        is_better = False
        
        if self.best_value is None:
            is_better = True
        elif self.mode == 'min' and current_value < self.best_value:
            is_better = True
        elif self.mode == 'max' and current_value > self.best_value:
            is_better = True
            
        if is_better:
            self.best_value = current_value
            self.best_epoch = current_epoch
            self._save_model(model)
            print(f"New best {self.target}: {self.best_value:.4f} at epoch {self.best_epoch}. Model saved.")
        else:
            print(f"Epoch {current_epoch}: {self.target} did not improve from {self.best_value:.4f}.")

    def _save_model(self, model):
        """
        Save the model to disk.
        
        Args:
            model (nn.Module): Model to save
        """
        torch.save(model.state_dict(), self.save_path)

    def load_best_model(self, model):
        """
        Load the best model from disk.
        
        Args:
            model (nn.Module): Model to load weights into
        """
        model.load_state_dict(torch.load(self.save_path))
        print(f"Loaded best model weights from {self.save_path} from epoch {self.best_epoch}.")
