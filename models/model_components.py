import torch
import torch.nn as nn
import torch.nn.functional as F

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

class SaveBestModelCallback:
    def __init__(self, save_path, target='loss', mode='min'):
        self.best_value = None
        self.save_path = save_path
        self.target = target
        self.mode = mode
        self.best_epoch = 0

    def __call__(self, model, current_value, current_epoch):
        if self.best_value is None or \
          (self.mode == 'min' and current_value < self.best_value) or \
          (self.mode == 'max' and current_value > self.best_value):
            self.best_value = current_value
            self.best_epoch = current_epoch
            self._save_model(model)
            print(f"New best {self.target}: {self.best_value} at epoch {self.best_epoch}. Model saved.")
        else:
            print(f"Epoch {current_epoch}: {self.target} did not improve from {self.best_value}.")

    def _save_model(self, model):
        torch.save(model.state_dict(), self.save_path)

    def load_best_model(self, model):
        model.load_state_dict(torch.load(self.save_path))
        print(f"Loaded best model weights from {self.save_path} from epoch {self.best_epoch}.")

class Attention(nn.Module):
    '''
    Apply soft-attention by projecting embeddings through softmax(tanh(dim_projections))
    '''
    def __init__(self, feature_dim, step_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x):
        projection = self.proj(x)
        projection = torch.tanh(projection)
        att = torch.exp(projection) # softmax pt.1
        att = att / (torch.sum(att, dim=1, keepdim=True) + 1e-10) # softmax pt.2
        weighted_input = x * att
        return weighted_input, att

class MLP_MultiLabel(nn.Module):
    def __init__(self, n_features, n_classes):
        super(MLP_MultiLabel, self).__init__()
        self.input_layer = nn.Linear(n_features, 64)
        self.hidden_layer1 = nn.Linear(64, 32)
        self.attention = Attention(32, n_features)
        self.hidden_layer2 = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, n_classes)

    def forward(self, model_input, return_embedding=False):
        x = F.relu(self.input_layer(model_input))
        x = F.dropout(x, p=0.5)
        x = F.relu(self.hidden_layer1(x))
        x = F.dropout(x, p=0.5)
        x = self.hidden_layer2(x)
        embedding = F.relu(x)
        if return_embedding:
            return embedding
        x = self.output_layer(embedding)
        return x