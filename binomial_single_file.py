import numpy as np
import pandas as pd
import random
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score


if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"    
else:
    device = "cpu"

random_seed = 11

# read parquet data
train_data, validation_data, test_data = pd.read_parquet('data/train_set.parquet'), pd.read_parquet('data/crossval_set.parquet'), pd.read_parquet('data/test_set.parquet')

try:
    columns_to_drop = ["flag_urg", "fwd_flag_urg", "bwd_flag_urg", "flow_id","timestamp",
                    "src_addr", "dst_addr"]
    train_data.drop(columns_to_drop, axis=1, inplace=True)
    validation_data.drop(columns_to_drop, axis=1, inplace=True)
    test_data.drop(columns_to_drop, axis=1, inplace=True)
except:
    pass

# Turn the problem into a binary problem (Beneign VS DoS Attack)
X_train_data, y_train_data = train_data.drop("label", axis=1), train_data["label"]
y_train_data = y_train_data.map(lambda x: 1 if x != "benign" else 0)

X_validation_data, y_validation_data = validation_data.drop("label", axis=1), validation_data["label"]
y_validation_data = y_validation_data.map(lambda x: 1 if x != "benign" else 0)

X_test_data, y_test_data = test_data.drop("label", axis=1), test_data["label"]
y_test_data = y_test_data.map(lambda x: 1 if x != "benign" else 0)

RFC_pipeline = Pipeline([("Scaler", sklearn.preprocessing.RobustScaler()), ("RFC", RandomForestClassifier(random_state=random_seed))])
RFC_pipeline.fit(X_train_data, y_train_data)
ftr_importance_RFC = pd.DataFrame(zip(X_train_data.columns, RFC_pipeline["RFC"].feature_importances_))
ftr_importance_RFC.columns = ["Column", "Weight"]
ftr_importance_RFC.sort_values("Weight", ascending=False, inplace=True)
features_RFC = ftr_importance_RFC[ftr_importance_RFC.Weight >= ftr_importance_RFC.Weight.mean()].Column.values

print(f"# RFC Features: {len(features_RFC)}")

class EarlyStopping:
    '''
    An early stopper is a technique in ML where training is halted if a model's performance on a validation dataset stops 
    improving, helping prevent overfitting and save training time. It monitors a specified metric and stops training when the metric plateaus or degrades.
    '''
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

def binary_accuracy(preds, y):
    rounded_preds = torch.round(preds)
    correct = (rounded_preds == y).float() 
    acc = correct.sum() / len(correct)
    return acc

def evaluate_model(model, X, y):
    """
    Function to evaluate classification metrics such as accuracy, precision, recall, and f-1 score
    """
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    model.eval()  # Set the model to evaluation mode

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Make predictions
    with torch.inference_mode():
        outputs = model(X_tensor)
        predicted = (outputs.squeeze() >= 0.5).float() #binary
    predicted_np = predicted.numpy()
    y_np = y_tensor.numpy()

    # Calculate metrics
    accuracy = accuracy_score(y_np, predicted_np)
    precision = precision_score(y_np, predicted_np)
    recall = recall_score(y_np, predicted_np)
    f1 = f1_score(y_np, predicted_np)

    return {'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1}

scaler = sklearn.preprocessing.RobustScaler()

X_train_data_scaled = scaler.fit_transform(X_train_data[features_RFC].values)
X_val_data_scaled = scaler.transform(X_validation_data[features_RFC].values)

X_train = torch.tensor(X_train_data_scaled, dtype=torch.float32)
y_train = torch.tensor(y_train_data.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

X_val = torch.tensor(X_val_data_scaled, dtype=torch.float32)
y_val = torch.tensor(y_validation_data.values, dtype=torch.float32)
val_dataset = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)


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
        """
        Loads the best model weights from the saved path.
        """
        model.load_state_dict(torch.load(self.save_path))
        print(f"Loaded best model weights from {self.save_path} from epoch {self.best_epoch}.")

class Attention(nn.Module):
    '''
    Apply soft-attention by projecting embeddings through softmax(tanh(dim_projections))
    '''
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.proj = nn.Linear(feature_dim, 1, bias=False)

    def forward(self, x):
        projection = self.proj(x)
        projection = torch.tanh(projection)
        att = torch.exp(projection) # softmax pt.1
        att = att / (torch.sum(att, dim=1, keepdim=True) + 1e-10) # softmax pt.2
        weighted_input = x * att
        return weighted_input, att # attention output, attention_weights

class MLP_Binary(nn.Module):
    def __init__(self, n_features):
        super(MLP_Binary, self).__init__()
        self.input_layer = nn.Linear(n_features, 32)
        self.hidden_layer = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, model_input, return_embedding=False):
        x = F.relu(self.input_layer(model_input))
        x = F.dropout(x)
        embedding = F.relu(self.hidden_layer(x))
        if return_embedding:
            return embedding
        # x = torch.sigmoid(self.output_layer(embedding))
        return x

class Binary_Att(nn.Module):
    def __init__(self, n_features):
        super(Binary_Att, self).__init__()
        self.input_layer = nn.Linear(n_features, 32)
        self.attention = Attention(32, n_features)
        self.after_attn = nn.Linear(32, 16)
        self.output_layer = nn.Linear(16, 1)

    def forward(self, model_input, return_embedding=False):
        x = F.relu(self.input_layer(model_input))
        x, att_weights = self.attention(x)
        embedding = F.relu(self.after_attn(x))
        if return_embedding:
            return embedding
        x = torch.sigmoid(self.output_layer(embedding))
        return x

def create_episodes(X, y, n_episodes, n_support, n_query):
    episodes = []
    class_indices = {0: [], 1: []}
    all_indices = set()  # Track all used indices

    # Separate indices for each class
    for idx, label in enumerate(y):
        class_indices[label.item()].append(idx)

    for _ in range(n_episodes):
        support_indices = []
        query_indices = []

        for class_label in [0, 1]:
            selected_indices = random.sample(class_indices[class_label], n_support + n_query)
            all_indices.update(selected_indices)  # Update the set with new indices
            support_indices.extend(selected_indices[:n_support])
            query_indices.extend(selected_indices[n_support:])

        support_set = (X[support_indices], y[support_indices])
        query_set = (X[query_indices], y[query_indices])
        episodes.append((support_set, query_set))

    return episodes, all_indices

def compute_prototypes(embeddings, labels):
    """
    Compute class prototypes from embeddings
    """
    classes = torch.unique(labels)
    prototypes = torch.stack([embeddings[labels == c].mean(0) for c in classes])
    return prototypes

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

# Prototypes
def validate_with_prototypes(model, dataloader, alpha=1):
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

def episodic_training(model, optimizer, epochs, episodes, val_dataloader):
    history = {}
    history["epochs"], history["loss"], history["val_loss"] = [], [], []
    save_best_model_cb = SaveBestModelCallback(save_path='best_model.pth', target='loss', mode='min')
    for epoch in range(epochs):
        total_loss = 0
        for episode_index, episode in enumerate(episodes):
            model.train()
            
            # Extract support and query sets
            support_set, query_set = episode
            support_embeddings = model(support_set[0], return_embedding=True)
            query_embeddings = model(query_set[0], return_embedding=True)

            # Compute prototypes
            prototypes = compute_prototypes(support_embeddings, support_set[1])

            # Calculate loss using the prototype loss function
            loss = hybrid_prototype_loss(query_embeddings, prototypes, query_set[1])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_accuracy, val_loss = validate_with_prototypes(model, val_dataloader)
        save_best_model_cb(model, val_loss, epoch)
        print(f"Epoch {epoch}, Training Loss: {total_loss / len(episodes)}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
        history["epochs"].append(epoch)
        history["loss"].append(total_loss / len(episodes))
        history["val_loss"].append(val_loss)
    save_best_model_cb.load_best_model(model=model)
    return history


samples_training = torch.randperm(X_train.shape[0])[:100]

# Define your model
Prototypical_MLP = MLP_Binary(len(features_RFC))  # or Binary_Att(n_features)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(Prototypical_MLP.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()  # or another suitable loss function

n_episodes = 70  # for example
n_support = 10   # number of support samples per class
n_query = 10     # number of query samples per class
episodes, unique_indices = create_episodes(X_train[samples_training,:], y_train[samples_training], n_episodes, n_support, n_query)#create_episodes(X_train, y_train, n_episodes, n_support, n_query)
print(f"Total unique inputs used for training: {len(unique_indices)}")

history_mlp_prototypical = episodic_training(Prototypical_MLP, optimizer, 10, episodes, val_dataloader)


# BEGIN: AdaptativeMulti-Modality ZeroDay DoS

# import numpy as np
# import pandas as pd
# import random
# import sklearn
# from sklearn.pipeline import Pipeline
# from sklearn.feature_selection import RFECV
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# import torch
# import torch.nn.functional as F
# from torch import nn
# from torch.utils.data import TensorDataset, DataLoader
# from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

# if torch.backends.mps.is_available():
#     device = "mps"
# elif torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

# random_seed = 16

# # Read parquet data
# train_data, validation_data, test_data = pd.read_parquet('data/train_set.parquet'), pd.read_parquet('data/crossval_set.parquet'), pd.read_parquet('data/test_set.parquet')

# try:
#     columns_to_drop = ["flag_urg", "fwd_flag_urg", "bwd_flag_urg", "flow_id", "timestamp", "src_addr", "dst_addr"]
#     train_data.drop(columns_to_drop, axis=1, inplace=True)
#     validation_data.drop(columns_to_drop, axis=1, inplace=True)
#     test_data.drop(columns_to_drop, axis=1, inplace=True)
# except:
#     pass

# # Convert labels to one-hot encoding for multi-label classification
# y_train_data = pd.get_dummies(train_data['label'])
# y_validation_data = pd.get_dummies(validation_data['label'])
# y_test_data = pd.get_dummies(test_data['label'])

# X_train_data, y_train_data = train_data.drop("label", axis=1), y_train_data
# X_validation_data, y_validation_data = validation_data.drop("label", axis=1), y_validation_data
# X_test_data, y_test_data = test_data.drop("label", axis=1), y_test_data

# RFC_pipeline = Pipeline([("Scaler", sklearn.preprocessing.RobustScaler()), ("RFC", RandomForestClassifier(random_state=random_seed))])
# RFC_pipeline.fit(X_train_data, y_train_data)
# ftr_importance_RFC = pd.DataFrame(zip(X_train_data.columns, RFC_pipeline["RFC"].feature_importances_))
# ftr_importance_RFC.columns = ["Column", "Weight"]
# ftr_importance_RFC.sort_values("Weight", ascending=False, inplace=True)
# features_RFC = ftr_importance_RFC[ftr_importance_RFC.Weight >= ftr_importance_RFC.Weight.mean()].Column.values

# print(f"# RFC Features: {len(features_RFC)}")
# # # Test RFC
# # RFC_pipeline = Pipeline([("Scaler", sklearn.preprocessing.RobustScaler()), ("RFC", RandomForestClassifier(random_state=random_seed))])
# # RFC_pipeline.fit(X_train_data[features_RFC], y_train_data)

# # y_test_pred = RFC_pipeline.predict(X_test_data[features_RFC])
# # print(classification_report(y_test_data, y_test_pred))
# # print(f"Accuracy: {accuracy_score(y_test_data, y_test_pred)}")

# class EarlyStopping:
#     def __init__(self, patience=5, delta=0):
#         self.patience = patience
#         self.best_loss = None
#         self.counter = 0
#         self.early_stop = False
#         self.delta = delta

#     def __call__(self, val_loss):
#         if self.best_loss is None:
#             self.best_loss = val_loss
#         elif val_loss > self.best_loss + self.delta:
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         else:
#             self.best_loss = val_loss
#             self.counter = 0

# def binary_accuracy(preds, y):
#     rounded_preds = torch.round(preds)
#     correct = (rounded_preds == y).float()
#     acc = correct.sum() / len(correct)
#     return acc

# def evaluate_model(model, X, y):
#     model.eval()

#     X_tensor = torch.tensor(X, dtype=torch.float32)
#     y_tensor = torch.tensor(y.values, dtype=torch.float32)

#     with torch.no_grad():
#         outputs = model(X_tensor)
#         predicted = torch.round(torch.sigmoid(outputs))
#     predicted_np = predicted.numpy()
#     y_np = y_tensor.numpy()

#     accuracy = accuracy_score(y_np, predicted_np)
#     precision = precision_score(y_np, predicted_np, average='macro')
#     recall = recall_score(y_np, predicted_np, average='macro')
#     f1 = f1_score(y_np, predicted_np, average='macro')

#     return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1_score': f1}

# scaler = sklearn.preprocessing.RobustScaler()

# X_train_data_scaled = scaler.fit_transform(X_train_data[features_RFC].values)
# X_val_data_scaled = scaler.transform(X_validation_data[features_RFC].values)

# X_train = torch.tensor(X_train_data_scaled, dtype=torch.float32)
# y_train = torch.tensor(y_train_data.values, dtype=torch.float32)

# train_dataset = TensorDataset(X_train, y_train)
# train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# X_val = torch.tensor(X_val_data_scaled, dtype=torch.float32)
# y_val = torch.tensor(y_validation_data.values, dtype=torch.float32)
# val_dataset = TensorDataset(X_val, y_val)
# val_dataloader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# class SaveBestModelCallback:
#     def __init__(self, save_path, target='loss', mode='min'):
#         self.best_value = None
#         self.save_path = save_path
#         self.target = target
#         self.mode = mode
#         self.best_epoch = 0

#     def __call__(self, model, current_value, current_epoch):
#         if self.best_value is None or \
#           (self.mode == 'min' and current_value < self.best_value) or \
#           (self.mode == 'max' and current_value > self.best_value):
#             self.best_value = current_value
#             self.best_epoch = current_epoch
#             self._save_model(model)
#             print(f"New best {self.target}: {self.best_value} at epoch {self.best_epoch}. Model saved.")
#         else:
#             print(f"Epoch {current_epoch}: {self.target} did not improve from {self.best_value}.")

#     def _save_model(self, model):
#         torch.save(model.state_dict(), self.save_path)

#     def load_best_model(self, model):
#         model.load_state_dict(torch.load(self.save_path))
#         print(f"Loaded best model weights from {self.save_path} from epoch {self.best_epoch}.")

# class Attention(nn.Module):
#     def __init__(self, feature_dim):
#         super().__init__()
#         self.feature_dim = feature_dim
#         self.proj = nn.Linear(feature_dim, 1, bias=False)

#     def forward(self, x):
#         projection = self.proj(x)
#         projection = torch.tanh(projection)
#         att = torch.exp(projection)
#         att = att / (torch.sum(att, dim=1, keepdim=True) + 1e-10)
#         weighted_input = x * att
#         return weighted_input, att

# class MLP_MultiLabel(nn.Module):
#     def __init__(self, n_features, n_classes):
#         super(MLP_MultiLabel, self).__init__()
#         self.input_layer = nn.Linear(n_features, 32)
#         self.hidden_layer = nn.Linear(32, 16)
#         self.output_layer = nn.Linear(16, n_classes)

#     def forward(self, model_input, return_embedding=False):
#         x = F.relu(self.input_layer(model_input))
#         x = F.dropout(x)
#         embedding = F.relu(self.hidden_layer(x))
#         if return_embedding:
#             return embedding
#         x = torch.sigmoid(self.output_layer(embedding))
#         return x

# class MultiLabel_Att(nn.Module):
#     def __init__(self, n_features, n_classes):
#         super(MultiLabel_Att, self).__init__()
#         self.input_layer = nn.Linear(n_features, 32)
#         self.attention = Attention(32)
#         self.after_attn = nn.Linear(32, 16)
#         self.output_layer = nn.Linear(16, n_classes)

#     def forward(self, model_input, return_embedding=False):
#         x = F.relu(self.input_layer(model_input))
#         x, att_weights = self.attention(x)
#         embedding = F.relu(self.after_attn(x))
#         if return_embedding:
#             return embedding
#         x = torch.sigmoid(self.output_layer(embedding))
#         return x

# def create_episodes(X, y, n_episodes, n_support, n_query):
#     episodes = []
#     class_indices = {label: [] for label in range(y.shape[1])}
#     all_indices = set()

#     # Separate indices for each class
#     for idx, labels in enumerate(y):
#         for label in range(y.shape[1]):
#             if labels[label] == 1:
#                 class_indices[label].append(idx)

#     for _ in range(n_episodes):
#         support_indices = []
#         query_indices = []

#         for class_label in range(y.shape[1]):
#             if len(class_indices[class_label]) >= n_support + n_query:
#                 selected_indices = random.sample(class_indices[class_label], n_support + n_query)
#             elif len(class_indices[class_label]) > 0:
#                 # If not enough samples, repeat the indices until we reach the required number
#                 selected_indices = random.sample(class_indices[class_label] * ((n_support + n_query) // len(class_indices[class_label]) + 1), n_support + n_query)
#             else:
#                 # Skip this class if there are no samples available
#                 continue

#             all_indices.update(selected_indices)
#             support_indices.extend(selected_indices[:n_support])
#             query_indices.extend(selected_indices[n_support:])

#         if support_indices and query_indices:
#             support_set = (X[support_indices], y[support_indices])
#             query_set = (X[query_indices], y[query_indices])
#             episodes.append((support_set, query_set))

#     return episodes, all_indices

# def compute_prototypes(embeddings, labels):
#     prototypes = []
#     for i in range(labels.shape[1]):
#         class_embeddings = embeddings[labels[:, i] == 1]
#         if len(class_embeddings) > 0:
#             prototypes.append(class_embeddings.mean(0))
#         else:
#             prototypes.append(torch.zeros(embeddings.shape[1], device=embeddings.device))
#     return torch.stack(prototypes)

# def cosine_similarity(x1, x2):
#     x1_norm = F.normalize(x1, p=2, dim=1)
#     x2_norm = F.normalize(x2, p=2, dim=1)
#     return torch.mm(x1_norm, x2_norm.transpose(0, 1))

# def hybrid_prototype_loss(query_embeddings, prototypes, query_labels, alpha=1):
#     euclidean_distances = torch.cdist(query_embeddings, prototypes)
#     cosine_distances = 1 - cosine_similarity(query_embeddings, prototypes)
#     combined_distances = (alpha * euclidean_distances) + ((1 - alpha) * cosine_distances)
#     predictions = torch.sigmoid(-combined_distances)
#     criterion = nn.BCELoss()
#     return criterion(predictions, query_labels)

# def validate_with_prototypes(model, dataloader):
#     model.eval()
#     embeddings = []
#     labels = []

#     with torch.no_grad():
#         for inputs, targets in dataloader:
#             emb = model(inputs, return_embedding=True)
#             embeddings.append(emb)
#             labels.append(targets)

#     embeddings = torch.cat(embeddings)
#     labels = torch.cat(labels)

#     prototypes = compute_prototypes(embeddings, labels)

#     distances_prot = torch.cdist(embeddings, prototypes)
#     cosine_distances = 1 - cosine_similarity(embeddings, prototypes)
#     combined_distances = (0.5 * distances_prot) + ((1 - 0.5) * cosine_distances)
#     predictions = torch.sigmoid(-combined_distances)
#     val_loss = hybrid_prototype_loss(embeddings, prototypes, labels, alpha=0.5)
    
#     # Convert predictions to binary labels (0 or 1)
#     predictions = torch.round(predictions)
    
#     # Calculate multi-label accuracy
#     correct = (predictions == labels).float().sum()
#     accuracy = correct / (labels.shape[0] * labels.shape[1])
    
#     return accuracy.item(), val_loss

# def episodic_training(model, optimizer, epochs, episodes, val_dataloader):
#     history = {}
#     history["epochs"], history["loss"], history["val_loss"] = [], [], []
#     save_best_model_cb = SaveBestModelCallback(save_path='best_model.pth', target='loss', mode='min')
#     for epoch in range(epochs):
#         total_loss = 0
#         for episode_index, episode in enumerate(episodes):
#             model.train()

#             support_set, query_set = episode
#             support_embeddings = model(support_set[0], return_embedding=True)
#             query_embeddings = model(query_set[0], return_embedding=True)

#             prototypes = compute_prototypes(support_embeddings, support_set[1])

#             loss = hybrid_prototype_loss(query_embeddings, prototypes, query_set[1])

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             total_loss += loss.item()

#         val_accuracy, val_loss = validate_with_prototypes(model, val_dataloader)
#         save_best_model_cb(model, val_loss, epoch)
#         print(f"Epoch {epoch}, Training Loss: {total_loss / len(episodes)}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}")
#         history["epochs"].append(epoch)
#         history["loss"].append(total_loss / len(episodes))
#         history["val_loss"].append(val_loss)
#     save_best_model_cb.load_best_model(model=model)
#     return history

# samples_training = torch.randperm(X_train.shape[0])[:1000]

# Prototypical_MLP = MultiLabel_Att(len(features_RFC), y_train_data.shape[1]) # MLP_MultiLabel(len(features_RFC), y_train_data.shape[1])

# optimizer = torch.optim.Adam(Prototypical_MLP.parameters(), lr=0.001)
# criterion = nn.BCELoss()

# n_episodes = 100
# n_support = 80
# n_query = 30
# episodes, unique_indices = create_episodes(X_train[samples_training, :], y_train[samples_training], n_episodes, n_support, n_query)
# print(f"Total unique inputs used for training: {len(unique_indices)}")

# history_mlp_prototypical = episodic_training(Prototypical_MLP, optimizer, 10, episodes, val_dataloader)