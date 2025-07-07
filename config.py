from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()


# config.py


dataset = "CIC"
# dataset = "EV"

# CIC dataset
train_data_path = "data/parquets/train_set.parquet"
test_data_path = "data/parquets/test_set.parquet"
val_data_path = "data/parquets/crossval_set.parquet"

# EV dataset
# Power Consumption
data_power_consumption = "data/EVs/Power Consumption/EVSE-B-PowerCombined.csv"

saved_models_path = "models/saved_models"

target_column_CIC = "label"
target_column_power_consumption = "Attack"

columns_to_drop_CIC = ["flag_urg", "fwd_flag_urg",
                       "bwd_flag_urg", "flow_id", "timestamp", "src_addr", "dst_addr"]
columns_to_drop_power_consumption = [
    'time', 'Attack-Group', 'Label', 'interface']

dual_space = True
if dual_space:
    model_type = "dual_space_prototypical"
    alpha = 0.5
    
    distances_weights = {
        'euclidean': 1,
        'chebyshev': 0,
        'cosine': 0,
        'wasserstein': 0,
        'mahalanobis': 0
    }
else:
    model_type = "regular_prototypical"
    alpha = 1
    distances_weights = {
        'euclidean': 1,
        'chebyshev': 0,
        'cosine': 0
    }
train_history_path = "train_history/"+model_type
best_model_path = "models/saved_models/best_model.pth"

# binary_class = True
# positive_label = "benign"
train_batch_size = 32
val_batch_size = 512
n_epochs = 10
learning_rate = 0.005
n_episodes = 200
n_support = 40
n_query = 20
n_experiments = 40

n_samples = 200  # 100_000
sample_per_class = 2
