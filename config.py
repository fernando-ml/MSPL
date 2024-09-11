train_data_path = "data/train_set.parquet"
test_data_path = "data/test_set.parquet"
val_data_path = "data/crossval_set.parquet"

saved_models_path = "models/saved_models"

columns_to_drop = ["flag_urg", "fwd_flag_urg", "bwd_flag_urg", "flow_id", "timestamp", "src_addr", "dst_addr"]

dual_space = True
if dual_space:
    model_type = "dual_space_prototypical"
    alpha = 0.5
    distances_weights = {
    'euclidean': 1/3,
    'chebyshev': 1/3,
    'cosine': 1/3
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
best_model_path = "best_model.pth"

target_column = "label"
# binary_class = True
positive_label = "benign"
train_batch_size = 32
val_batch_size = 512
n_epochs = 10
learning_rate = 0.004
n_episodes = 200
n_support = 40
n_query = 20
n_experiments = 30

n_samples = 100#100_000
sample_per_class = 1