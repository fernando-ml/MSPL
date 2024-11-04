from pandas import read_parquet
from torch.utils.data import TensorDataset, DataLoader
from sklearn import preprocessing
import torch


def read_ciciov(data_path, target):
    '''
    Read parquet file and return the dataset features and target separated.
    '''
    df = read_parquet(data_path)
    x = df.drop(target, axis=1).iloc[:, 1:-3]
    y = df[target]
    return x, y


def read_cicevse(data_path, target):
    '''
    Read parquet file and return the dataset features and target separated.
    '''
    df = read_parquet(data_path)
    x = df.drop(target, axis=1).iloc[:, 14:-6]
    y = df[target]
    return x, y


def read_data(data_path, target):
    '''
    Read parquet file and return the dataset features and target separated.
    '''
    df = read_parquet(data_path)
    x = df.drop(target, axis=1)
    y = df[target]
    return x, y


def preprocess_data(X, y, features, batch_size=16):
    scaler = preprocessing.RobustScaler()

    X_scaled = scaler.fit_transform(X[features].values)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.float32)
    n_features = X_scaled.shape[1]

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return X_tensor, y_tensor, dataloader
