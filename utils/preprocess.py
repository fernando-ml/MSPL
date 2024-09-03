import os
import numpy as np
import pandas as pd
import pickle
import sklearn
import sklearn.pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import torch
from torch.utils.data import TensorDataset, DataLoader

def preprocess_df(df: pd.DataFrame, 
                  target_column: str, 
                  columns_to_drop: list, 
                  positive_label: str, 
                  project_path: str, 
                  binary_class: bool=True,
                  batch_size: int=32,
                  train: bool=False, 
                  random_seed: int=10):
    
    le_path = os.path.join(project_path, 'models', 'saved_models', 'label_encoder.pkl')
    features_path = os.path.join(project_path, 'models', 'saved_models', 'features.pkl')
    scaler_path = os.path.join(project_path, 'models', 'saved_models', 'scaler.pkl')
    df = drop_columns(df, columns_to_drop=columns_to_drop)
    X, y = split(df=df, target_column=target_column)
    
    if binary_class:
        y = y.map(lambda x: 1 if x != positive_label else 0)
    else:
        if train:
            le = LabelEncoder()
            y = le.fit_transform(y)
            with open(le_path, 'wb') as f:
                pickle.dump(le, f)
        else:
            if os.path.exists(le_path):
                with open(le_path, 'rb') as f:
                    le = pickle.load(f)
                y = le.transform(y)
            else:
                raise FileNotFoundError(f'Could not find {le_path} to load label encoder')
    if os.path.exists(features_path):
        with open(features_path, 'rb') as f:
            relevant_features = pickle.load(f)
    else:
        relevant_features = extract_relevant_features(X, y, random_seed=random_seed)
        with open(features_path, 'wb') as f:
            pickle.dump(relevant_features, f)

    if train:
        scaler = sklearn.preprocessing.RobustScaler()
        scaler.fit(X[relevant_features])
        with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
    else:
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            raise FileNotFoundError(f'Could not find {scaler_path} to load scaler')
    X = X[relevant_features]
    X = scaler.transform(X)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y.values, dtype=torch.float32)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle= True if train else False)
    return dataloader
    
    

def drop_columns(df: pd.DataFrame, 
                 columns_to_drop: list | str | None = None) -> pd.DataFrame:
    """
    Drops specified columns from a DataFrame.

    Args:
        df (pandas.DataFrame): The DataFrame from which columns will be dropped.
        columns_to_drop (list or str): The column(s) to be dropped. If a single string is provided, it will be treated as a single column to drop. If a list is provided, each element will be treated as a separate column to drop.

    Returns:
        pandas.DataFrame: The DataFrame with the specified columns dropped.
    """
    if columns_to_drop is None:
        return df
    if not isinstance(columns_to_drop, list):
        columns_to_drop = [columns_to_drop]
    else:
        columns_to_drop = columns_to_drop
    df = df.drop(columns=columns_to_drop, axis=1)
    return df

def split(df, target_column):
    '''
    Splits the dataframe into X and y
    '''
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y

def extract_relevant_features(X, y, random_seed=10) -> np.ndarray:
    """
    Extracts relevant features from the given dataset using a Random Forest Classifier.

    Returns:
        numpy.ndarray: An array of column names corresponding to the relevant features.

    Description:
        This function uses a Random Forest Classifier to estimate the importance of each feature in the dataset.
        It then selects the features with importance greater than or equal to the mean importance and returns their column names.
    """
    RFC_pipeline = sklearn.pipeline.Pipeline([("Scaler", sklearn.preprocessing.RobustScaler()), 
                                              ("RFC", RandomForestClassifier(random_state=random_seed))
                                              ])
    RFC_pipeline.fit(X, y)
    ftr_importance_RFC = pd.DataFrame(zip(X.columns, RFC_pipeline["RFC"].feature_importances_))
    ftr_importance_RFC.columns = ["Column", "Weight"]
    ftr_importance_RFC.sort_values("Weight", ascending=False, inplace=True)
    features_RFC = ftr_importance_RFC[ftr_importance_RFC.Weight >= ftr_importance_RFC.Weight.mean()].Column.values
    return features_RFC

