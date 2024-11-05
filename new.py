# import json
# import pandas as pd
# import sklearn
# from sklearn.pipeline import Pipeline
# from sklearn.ensemble import RandomForestClassifier
# import torch
# from torch import nn
# from torch.utils.data import TensorDataset, DataLoader
# from utils.utils import *
# from models.model_components import *
# from models.prototypical_components import *
# from config import *

import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

# device = device()

config = yaml.safe_load(open("config.yaml"))

class DatasetManager:
    def __init__(self, config):
        self.selected_dataset = config['selected-dataset']
        self.loaded_dataset = None
        self.target_column = None

    def get_data(self):
        self.datasets = yaml.safe_load(open("utils/datasets_config.yaml"))
        # print(self.datasets)
        for dataset in self.datasets:
            if dataset['name'] == self.selected_dataset:
                if len(dataset.keys()) > 2:
                    self.loaded_dataset = pd.read_parquet(dataset['path_train']), pd.read_parquet(dataset['path_val'])
                    self.target_column = dataset['target_column']
                else:
                    self.loaded_dataset = pd.read_parquet(dataset['path'])
                    self.target_column = dataset['target_column']
            print("Loaded dataset:", self.selected_dataset)

    def _preprocess_data(self):
        if len(self.loaded_dataset) > 2:
            train_data, validation_data = self.loaded_dataset[0], self.loaded_dataset[1]
        else:
            train_data, validation_data = train_test_split(self.loaded_dataset, test_size=0.5)
        try:
            # columns to drop
            

            y_train_data = pd.get_dummies(train_data[self.target_column])
            y_validation_data = pd.get_dummies(validation_data[self.target_column])

            X_train_data, y_train_data = train_data.drop(
                self.target_column, axis=1), y_train_data
            X_validation_data, y_validation_data = validation_data.drop(
                self.target_column, axis=1), y_validation_data
        except:
            pass

        
DatasetManager(config).get_data()

# for dataset in config['datasets']:
#     print(dataset['name'], dataset['path'])
