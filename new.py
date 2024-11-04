import json
import pandas as pd
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from utils.utils import *
from models.model_components import *
from models.prototypical_components import *
from config import *

import yaml


device = device()

config = yaml.safe_load(open("config.yaml"))


for dataset in config['datasets']:
    print(dataset['name'], dataset['path'])
