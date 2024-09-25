import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import os

df = pd.read_csv('/Users/femartinez/Documents/Research/Dos Project/data/EVs/Power Consumption/EVSE-B-PowerCombined.csv')
columns_to_drop_power_consumption = ['time', 'Attack-Group', 'Label', 'interface']
target_column_power_consumption = "Attack"
df.drop(columns_to_drop_power_consumption, axis=1, inplace=True)
state_columns = pd.get_dummies(df['State'], prefix='State')

df = pd.concat([df, state_columns], axis=1)
df.drop('State', axis=1, inplace=True)

train_test_split = train_test_split(df, test_size=0.2, random_state=42)

train_data, test_data = train_test_split[0], train_test_split[1]

X_train_data, y_train_data = train_data.drop(target_column_power_consumption, axis=1), train_data[target_column_power_consumption]
X_test_data, y_test_data = test_data.drop(target_column_power_consumption, axis=1), test_data[target_column_power_consumption]

y_train_data = pd.get_dummies(y_train_data)
y_test_data = pd.get_dummies(y_test_data)

RFC_pipeline = Pipeline([("Scaler", sklearn.preprocessing.RobustScaler()), ("RFC", RandomForestClassifier())])
RFC_pipeline.fit(X_train_data, y_train_data)

predicitions = RFC_pipeline.predict(X_test_data)
print(classification_report(y_test_data, predicitions))