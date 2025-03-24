from ucimlrepo import fetch_ucirepo
import pandas as pd
# fetch dataset
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)

# data (as pandas dataframes)
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets

df = pd.concat([X, y], axis=1)



#%%
import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb


# Convert object columns to categorical
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

for col in y.select_dtypes(include=['object']).columns:
    y[col] = y[col].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the dataset to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# Define the parameters
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss'
}

# Function to train and time the model
def train_and_time(params, use_cuda):
    if use_cuda:
        params['tree_method'] = 'gpu_hist'
        params['predictor'] = 'gpu_predictor'
    else:
        params['tree_method'] = 'hist'
        params['predictor'] = 'cpu_predictor'

    start_time = time.time()
    model = xgb.train(params, dtrain, num_boost_round=100)
    end_time = time.time()

    elapsed_time = end_time - start_time
    return elapsed_time

# Train and time the model without CUDA
cpu_time = train_and_time(params, use_cuda=False)
print(f"Training time without CUDA: {cpu_time:.2f} seconds")

# Train and time the model with CUDA
gpu_time = train_and_time(params, use_cuda=True)
print(f"Training time with CUDA: {gpu_time:.2f} seconds")
#%%
