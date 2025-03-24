import time
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Create a synthetic dataset
X, y = make_classification(n_samples=1_000_000, n_features=10, n_informative=5, n_redundant=5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the dataset to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

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

# Optional: Evaluate the models
# pred_cpu = model.predict(dtest)
# pred_gpu = model.predict(dtest)
# You can also compare the predictions and performance metrics if needed

#%%

