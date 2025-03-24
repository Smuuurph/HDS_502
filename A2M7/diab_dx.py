from ucimlrepo import fetch_ucirepo

# fetch dataset
cdc_diabetes_health_indicators = fetch_ucirepo(id=296)

# data (as pandas dataframes)
X = cdc_diabetes_health_indicators.data.features
y = cdc_diabetes_health_indicators.data.targets

print(X.columns)

# metadata
meta = (cdc_diabetes_health_indicators.metadata)

# variable information
vars = cdc_diabetes_health_indicators.variables


#%%
import pandas as pd
df = pd.read_csv('diabetes_012_health_indicators_BRFSS2015.csv')

y = df["Diabetes_012"]
X = df.drop(columns = ["Diabetes_012"])

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from ucimlrepo import fetch_ucirepo
import time
# Fetch dataset
dataset = fetch_ucirepo(id=544)
X = pd.DataFrame(dataset.data.features)
y = pd.Series(dataset.data.targets)

# Convert object columns to categorical
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype('category')

# Ensure target values are numeric
# Adjust this based on the actual values in your target column
y = y.map({'Insufficient_Weight': 0, 'Normal_Weight': 1, 'Overweight_Level_I': 2,
           'Overweight_Level_II': 3, 'Obesity_Type_I': 4, 'Obesity_Type_II': 5, 'Obesity_Type_III': 6})

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the dataset to DMatrix format with enable_categorical=True
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

# Define the parameters
params = {
    'objective': 'multi:softmax',  # Change to 'multi:softprob' if you need probabilities
    'num_class': 7,  # Update this based on the number of unique classes
    'eval_metric': 'mlogloss'
}

# Function to train and time the model
def train_and_time(params, use_cuda):
    if use_cuda:
        params['tree_method'] = 'hist'
        params['device'] = 'cuda'
    else:
        params['tree_method'] = 'hist'
    start_time = time.time()
    model = xgb.train(params, dtrain, num_boost_round=100)
    end_time = time.time()
    elapsed_time = end_time - start_time
    return model, elapsed_time

# Train and time the model without CUDA
model_cpu, cpu_time = train_and_time(params, use_cuda=False)
print(f"Training time without CUDA: {cpu_time:.2f} seconds")

# Train and time the model with CUDA
model_gpu, gpu_time = train_and_time(params, use_cuda=True)
print(f"Training time with CUDA: {gpu_time:.2f} seconds")

# Predict on the test set using the CPU-trained model
y_pred = model_cpu.predict(dtest)
y_pred = y_pred.astype(int)  # Ensure predictions are integer class labels

# Calculate the balanced accuracy score
balanced_acc = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy Score: {balanced_acc:.2f}")
#%%
