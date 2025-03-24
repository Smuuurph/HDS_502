import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
import time

# Fetch dataset
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

# Data (as pandas dataframes)
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

# Display metadata
print(diabetes_130_us_hospitals_for_years_1999_2008.metadata)

# Display variable information
print(diabetes_130_us_hospitals_for_years_1999_2008.variables)

# Select features
features = [
    "time_in_hospital",          #continuous, no missing
    "age"                        #categorical, no missing
    "num_procedures",            #continuous, no missing
    "number_inpatient",          #continuous, no missing
    #Missing DAYS FROM ADMIT TO FIRST LAB
    "number_diagnoses"           #continuous, no missing,      #proxy for number of chronic conditions
    #Missing number of external injurys
    "admission_source_id",       #categorical,                 #proxy for observation stay
    "discharge_dispostion_id",   #categorical,                 #proxy for observation stay
    "A1Cresult",
    # "change",
    # "medical_specialty"

]

categorical_ordinal = ["age", "A1Cresult"]

age_key = {"[0-10)": 0,
           "[10-20)": 1,
           "[20-30)": 2,
           "[30-40)": 3,
           "[40-50)": 4,
           "[50-60)": 5,
           "[60-70)": 6,
           "[70-80)": 7,
           "[80-90)": 8,
           "[90-100)": 9}
A1C_key = {"Norm": 0,
           ">7": 1,
           ">8": 2}

readmitt_key = {"NO":0,
              "<30":1,
              ">30":2}


categorical = ["admission_source_id", "discharge_dispostion_id"]

# Convert categorical columns to category dtype
categorical_columns = ["age","admission_source_id", "diag_1", "diag_2", "diag_3", "change", "insulin", "A1Cresult"]
for col in categorical_columns:
    X[col] = X[col].astype('category')

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.values.ravel())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define the objective function for Optuna with GPU
def objective_gpu(trial):
    param = {
        'tree_method': 'gpu_hist',  # Use GPU acceleration
        'objective': 'multi:softmax',
        'num_class': len(set(y_encoded)),
        'eval_metric': 'mlogloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
    }

    # Create the DMatrix with enable_categorical=True
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    # Train the model
    bst = xgb.train(param, dtrain, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)

    # Evaluate the model
    preds = bst.predict(dtest)
    accuracy = (preds == y_test).mean()
    return accuracy

# Define the objective function for Optuna without GPU
def objective_cpu(trial):
    param = {
        'tree_method': 'hist',  # Use CPU acceleration
        'objective': 'multi:softmax',
        'num_class': len(set(y_encoded)),
        'eval_metric': 'mlogloss',
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
    }

    # Create the DMatrix with enable_categorical=True
    dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
    dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

    # Train the model
    bst = xgb.train(param, dtrain, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)

    # Evaluate the model
    preds = bst.predict(dtest)
    accuracy = (preds == y_test).mean()
    return accuracy

# Run Optuna study with GPU acceleration
study_gpu = optuna.create_study(direction='maximize')
start_time_gpu = time.time()
study_gpu.optimize(objective_gpu, n_trials=50, timeout=600)
end_time_gpu = time.time()
gpu_time = end_time_gpu - start_time_gpu

# Print the best parameters for GPU run
print(f"Best parameters for GPU: {study_gpu.best_params}")
print(f"Time taken with GPU: {gpu_time} seconds")

# Run Optuna study without GPU acceleration
study_cpu = optuna.create_study(direction='maximize')
start_time_cpu = time.time()
study_cpu.optimize(objective_cpu, n_trials=50, timeout=600)
end_time_cpu = time.time()
cpu_time = end_time_cpu - start_time_cpu

# Print the best parameters for CPU run
print(f"Best parameters for CPU: {study_cpu.best_params}")
print(f"Time taken with CPU: {cpu_time} seconds")

# Train the final model with the best parameters on GPU
best_params_gpu = study_gpu.best_params
best_params_gpu['tree_method'] = 'gpu_hist'
best_params_gpu['objective'] = 'multi:softmax'
best_params_gpu['num_class'] = len(set(y_encoded))
best_params_gpu['eval_metric'] = 'mlogloss'

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

bst_gpu = xgb.train(best_params_gpu, dtrain, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=True)

# Evaluate the final GPU model
final_preds_gpu = bst_gpu.predict(dtest)
final_accuracy_gpu = (final_preds_gpu == y_test).mean()
print(f"Final model accuracy with GPU: {final_accuracy_gpu}")

# Train the final model with the best parameters on CPU
best_params_cpu = study_cpu.best_params
best_params_cpu['tree_method'] = 'hist'
best_params_cpu['objective'] = 'multi:softmax'
best_params_cpu['num_class'] = len(set(y_encoded))
best_params_cpu['eval_metric'] = 'mlogloss'

bst_cpu = xgb.train(best_params_cpu, dtrain, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=True)

# Evaluate the final CPU model
final_preds_cpu = bst_cpu.predict(dtest)
final_accuracy_cpu = (final_preds_cpu == y_test).mean()
print(f"Final model accuracy with CPU: {final_accuracy_cpu}")

#%%

import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import time

# Fetch dataset
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

# Data (as pandas dataframes)
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

# Select features
features = [
    "time_in_hospital",
    "admission_source_id",
    "diag_1",
    "diag_2",
    "diag_3",
    "change",
    "insulin",
    # "A1Cresult",
    "num_procedures",
    "number_emergency"
]

X = X[features]

# Handle missing values using .loc to avoid SettingWithCopyWarning
X.loc[:, "diag_1"] = X["diag_1"].fillna("0")
X.loc[:, "diag_2"] = X["diag_2"].fillna("0")
X.loc[:, "diag_3"] = X["diag_3"].fillna("0")

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y.values.ravel())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Combine training and testing data for LabelEncoder
combined = pd.concat([X_train, X_test], axis=0)

# Convert categorical columns to category dtype for XGBoost
categorical_columns = ["admission_source_id", "diag_1", "diag_2", "diag_3", "change", "insulin", "A1Cresult"]
for col in categorical_columns:
    combined[col] = combined[col].astype('category')

# Split back to training and testing sets
X_train = combined.iloc[:len(X_train), :]
X_test = combined.iloc[len(X_train):, :]

# Define parameters for XGBoost training
params = {
    'objective': 'multi:softmax',
    'num_class': len(set(y_encoded)),
    'eval_metric': 'mlogloss',
    'learning_rate': 0.1,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist'
}

# Train XGBoost on CPU
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dtest = xgb.DMatrix(X_test, label=y_test, enable_categorical=True)

start_time_cpu = time.time()
bst_cpu = xgb.train(params, dtrain, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)
end_time_cpu = time.time()
cpu_time = end_time_cpu - start_time_cpu

# Train XGBoost on GPU
params['tree_method'] = 'hist'
params['device'] = 'cuda'

start_time_gpu = time.time()
bst_gpu = xgb.train(params, dtrain, evals=[(dtest, 'eval')], early_stopping_rounds=10, verbose_eval=False)
end_time_gpu = time.time()
gpu_time = end_time_gpu - start_time_gpu

# Evaluate XGBoost models
final_preds_cpu = bst_cpu.predict(dtest)
final_preds_gpu = bst_gpu.predict(dtest)
final_accuracy_cpu = (final_preds_cpu == y_test).mean()
final_accuracy_gpu = (final_preds_gpu == y_test).mean()

print(f"XGBoost CPU training time: {cpu_time} seconds")
print(f"XGBoost GPU training time: {gpu_time} seconds")
print(f"XGBoost final model accuracy with CPU: {final_accuracy_cpu}")
print(f"XGBoost final model accuracy with GPU: {final_accuracy_gpu}")

# Label encode categorical features for Random Forest
for col in categorical_columns:
    le = LabelEncoder()
    combined[col] = le.fit_transform(combined[col])

# Split back to training and testing sets
X_train = combined.iloc[:len(X_train), :]
X_test = combined.iloc[len(X_train):, :]

# Train Random Forest model
start_time_rf = time.time()
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
end_time_rf = time.time()
rf_time = end_time_rf - start_time_rf

# Evaluate Random Forest model
rf_preds = rf.predict(X_test)
rf_accuracy = (rf_preds == y_test).mean()

print(f"Random Forest training time: {rf_time} seconds")
print(f"Random Forest model accuracy: {rf_accuracy}")


#%%
