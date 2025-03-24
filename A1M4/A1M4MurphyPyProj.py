"""
Author: Sean Murphy
Usage: This code is available for open use especially for educational purposes at
Thomas Jefferson College of Population Health

"""


import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc, accuracy_score, balanced_accuracy_score, classification_report

#%% functions
def plot_roc_curve(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]  # Probability of the positive class
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')

def get_value_counts(df: pd.DataFrame, out_file: str = None):
    count_dict = {}
    for v in df.columns:
        count_dict[v] = df[v].value_counts().to_dict()

    # Create and display tables for each count dictionary
    tables = {}
    for var, counts in count_dict.items():
        tables[var] = pd.DataFrame(list(counts.items()), columns=[var, 'Count'])
        print(f"Table for {var}:")
        pprint(tables[var])
    if out_file:
        with pd.ExcelWriter(out_file, engine='openpyxl') as writer:
            for sheet, dfi in tables.items():
                dfi.to_excel(writer, sheet_name=sheet, index=False)

    return tables

#%% read data
df_raw = pd.read_csv("h209.csv")
print(df_raw.head())

#%% select columns
df = df_raw[[
    "HIBPDX",          # categorical High blood pressure diagnosis
    "ADAGE42",         # Categorical Age at survey
    "RACETHX",         # Race
    "ADSEX42",         # Categorical: Gender at survey
    "ADGENH42",        # Categorical: Health in general
    "OFTSMK53",        # Categorical: How often do you smoke cigarettes
    "ADNERV42",        # categorical: How often felt nervous
    "ADPCFL42",        # categorical: Felt calm/peaceful
    "ADBMI42",         # Continuous: BMI
    "PHYEXE53",        # Moderate to vigorous physical exercise 5x per week
    "TTLP18X"         # Total income
]]

target = "HIBPDX"

#%% Consolidate groups

dict_HIBPDX = {1: 1,
               2: 0}
dict_OFTSMK53 = {1: 1,
                 2: 1,
                 3: 0}
dict_ADSEX42 = {1: 1,
                2: 0}

dict_ADGENH42 = {1: 1,
                 2: 1,
                 3: 0,
                 4: 0,
                 5: 0}

dict_ADNERV42 = {0: 0,
                 1: 0,
                 2: 1,
                 3: 1}

dict_ADPCFL42 = {1: 1,
                 2: 1,
                 3: 0,
                 4: 0,
                 5: 0,
                 6: 0}

dict_PHYEXE53 = {1: 1,
                 2: 0}

df["HIBPDX"] = df["HIBPDX"].map(dict_HIBPDX)
df["OFTSMK53"] = df["OFTSMK53"].map(dict_OFTSMK53)
df["ADSEX42"] = df["ADSEX42"].map(dict_ADSEX42)
df["ADGENH42"] = df["ADGENH42"].map(dict_ADGENH42)
df["ADNERV42"] = df["ADNERV42"].map(dict_ADNERV42)
df["ADPCFL42"] = df["ADPCFL42"].map(dict_ADPCFL42)
df["PHYEXE53"] = df["PHYEXE53"].map(dict_PHYEXE53)


#%% Preprocess

df = df[(df >= 0).all(axis=1)]
df = df[df["ADAGE42"] != 1] #drop under 18
df = df[df["ADBMI42"] > 5 ] #drop under 5
print(df.head())

#%% EDA distributions
categorical = [ "OFTSMK53","ADSEX42","ADGENH42","ADNERV42","ADPCFL42","PHYEXE53","ADAGE42"]
categorical_plus_target = categorical + ["HIBPDX"]
continuous = ["ADBMI42","TTLP18X"]

get_value_counts(df[categorical_plus_target], out_file="Tables/cat_count.xlsx")
dist_summary = df[continuous].describe()
dist_summary.to_excel("Tables/distr.xlsx")

#%% EDA coorelations

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import math

# Create pairs of categorical variables
pairs = list(itertools.combinations(categorical_plus_target, 2))

# Calculate the number of rows and columns needed for the grid
num_pairs = len(pairs)
cols = int(math.ceil(math.sqrt(num_pairs)))
rows = int(math.ceil(num_pairs / cols))


fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), sharex=False, sharey=False)


axes = axes.flatten()

# Iterate over pairs and plot
for ax, (var1, var2) in zip(axes, pairs):
    sns.countplot(data=df, x=var1, hue=var2, ax=ax)
    ax.set_title(f'{var1} vs {var2}')
    ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.05))

# Remove any unused subplots
for i in range(len(pairs), len(axes)):
    fig.delaxes(axes[i])

# Add a title for the whole figure
fig.suptitle("Frequency of Categorical Variables Comparison", fontsize=16)
plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.4)
plt.savefig("Plots/CorrelationX2.png")

plt.show()


#%% Encode Race varaible

df = df.drop(columns = ["ADPCFL42","ADGENH42"])
df = pd.get_dummies(df, columns=["RACETHX","ADAGE42"])

#%% split data
X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

#%% scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#%% LDA
lda = LinearDiscriminantAnalysis()
lda.fit(X_train_scaled, y_train)

y_pred = lda.predict(X_test_scaled)
print("LDA Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("LDA Classification Report:")
print(classification_report(y_test, y_pred))

#%% QDA
qda = QuadraticDiscriminantAnalysis()
qda.fit(X_train_scaled, y_train)

y_pred = qda.predict(X_test_scaled)
print("QDA Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("QDA Classification Report:")
print(classification_report(y_test, y_pred))

#%% KNN
knn = KNeighborsClassifier(n_neighbors=35)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
print("KNN Accuracy:", accuracy_score(y_test, y_pred))
print("Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
print("KNN Classification Report:")
print(classification_report(y_test, y_pred))

#%% Plot ROC curves
plt.figure(figsize=(10, 8))

plot_roc_curve(lda, X_test_scaled, y_test, 'LDA')
plot_roc_curve(qda, X_test_scaled, y_test, 'QDA')
plot_roc_curve(knn, X_test_scaled, y_test, 'KNN')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.savefig("Plots/ROC_Curve.png")
plt.show()

#%% LDA distribution

from scipy.stats import norm

# discriminant function values
X_test_lda = lda.transform(X_test_scaled)

# Separate the discriminant function values by class
X_test_lda_class0 = X_test_lda[y_test == 0]
X_test_lda_class1 = X_test_lda[y_test == 1]

# Fit normal distributions
mean0, std0 = norm.fit(X_test_lda_class0)
mean1, std1 = norm.fit(X_test_lda_class1)

# Plot the normal distributions
x = np.linspace(min(X_test_lda.min(), mean0 - 3*std0, mean1 - 3*std1),
                max(X_test_lda.max(), mean0 + 3*std0, mean1 + 3*std1), 1000)

pdf_class0 = norm.pdf(x, mean0, std0)
pdf_class1 = norm.pdf(x, mean1, std1)

plt.figure(figsize=(10, 6))
plt.plot(x, pdf_class0, label='Class 0', color='blue')
plt.plot(x, pdf_class1, label='Class 1', color='orange')
plt.fill_between(x, pdf_class0, alpha=0.2, color='blue')
plt.fill_between(x, pdf_class1, alpha=0.2, color='orange')

# Calculate and plot the decision boundary
decision_boundary = (mean0 + mean1) / 2
plt.axvline(decision_boundary, color='red', linestyle='--', label='Decision Boundary')

print("Dicision Boundary: ",decision_boundary)

plt.title('Normal Distributions of Discriminant Function Values')
plt.xlabel('Discriminant Function Value')
plt.ylabel('Probability Density')
plt.legend()
plt.savefig("Plots/normal_of_discriminant_function.png")
plt.show()
#%% Discriminants coeficcents

df_coefficients = pd.DataFrame(lda.coef_.T,index = X_train_scaled.columns,columns=['Coefficient'])
df_coefficients.to_excel("Tables/discrim_coefficients.xlsx", index=True)



