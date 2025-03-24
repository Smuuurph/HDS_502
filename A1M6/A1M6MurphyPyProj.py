"""
Author: Sean Murphy
Usage: This code is available for open use especially for educational purposes at
Thomas Jefferson College of Population Health

"""


import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc
#%% functions
from scipy.stats import pearsonr
def plot_roc_curve(model, X_test, y_test, model_name):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.2f})')


def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax .transAxes, color = "red", fontsize = 15)


#%% read data
df_raw = pd.read_csv("h209.csv")
print(df_raw.head())

#%% select columns

#All features are continuous except for health in genearl which will be treated as continuous
features = [
    "AGELAST",         # Age
    "ADGENH42",        # Health in general
    "ADBMI42",         # BMI
    "TTLP18X",         # Total income
    "WAGEP18X",        # Wage Income
    "FAMINC18",         # Family's Total Income
    "ERTEXP18",        # ER Facility + Dr Expense
    "TOTEXP18"         # Total Medical Expenditures
]
target = "HIBPDX" #not a real target because this is unsupervised
df = df_raw[features + [target]]

#%% transform
bp_dict = {1:1,
           2:0}
df["HIBPDX"] = df["HIBPDX"].map(bp_dict)

#%% Preprocess

df = df[(df >= 0).all(axis=1)]
df = df.dropna()
df = df[df["ADBMI42"] > 5 ] #drop under 5
print(df.head())

#%% EDA distributions

dist_summary = df[features].describe()
variances = df[features].var()
dist_summary.loc['variance'] = variances
dist_summary.to_excel("Tables/Dist.xlsx")

#%% continuous scatter plots
import seaborn as sns
import matplotlib.pyplot as plt


path = "Plots/CorrelationX1.png"
main = "Continuous Correlations"
g = sns.pairplot(df[features])
g.map_upper(corrfunc)
plt.suptitle(main, y=1, fontsize=16)
plt.savefig(path)
plt.show()

#%% Split dataframes by blood pressure output

df_all =  df.drop(columns = [target])
df_high = df[df[target] == 1].drop(columns = [target])
df_norm = df[df[target] == 0].drop(columns = [target])


#%% scale to numpy arrays
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
all_scaled = scaler.fit_transform(df_all)
high_scaled = scaler.fit_transform(df_high)
norm_scaled = scaler.fit_transform(df_norm)

#%% Covariance Matrix
df_all_scaled = pd.DataFrame(all_scaled, columns=features)
covariance_matrix = df_all_scaled[features].cov()
covariance_matrix.to_excel("Tables/Cov.xlsx")

#%% PCA models
from sklearn.decomposition import PCA
pca_all = PCA()
pca_data_all = pca_all.fit_transform(all_scaled)
pca_high = PCA()
pca_data_high = pca_high.fit_transform(high_scaled)
pca_norm = PCA()
pca_data_norm = pca_norm.fit_transform(norm_scaled)

#%% Explained variance table
explained_variance_df = pd.DataFrame({
    'All Patients': pca_all.explained_variance_ratio_,
    'High Blood Pressure Patients': pca_high.explained_variance_ratio_,
    'Normal Blood Pressure Patients': pca_norm.explained_variance_ratio_
}).T

explained_variance_df.columns = ["PC_"+str(c+1) for c in explained_variance_df.columns]
explained_variance_df.to_excel("Tables/PC_Variance_Explained.xlsx")


#%% #Biplots
def plot_biplot(pca, data, labels=None, title=None,name = None):
    plt.figure(figsize=(10, 7))
    ax = plt.gca()
    ax.set_xlim([-6, 12])
    ax.set_ylim([-5, 25])
    pcs = pca.transform(data)
    plt.scatter(pcs[:, 0], pcs[:, 1], c=labels, cmap='viridis', edgecolor='k', s=50)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    plt.grid()
    if name:
        plt.savefig(f"Plots/biplot_{name}.png")
    plt.show()

plot_biplot(pca_all, all_scaled, title='PCA - All Patients', name = "all")
plot_biplot(pca_high, high_scaled, title='PCA - High Blood Pressure Patients', name = "high")
plot_biplot(pca_norm, norm_scaled, title='PCA - Normal Blood Pressure Patients', name = "norm")


#%%
loadings_all = pca_all.components_.T[:, 0]
loadings_high = pca_high.components_.T[:, 0]
loadings_norm = pca_norm.components_.T[:, 0]


fig, ax = plt.subplots(figsize=(20, 7))
width = 0.2  # Width of the bars

x = np.arange(len(features))  # Label locations

bars_all = ax.bar(x - width, loadings_all, width, label='All Patients')
bars_high = ax.bar(x, loadings_high, width, label='High Blood Pressure Patients')
bars_norm = ax.bar(x + width, loadings_norm, width, label='Normal Blood Pressure Patients')

ax.set_xlabel('Features')
ax.set_ylabel('Loading')
ax.set_title('PCA Loadings')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=90)
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('Plots/pca_loadings_clustered.png')
plt.show()

#%% Cumulative Scree plot

import numpy as np
import matplotlib.pyplot as plt

pcas = [pca_all, pca_high, pca_norm]
labels = ['All Patients', 'High Blood Pressure Patients', 'Normal Blood Pressure Patients']
title = "Cumulative Variance"

plt.figure(figsize=(10, 7))
ax = plt.gca()
ax.set_ylim([0, 1])

for pca, label in zip(pcas, labels):
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='--', label=label)

# Update x-axis labels
num_components = len(pca_all.explained_variance_ratio_)
plt.xticks(ticks=np.arange(num_components), labels=[f'PC_{i+1}' for i in range(num_components)])

plt.xlabel('Principal Components')
plt.ylabel('Cumulative Proportion of Variance Explained')
plt.title(title)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/cumulative_variance_combined.png')
plt.show()



#%% Scree Plot

import numpy as np
import matplotlib.pyplot as plt

pcas = [pca_all, pca_high, pca_norm]
labels = ['All Patients', 'High Blood Pressure Patients', 'Normal Blood Pressure Patients']
title = "Variance"

plt.figure(figsize=(10, 7))

for pca, label in zip(pcas, labels):
    plt.plot(pca.explained_variance_ratio_, marker='o', linestyle='--', label=label)

# Update x-axis labels
num_components = len(pca_all.explained_variance_ratio_)
plt.xticks(ticks=np.arange(num_components), labels=[f'PC_{i+1}' for i in range(num_components)])

plt.xlabel('Principal Components')
plt.ylabel('Proportion of Variance Explained')
plt.title(title)
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig('Plots/variance_explained_combined.png')
plt.show()


#%% Elbow Plot
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import seaborn as sns

df_cluster = df.copy()
df_cluster_scaled = scaler.fit_transform(df_cluster)


model = KMeans()
visualizer = KElbowVisualizer(model, k=(2,10), timings=False)
visualizer.fit(df_cluster_scaled)        # Fit the data to the visualizer
visualizer.show(outpath = "Plots/elbow.png") # Finalize and render the figure

# Get the optimal number of clusters
optimal_clusters = visualizer.elbow_value_

print(f"Optimal number of clusters: {optimal_clusters}")

#%% Cluster Pair Plots 2-4 clusters included
for n_clusters in range(2,5):
    kmeans = KMeans(n_clusters=n_clusters)
    km = kmeans.fit(df_cluster_scaled)
    df_km = pd.DataFrame(df_cluster_scaled,columns = features + [target])
    df_km['Cluster'] = km.labels_
    df_km['Cluster'] = "Cluster_"+df_km['Cluster'].astype(str)


    # Visualize the clusters
    sns.pairplot(df_km, hue='Cluster', palette=sns.color_palette("bright",n_colors=n_clusters), diag_kind='kde')
    plt.suptitle("Cluster Visualization", y=1.02)
    plt.legend()
    plt.savefig(f'Plots/clusters_{n_clusters}.png')
    plt.show()

#%%
