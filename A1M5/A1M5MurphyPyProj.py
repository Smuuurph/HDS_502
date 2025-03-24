"""
Author: Sean Murphy
Usage: This code is available for open use especially for educational purposes at
Thomas Jefferson College of Population Health

"""

#%% define Functions
from scipy.stats import pearsonr
import pandas as pd
from pprint import pprint
import time
from functools import wraps
def get_value_counts(df:pd.DataFrame,categorical_vars, out_file:str= None):

    """
    counts the unique values in each column of a dataframe. returns a dictionary with a key of column name
    and value of a dataframe containing the count of each unique value
    :param df:
    :param out_file:
    :return: dict[str,pd.DataFrame]
    """

    count_dict = {}
    for v in categorical_vars:
        count_dict[v] = df[v].value_counts().to_dict()


    # Create and display tables for each count dictionary
    tables = {}
    for var, counts in count_dict.items():
        tables[var] = pd.DataFrame(list(counts.items()), columns=[var, 'Count'])
        print(f"Table for {var}:")
        pprint(tables[var])
    if out_file:
        with pd.ExcelWriter(out_file, engine = 'openpyxl') as writer:
            for sheet, dfi in tables.items():
                dfi.to_excel(writer, sheet_name = sheet, index=False)

    return tables
def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax .transAxes, color = "red", fontsize = 15)


#create an object that stores easily adds model metrics to a dataframe. for comparison
class ModelFitMetrics:
    def __init__(self, X_train,x_test):
        self.X_train = X_train
        self.x_test = x_test
        self.df = pd.DataFrame(index = ['Train MSE', 'Train R^2', 'Test MSE', 'Test R^2', "Fit Time"])

    def append_model(self, model, fit_time):

        y_pred_train = model.predict(self.X_train)
        y_pred_test = model.predict(self.x_test)

        # Calculate metrics
        train_mse = mean_squared_error(y_train, y_pred_train)
        train_r2 = r2_score(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)

        # Create a series with the metrics
        metrics_series = pd.Series({
            'Train MSE': train_mse,
            'Train R^2': train_r2,
            'Test MSE': test_mse,
            'Test R^2': test_r2,
            "Fit Time": fit_time
        }, name=model.__class__.__name__)

        self.df = self.df.merge(metrics_series, left_index=True, right_index=True, how = "left")




#%% Read Data from directory "data"

import pyreadstat
import pandas as pd
import os

#read in all files form a folder in my working directy called "data"
# os.chdir("A1M3")
# print(os.getcwd())
data_files = os.listdir("data")
df_list = []
meta_list = []
df_raw = pd.DataFrame()
for f in data_files:
    df_i, meta_i = pyreadstat.read_xport(f"data/{f}")
    df_list.append(df_i)
    meta_list.append(meta_i)

#join all data sets together
df_raw = df_list[0]
for df_i in df_list[1::]:
    df_raw = pd.merge(df_raw, df_i, on = "SEQN", how = "outer")

#%% subset data for faster processing

target = "BMXBMI"


predictors = ["RIAGENDR",  # gender
              "RIDAGEYR",  # Age in years
              "DBQ700",    # How Healthy is your diet
              "ALQ121",    # Past 12 months how often do you drink alchol
              "SLD012",    # Usual sleep hours on workdays
              "SLD013",    # Usual sleep hours on non workdays
              "PAD680",    # minutes sedentary
              "SMQ040",    # Smoke Cigarettes
              "DLQ100",    # How often do you feel anxious
              "DIQ010",    # Diabetes
              "BMXARMC",   # Arm measurement
              "BMXWAIST",  # Hip Circumfrence
              "BMXHIP",     # Waist measuremennt
              "LBXTC"]     # Cholesterol mm/dl

include = predictors + [target]

df = df_raw[include]


mask = (df.astype(str) == '.').any(axis=1)
df = df[~mask]

#%% variable mapping - consolidate variables for simpler model
import numpy as np

# all categories included
dict_diet = {1: 1, # Excellent
             2: 2, # Very Good
             3: 3, # Good
             4: 4, # Fair
             5: 5, # Poor
             7: np.nan, # refused
             9: np.nan, # dont know
             ".": np.nan # missing
             }

# # change to boolean alchol once per week or more
# dict_alc = {0:0 , #Never in the last year
#             1:1 , #Every day
#             2:1 , #Nearly every day
#             3:1 , #3 to 4 times a week
#             4:1 , #2 times a week
#             5:1 , #Once a week
#             6:0 , #2 to 3 times a month
#             7:0 , #Once a month
#             8:0 , #7 to 11 times in the last year
#             9:0 , #3 to 6 times in the last year
#             10:0, #1 to 2 times in the last year
#             77:np.nan, #Refused
#             99:np.nan, #Don't know
#             ".":np.nan}  #Missing

#smoke at all
dict_smk = {1: 1, # Yes
            2: 1, # Sometimes
            3: 0, # no
            7: np.nan, # refused
            9: np.nan, # dont know
            ".": np.nan # missing
            }
# feel anxious once per week or more
dict_anxious = {1:1, #Daily
                2:1, #Weekly
                3:0, #Monthly
                4:0, #A few times a year
                5:0, #Never
                7:np.nan, #Refused
                9:np.nan, #Don't know
                ".": np.nan   #	Missing
}

# has diabetes or boarderline daibetes
dict_diabetes = {   1: 1,#	Yes
                    2: 0,#	No
                    3: 1,#	Borderline
                    7: np.nan,#	Refused
                    9: np.nan,#	Don't know
                    ".": np.nan,#	Missing
                    }


df["PAD680"] = df["PAD680"].replace([9999, 7777], np.nan)
df["DBQ700"] = df["DBQ700"].map(dict_diet)
# df["ALQ121"] = df["ALQ121"].map(dict_alc)
df["SMQ040"] = df["SMQ040"].map(dict_smk)
df["DLQ100"] = df["DLQ100"].map(dict_anxious)
df["DIQ010"] = df["DIQ010"].map(dict_diabetes)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.figure(figsize=(10, 6))


cmap = sns.color_palette(["#d3d3d3", "#0000ff"], as_cmap=True)

# Create the heatmap
sns.heatmap(df.isnull(), cmap=cmap, yticklabels=False, cbar=False)
plt.title('Missing Value Map')

# Create a custom legend
legend_labels = {0: 'Not Missing', 1: 'Missing'}
handles = [Patch(color="#d3d3d3", label=legend_labels[0]),
           Patch(color="#0000ff", label=legend_labels[1])]

# Place the legend outside the plot
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend')
plt.savefig('Plots/mis_map.png', bbox_inches='tight')
plt.show()


#%% drop invalid
# df = df.drop(columns = "SMQ040")
df = df.dropna()


#%% summarise categorical vars


continuous = ["RIDAGEYR","SLD012","SLD013","PAD680","BMXWAIST","SMQ040","LBXTC","BMXARMC","BMXHIP"]
categorical = ["RIAGENDR","DBQ700","ALQ121","DLQ100","DIQ010"]

cont_summary = df[continuous + [target]].describe() #include target in summary
cont_summary.to_excel("tables/distr.xlsx")
get_value_counts(df,categorical,"tables/categorical.xlsx")


#%% Continuous correlations

corr_matrix = df[continuous].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig("Plots/CorrelationNumericX1.png")
plt.show()


#%% Split
from sklearn.model_selection import train_test_split, GridSearchCV

X = df[continuous + continuous]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#%% globals
random_state = 123

model_comparison = ModelFitMetrics(X_train, X_test)


#%% Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


regressor = DecisionTreeRegressor(random_state=random_state)
time_start = time.time()
regressor.fit(X_train, y_train)
time_end = time.time()
time_elapsed = time_end - time_start
model_comparison.append_model(regressor, time_elapsed)


plt.figure(figsize=(20,10))
plot_tree(regressor, feature_names=X.columns, filled=True, rounded=True)
plt.savefig("Plots/BaseDecisionTree.png")
plt.show()

#%% bagged regressor
from sklearn.ensemble import BaggingRegressor

bagging_regressor = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=random_state),
                                     n_estimators=50,
                                     random_state=random_state)

time_start = time.time()
bagging_regressor.fit(X_train, y_train)
time_end = time.time()
time_elapsed = time_end - time_start

model_comparison.append_model(bagging_regressor,time_elapsed)

#%% random forrest
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(n_estimators=100, random_state=random_state,oob_score=True) #oob used later

time_start = time.time()
rf_regressor.fit(X_train, y_train)
time_end = time.time()
time_elapsed = time_end - time_start
model_comparison.append_model(rf_regressor, time_elapsed)

#%% Gradient boosted regression
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)


time_start = time.time()
gbr.fit(X_train, y_train)
time_end = time.time()
time_elapsed = time_end - time_start
model_comparison.append_model(gbr, time_elapsed)


#%% Gradient boosted regression with grid search
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}


grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)


time_start = time.time()
grid_search.fit(X_train, y_train)
time_end = time.time()
time_elapsed = time_end - time_start
best_params = grid_search.best_params_
gbr_best = grid_search.best_estimator_
model_comparison.append_model(gbr_best, time_elapsed)

#%% compare metric output

print(model_comparison.df)
model_comparison.df.to_excel("Tables/ModelComparison.xlsx")

#%% underfit model by removing body measures, data leakage

df1 = df.drop(columns = ["BMXARMC","BMXWAIST","BMXHIP",])

y1 = df1[target]
X1 = df1.drop(columns = [target])

X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.2, random_state=0)
model_comparison1 = ModelFitMetrics(X_train1, X_test1)

#%%

regressor1 = DecisionTreeRegressor(random_state=random_state)

time_start = time.time()
regressor1.fit(X_train1, y_train1)
time_end = time.time()
time_elapsed = time_end - time_start
model_comparison1.append_model(regressor1, time_elapsed)

plt.figure(figsize=(20,10))
plot_tree(regressor1, feature_names=X1.columns, filled=True, rounded=True)
plt.show()


#%% bagged regressor
from sklearn.ensemble import BaggingRegressor


bagging_regressor1 = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=random_state),
                                     n_estimators=50,
                                     random_state=random_state)


time_start = time.time()
bagging_regressor1.fit(X_train1, y_train1)
time_end = time.time()
time_elapsed = time_end - time_start
model_comparison1.append_model(bagging_regressor1, time_elapsed)


#%% random forrest
from sklearn.ensemble import RandomForestRegressor


rf_regressor1 = RandomForestRegressor(n_estimators=100, random_state=random_state)
time_start = time.time()
rf_regressor1.fit(X_train1, y_train1)
time_end = time.time()
time_elapsed = time_end - time_start
model_comparison1.append_model(rf_regressor1,time_elapsed)
#%% Gradient boosted regression
from sklearn.ensemble import GradientBoostingRegressor



gbr1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=random_state)
time_start = time.time()
gbr1.fit(X_train1, y_train1)
time_end = time.time()
time_elapsed = time_end - time_start
model_comparison1.append_model(gbr1, time_elapsed)

#%% Gradient boosted regression with grid search
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 0.9, 1.0]
}


grid_search1 = GridSearchCV(estimator=gbr1, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error', verbose=2)

time_start = time.time()
grid_search1.fit(X_train1, y_train1)
time_end = time.time()
time_elapsed = time_end - time_start
best_params1 = grid_search1.best_params_
gbr_best1 = grid_search1.best_estimator_
model_comparison1.append_model(gbr_best1, time_elapsed)

print("Optimized Parameters: ",best_params1)

#%% comparison
model_comparison.df.to_excel("Tables/ModelComparison.xlsx")
model_comparison1.df.to_excel("Tables/ModelComparison1.xlsx")
print("DONE")

#%% Best model using OVERFIT schema (Random Forrest)

# Get feature importance
feature_importance = rf_regressor.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.savefig("Plots/FeatureImportance.png")
plt.show()

# Predict on the test set
y_pred_test = rf_regressor.predict(X_test)
residuals = y_test - y_pred_test

# Residuals vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.savefig("Plots/Residuals.png")
plt.show()

oob_error = 1 - rf_regressor.oob_score_
print("Out of Bag error RF:",oob_error)


#%% Best model using UNDERFIT schema (Random Forrest)

# Get feature importance
feature_importance1 = gbr_best1.feature_importances_
feature_importance_df1 = pd.DataFrame({'Feature': X_train1.columns, 'Importance': feature_importance1})
feature_importance_df1 = feature_importance_df1.sort_values(by='Importance', ascending=True)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df1['Feature'], feature_importance_df1['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.savefig("Plots/FeatureImportance1.png")
plt.show()

# Predict on the test set
y_pred_test1 = gbr_best1.predict(X_test1)
residuals1 = y_test1 - y_pred_test1

# Residuals vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test1, residuals1)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted Values')
plt.savefig("Plots/Residuals1.png")
plt.show()




#%% categorical

# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, classification_report
#
# # Vectorize the BMI categorization
# bins = [-np.inf, 18.5, 25, 30, np.inf]
# labels = ['<18.5', '18.5-25', '25-30', '>30']
#
#
# df2 = df.copy()
# df2['BMI_Category'] = pd.cut(df['BMXBMI'], bins=bins, labels=labels)
#
# # Features and target
# X2 = df2.drop(columns=['BMXBMI', 'BMI_Category'])  # Assuming 'BMI' and 'target' columns exist and should be excluded from features
# y2 = df2['BMI_Category']
#
# X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=random_state)
#
# # Step 3: Train the Decision Tree Classifier
# classifier2 = DecisionTreeClassifier(random_state=random_state)
# classifier2.fit(X_train2, y_train2)
#
# # Step 5: Visualize the Tree
# plt.figure(figsize=(20,10))
# plot_tree(classifier2, feature_names=X.columns, class_names=classifier2.classes_, filled=True, rounded=True)
# plt.show()
#
# # Predict on the test set
# y_pred_test2 = classifier2.predict(X_test2)
#
# # Calculate and print classification metrics
# accuracy = accuracy_score(y_test2, y_pred_test2)
# print(f"Accuracy: {accuracy:.4f}")
# print(classification_report(y_test2, y_pred_test2))
