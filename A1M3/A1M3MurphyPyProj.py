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

def timeit(func):
    @wraps(func)
    def timed(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        return result
    return timed

def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax .transAxes, color = "red", fontsize = 15)

def save_text(file_name, content):
    with open(file_name,"w") as f:
        f.write(content)


#%% Nhanes python package.
# I did not use this because I could not figure out how to specify a data set

# import nhanes
# from nhanes.load import load_NHANES_data, load_NHANES_metadata
#
# data_df = load_NHANES_data(year='2017-2018')
# metadata_df = load_NHANES_metadata(year='2017-2018')

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
              "BMXARMC",   # BMI
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

# change to boolean alchol once per week or more
dict_alc = {0:0 , #Never in the last year
            1:1 , #Every day
            2:1 , #Nearly every day
            3:1 , #3 to 4 times a week
            4:1 , #2 times a week
            5:1 , #Once a week
            6:0 , #2 to 3 times a month
            7:0 , #Once a month
            8:0 , #7 to 11 times in the last year
            9:0 , #3 to 6 times in the last year
            10:0, #1 to 2 times in the last year
            77:np.nan, #Refused
            99:np.nan, #Don't know
            ".":np.nan}  #Missing

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
df["ALQ121"] = df["ALQ121"].map(dict_alc)
df["SMQ040"] = df["SMQ040"].map(dict_smk)
df["DLQ100"] = df["DLQ100"].map(dict_anxious)
df["DIQ010"] = df["DIQ010"].map(dict_diabetes)
df = df.dropna()
#%% summarise categorical vars

continuous = ["RIDAGEYR","SLD012","SLD013","PAD680","BMXWAIST","LBXTC","BMXARMC","BMXHIP"]
categorical = ["RIAGENDR","DBQ700","ALQ121","SMQ040","DLQ100","DIQ010"]


cont_summary = df[continuous + [target]].describe() #include continuous in summary
cont_summary.to_excel("tables/distr.xlsx")
get_value_counts(df,categorical,"tables/categorical.xlsx")

#%% Encode categorical vars
df[categorical] = df[categorical].astype(int).astype(str)
df_encoded = pd.get_dummies(df[categorical].astype(int).astype(str), drop_first=True).astype(int)
df_encoded = pd.concat([df_encoded, df[continuous]], axis = 1)

#%% Continuous correlations
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df[continuous].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig("Plots/CorrelationNumericX1.png")
plt.show()


#%% Multiple Colinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = []
for i in range(df_encoded.shape[1]):
    vif = variance_inflation_factor(df_encoded.values, i)
    rsquared = 1 - 1 / vif
    vif_data.append({
        "feature": df_encoded.columns[i],
        "VIF": vif,
        "R-squared": rsquared
    })

vif_df = pd.DataFrame(vif_data)
vif_df.to_excel("tables/vif.xlsx")
print(vif_df)

#%% split and fit standard OLS regression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm


X = df_encoded
y = df[target]

# Add constant to the model (for intercept)
X = sm.add_constant(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Standard Linear Regression
lr_model = sm.OLS(y_train, X_train).fit()


print("Beta values (coefficients):")
print(lr_model.params)

print(lr_model.summary())
lr_summary = lr_model.summary()
save_text("tables/lr_summary.txt", lr_summary.as_text())


print(f"Standard Linear Regression score: {lr_model.rsquared}")

y_pred = lr_model.predict(X_test)
predicted = pd.DataFrame({"Predicted": y_pred, "Actuatl": y_test})


lr_aic = lr_model.aic
print(f"Standard Linear Regression AIC: {lr_aic}")

#%% Forward stepwise subset
@timeit
def forward_selection(X, y, threshold_in=0.05):
    initial_features = []
    best_features = initial_features[:]
    remaining_features = list(X.columns)

    while remaining_features:
        scores_with_candidates = []
        for feature in remaining_features:
            if feature not in best_features:
                model = sm.OLS(y, sm.add_constant(X[best_features + [feature]])).fit()
                pvalues = model.pvalues.iloc[1:]
                best_pval = pvalues.min()
                if best_pval < threshold_in:
                    scores_with_candidates.append((best_pval, feature))

        if not scores_with_candidates:
            break

        scores_with_candidates.sort()
        best_pval, best_feature = scores_with_candidates[0]

        if best_pval < threshold_in:
            best_features.append(best_feature)
            remaining_features.remove(best_feature)
        else:
            break

    return best_features

selected_features_forward = forward_selection(X_train, y_train)
print(f"Selected features (Forward Selection): {selected_features_forward}")


# Fit and evaluate the model using the selected features
X_train_selected_forward = X_train[selected_features_forward]
X_test_selected_forward = X_test[selected_features_forward]

lr_model_forward = sm.OLS(y_train, X_train_selected_forward).fit()
print(f"Model score on test set (Forward Selection): {lr_model_forward.rsquared}")

lr_forward_aic = lr_model_forward.aic
print(f"Forward Selection AIC: {lr_forward_aic}")

print(lr_model_forward.summary())
lr_summary_forward = lr_model_forward.summary()
save_text("tables/lr_summary_forward.txt", lr_summary_forward.as_text())

#%% Backwards stepwise subset

@timeit
def backward_elimination(X, y, threshold_out=0.05):
    features = list(X.columns)
    while True:
        model = sm.OLS(y, sm.add_constant(X[features])).fit()
        pvalues = model.pvalues.iloc[1:]
        worst_pval = pvalues.max()
        if worst_pval > threshold_out:
            worst_feature = pvalues.idxmax()
            features.remove(worst_feature)
        else:
            break
    return features

selected_features_backward = backward_elimination(X_train, y_train)
print(f"Selected features (Backward Elimination): {selected_features_backward}")

# Fit and evaluate the model using the selected features
X_train_selected_backward = X_train[selected_features_backward]
X_test_selected_backward = X_test[selected_features_backward]

lr_model_backward = sm.OLS(y_train, X_train_selected_backward).fit()
print(f"Model score on test set (Backward Elimination): {lr_model_backward.rsquared}")

lr_backward_aic = lr_model_backward.aic
print(f"Backward Elimination AIC: {lr_backward_aic}")

print(lr_model_backward.summary())
lr_summary_backward = lr_model_backward.summary()
save_text("tables/lr_summary_backward.txt", lr_summary_backward.as_text())

#%% Exhaustive Feature Selection

##COMMENT BACK IN TO RUN BUT IT IS SLOW

# from itertools import combinations

#
# @timeit
# def optimal_subset_selection(X, y, criterion='AIC'):
#     best_score = np.inf
#     best_subset = None
#
#     X = sm.add_constant(X)
#
#     for k in range(1, len(X.columns)):
#         for subset in combinations(X.columns[1:], k):
#             subset = ('const',) + subset
#             model = sm.OLS(y, X[list(subset)]).fit()
#             if criterion == 'AIC':
#                 score = model.aic
#             elif criterion == 'BIC':
#                 score = model.bic
#             elif criterion == 'AdjR2':
#                 score = -model.rsquared_adj
#             else:
#                 raise ValueError("Criterion must be 'AIC', 'BIC', or 'AdjR2'")
#
#             if score < best_score:
#                 best_score = score
#                 best_subset = subset
#
#     return list(best_subset)
#
# selected_features_exhaustive = optimal_subset_selection(X_train, y_train, criterion='AIC')
# print(f"Selected features (Exhaustive Selection): {selected_features_exhaustive}")
#
# X_train_selected_exhaustive = X_train[selected_features_exhaustive]
# X_test_selected_exhaustive = X_test[selected_features_exhaustive]
#
# lr_model_exhaustive = sm.OLS(y_train, X_train_selected_exhaustive).fit()
# print(f"Model score on test set (Exhaustive Selection): {lr_model_exhaustive.rsquared}")
#
# print(lr_model_exhaustive.summary())
# lr_model_exhaustive = lr_model_exhaustive.summary()
# save_text("tables/lr_summary_exaustive2.txt", lr_model_exhaustive.as_text())
#%% Standard Linear Regression with scaling to compare to ridge and lasso

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV



scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred = lr.predict(X_test_scaled)


lr_r2 = r2_score(y_test, y_pred)
lr_mse = mean_squared_error(y_test, y_pred)

print(f"Linear Regression R^2: {lr_r2}")
print(f"Linear Regression MSE: {lr_mse}")


lr_coef_standard = pd.DataFrame({'Feature': ['Intercept'] + list(X_train.columns),
                              'Coefficient': [lr.intercept_] + list(lr.coef_)})

n = X_test_scaled.shape[0]
k = X_test_scaled.shape[1]
standard_mse = mean_squared_error(y_test, y_pred)


lasso_aic = n * np.log(standard_mse) + 2 * k
print(f"Lasso Regression AIC: {lasso_aic}")


lr_coef_standard.to_excel("tables/lr_coef_standard.xlsx")

print("Linear Regression Coefficients:")
print(lr_coef_standard)

#%% Ridge Regression

ridge = Ridge()
ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
ridge_cv = GridSearchCV(ridge, ridge_params, scoring='neg_mean_squared_error', cv=5)
ridge_cv.fit(X_train_scaled, y_train)
best_ridge_alpha = ridge_cv.best_params_['alpha']
ridge_best = ridge_cv.best_estimator_
y_pred_ridge_best = ridge_best.predict(X_test_scaled)
ridge_best_score = r2_score(y_test, y_pred_ridge_best)
ridge_mse = mean_squared_error(y_test, y_pred_ridge_best)
print(f"Best Ridge Regression alpha: {best_ridge_alpha}")
print(f"Best Ridge Regression R^2 score: {ridge_best_score}")
print(f"Best Ridge Regression MSE: {ridge_mse}")



ridge_aic = n * np.log(ridge_mse) + 2 * k
print(f"Ridge Regression AIC: {ridge_aic}")


ridge_coef_table = pd.DataFrame({'Feature': ['Intercept'] + list(X_train.columns),
                                 'Coefficient': [ridge_best.intercept_] + list(ridge_best.coef_)})
ridge_coef_table.to_excel("tables/ridge_coef_table.xlsx")
print("Ridge Regression Coefficients:")
print(ridge_coef_table)

#%% Lasso Regression with Cross-Validation

lasso = Lasso()
lasso_params = {'alpha': [0.01, 0.1, 1.0, 10.0]}
lasso_cv = GridSearchCV(lasso, lasso_params, scoring='neg_mean_squared_error', cv=5)
lasso_cv.fit(X_train_scaled, y_train)
best_lasso_alpha = lasso_cv.best_params_['alpha']
lasso_best = lasso_cv.best_estimator_
y_pred_lasso_best = lasso_best.predict(X_test_scaled)
lasso_best_score = r2_score(y_test, y_pred_lasso_best)
lasso_mse = mean_squared_error(y_test, y_pred_lasso_best)
print(f"Best Lasso Regression alpha: {best_lasso_alpha}")
print(f"Best Lasso Regression R^2 score: {lasso_best_score}")
print(f"Best Lasso Regression MSE: {lasso_mse}")


lasso_aic = n * np.log(lasso_mse) + 2 * k
print(f"Lasso Regression AIC: {lasso_aic}")

lasso_coef_table = pd.DataFrame({'Feature': ['Intercept'] + list(X_train.columns),
                                 'Coefficient': [lasso_best.intercept_] + list(lasso_best.coef_)})
ridge_coef_table.to_excel("tables/lasso_coef_table.xlsx")
print("Lasso Regression Coefficients:")
print(lasso_coef_table)
#%% recreate optimal subset for time sake dropped waist because of the extreme VIF


continuous = ["RIDAGEYR","BMXWAIST","BMXARMC"]
categorical = ["RIAGENDR","SMQ040","DIQ010"]

df[categorical] = df[categorical].astype(int).astype(str)
df_encoded = pd.get_dummies(df[categorical].astype(int).astype(str), drop_first=True).astype(int)
df_encoded = pd.concat([df_encoded, df[continuous]], axis = 1)

X = df_encoded
y = df[target]

print(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train_const = sm.add_constant(X_train)  # Add constant to the model (for intercept)
X_test_const = sm.add_constant(X_test)

lr_model_best = sm.OLS(y_train, X_train_const).fit()

lr_model_best_sumary = lr_model_best.summary()

save_text("tables/lr_summary_best.txt", lr_model_best_sumary.as_text())


print(f"Standard Linear Regression score: {lr_model_best.rsquared}")

# y_pred = lr_model_best.predict(X_test_const)
# predicted = pd.DataFrame({"Predicted": y_pred, "Actuatl": y_test})


lr_aic = lr_model.aic
print(f"Standard Linear Regression AIC: {lr_aic}")
#%%


residuals = lr_model_best.resid
standardized_residuals = (residuals - np.mean(residuals)) / np.std(residuals)


qqplot = sm.qqplot(standardized_residuals, line='45')
plt.title('Q-Q Plot of Residuals')
plt.savefig("Plots/residuals_QQ.png")
plt.show()


continuous_var = X_test["BMXARMC"]

plt.figure(figsize=(8, 6))
sns.residplot(x=continuous_var, y=residuals, lowess=True, line_kws={'color': 'red', 'lw': 1})
plt.xlabel('Age')
plt.ylabel('Residuals')
plt.title('Residuals vs Arm Circumference.png')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.savefig("Plots/Residuals_Age.png")
plt.show()


categorical_var = X_test["RIAGENDR_2"]  # Example categorical variable

plt.figure(figsize=(8, 6))
sns.boxplot(x=categorical_var, y=residuals)
plt.xlabel('Gender (encoded)')
plt.ylabel('Residuals')
plt.title('Residuals vs Gender')
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.savefig("Plots/Residuals_vs_Gender.png.")
plt.show()

#%% prediction evaluation


predictions = lr_model_best.get_prediction(X_test_const)
pred_summary = predictions.summary_frame(alpha=0.05)  # 95% confidence intervals


predicted_mean = pred_summary['mean']
conf_int_lower = pred_summary['obs_ci_lower']
conf_int_upper = pred_summary['obs_ci_upper']

# Print the first few predictions with their confidence intervals
print(pred_summary[['mean', 'obs_ci_lower', 'obs_ci_upper']].head())

pred_summary.to_excel("Tables/Pred_Confidence.xlsx")

y_pred = lr_model_best.predict(X_test_const)
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R^2: {r2}")
print(f"MSE: {mse}")

#%%
