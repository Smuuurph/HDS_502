"""
Author: Sean Murphy
Usage: This code is available for open use especially for educational purposes at
Thomas Jefferson College of Population Health

"""

#%% convert data to csv
import numpy as np
import pandas as pd

#%% read in excel - first time only
#pandas is not compatable with .dat file. initially read in the xlsx and create a csv for faster loading
# df = pd.read_excel("h209.xlsx")
# df.to_csv("h209.csv")

#%% functions

from pprint import pprint
def get_value_counts(df:pd.DataFrame, out_file:str= None):

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

#%% read data
import pandas as pd
import numpy as np
df_raw = pd.read_csv("h209.csv")
print(df_raw.head())

#%% select columns
df = df_raw[[
         "DUPERSID",        # PK - Int: Person ID
         "ADCHLC42",        # TARGET - Binary: doctor check cholesterol
         "ADBMI42",         # Continuous: BMI
         "OFTSMK53",        # Categorical: How often do you smoke cigarettes
         "ADAGE42",         # Categorical: Patient Sex
         "ADSEX42",         # Categorical: Gender at survey
         "MIDX",            # Categorical: Heart Attack diagnosis
         "ADGENH42",        # Categorical: Health in general
         ]]

#%% Variable Coding

#ADCHLC42, MIDX, DIABDX_M18
bool_encode = {-15: np.nan,
                -8:np.nan,
                -7:np.nan,
                -1:np.nan,
                1:True,
                2:False}

smoke_encode = {-8: np.nan,
              -7:np.nan,
              -1:np.nan,
              1:"Every Day",
              2:"Some Days",
              3:"Never"}

age_encode = {-15: np.nan,
              -1:np.nan,
              1:"<18",
              2:"18-35",
              3:"35-49",
              4:">50"}

gender_encode = {-15: np.nan,
              -1:np.nan,
              1:"Male",
              2:"Female"}

health_encode = {-15: np.nan,
                 -1:np.nan,
                 1:"Excellent",
                 2:"Very Good",
                 3:"Good",
                 4:"Fair",
                 5:"Poor"}
df_decoded = df.copy()
df_decoded.loc[:,"ADCHLC42"] = df["ADCHLC42"].astype(int).map(bool_encode)
df_decoded.loc[:,"OFTSMK53"] = df["OFTSMK53"].astype(int).map(smoke_encode)
df_decoded.loc[:,"ADAGE42"] = df["ADAGE42"].astype(int).map(age_encode)
df_decoded.loc[:,"ADSEX42"] = df["ADSEX42"].astype(int).map(gender_encode)
df_decoded.loc[:,"MIDX"] = df["MIDX"].astype(int).map(bool_encode)
df_decoded.loc[:,"ADGENH42"] = df["ADGENH42"].astype(int).map(health_encode)
df_decoded.loc[df["ADBMI42"] < 0, "ADBMI42"] = np.nan

#%% miss map
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.figure(figsize=(10, 6))


cmap = sns.color_palette(["#d3d3d3", "#0000ff"], as_cmap=True)

# Create the heatmap
sns.heatmap(df_decoded.isnull(), cmap=cmap, yticklabels=False, cbar=False)
plt.title('Missing Value Map')

# Create a custom legend
legend_labels = {0: 'Not Missing', 1: 'Missing'}
handles = [Patch(color="#d3d3d3", label=legend_labels[0]),
           Patch(color="#0000ff", label=legend_labels[1])]

# Place the legend outside the plot
plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend')

plt.savefig('Plots/mis_map.png', bbox_inches='tight')
plt.show()


#%% get catgorical counts and drop NA

categorical_vars = ["ADCHLC42", "OFTSMK53", "ADAGE42", "ADSEX42", "MIDX", "ADGENH42"]

get_value_counts(df_decoded[categorical_vars], out_file= "Tables/cat_counts_i.xlsx")
dist_i = df_decoded[["ADBMI42"]].describe()

rows_i = len(df_decoded)

df_decoded = df_decoded.where(df > 0, np.nan)
df_decoded = df_decoded.dropna()
get_value_counts(df_decoded[categorical_vars], out_file= "Tables/cat_counts_f.xlsx")
rows_f = len(df_decoded)

print(f"{rows_i - rows_f}/{rows_i} rows dropped due to missing values")
#%%continuous distributions
dist_f = df_decoded[["ADBMI42"]].describe()
dist_f.to_excel("Tables/dist_f.xlsx")

#drop invalid BMIs - not nessecary already dropped
# df_decoded = df_decoded.drop(df_decoded["ADBMI42"] < 5)

#%% apply dropped NA to encoded df

df = df[df["DUPERSID"].isin(df_decoded["DUPERSID"])]


#%% Categorical correlations
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import math

# Create pairs of categorical variables
pairs = list(itertools.combinations(categorical_vars + ["ADCHLC42"], 2))

# Calculate the number of rows and columns needed for the grid
num_pairs = len(pairs)
cols = int(math.ceil(math.sqrt(num_pairs)))
rows = int(math.ceil(num_pairs / cols))


fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), sharex=False, sharey=False)


axes = axes.flatten()

# Iterate over pairs and plot
for ax, (var1, var2) in zip(axes, pairs):
    sns.countplot(data=df_decoded, x=var1, hue=var2, ax=ax)
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

#%% Remap target to bool

bool_dict = {1:True,
             2:False}

df["ADCHLC42"] = df["ADCHLC42"].map(bool_dict)


#%% Train Model
from sklearn.model_selection import train_test_split
from statsmodels.discrete.discrete_model import Logit
from statsmodels.tools import add_constant


predictors = ["ADBMI42", "ADAGE42", "ADSEX42", "ADGENH42"]

df = df.astype(int)
df_feat = pd.get_dummies(df[predictors], columns=["ADGENH42", "ADAGE42","ADSEX42"], drop_first=True)
df_feat = df_feat.astype(int)
# Redefine predictors to match the new columns after one-hot encoding
predictors_with_dummys = [col for col in df_feat.columns if col not in ["ADCHLC42","DUPERSID"]]

y_train, y_test, X_train, X_test = train_test_split(df["ADCHLC42"],df_feat[predictors_with_dummys],
                                                    train_size=0.8, random_state=42)
# Adding constant to predictors
X_train = add_constant(X_train)
X_test = add_constant(X_test)
model = Logit(y_train,X_train)
res = model.fit()

res.summary()

# Extract the summary tables as DataFrames
summary_as_html = res.summary().tables[1].as_html()
summary_df = pd.read_html(summary_as_html, header=0, index_col=0)[0]
summary_df.to_html(index=False)

rename_map = {"const":"Intercept",
              "ADBMI42":"BMI",
              "ADGENH42_2":"Overall_health_very_good",
              "ADGENH42_3":"Overall_health_good",
              "ADGENH42_4":"Overall_health_fair",
              "ADGENH42_5":"Overall_health_poor",
              "ADAGE42_2":"Age_18-34",
              "ADAGE42_3":"Age_35-49",
              "ADAGE42_4":"Age_>50",
              # "OFTSMK53_2":"Smoke_some_days",
              # "OFTSMK53_3":"Smoke_never",
              "ADSEX42_2":"Female"}

summary_df.index = summary_df.index.map(rename_map)

# Save the summary DataFrame to an Excel file
summary_df.to_excel("Tables/ModelSummary.xlsx")

#%% Odds ratios and confidence intervals
params = res.params
conf = res.conf_int()
conf['OR'] = np.exp(params)
conf.columns = ['2.5%', '97.5%', 'OR']
conf.to_excel("Tables/OR.xlsx")
print("Odds Ratios and Confidence Intervals:\n", conf)

#%% Marginal Effect
marg = res.get_margeff(at="overall", method="dydx")
marg_df = marg.summary_frame()
marg_df.to_excel("Tables/marg.xlsx")

#%% ROC
from sklearn.metrics import roc_curve, auc
y_pred = res.predict(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig("Plots/ROP.png")
plt.show()


#%% Hosmer-Lemeshow test

from scipy.stats import chi2

# Create a DataFrame with actual and predicted values
data = pd.DataFrame({'actual': y_test, 'predicted': y_pred})

# Create deciles based on predicted probabilities
data['decile'] = pd.qcut(data['predicted'], 10, labels=False)

# Calculate observed and expected frequencies for each group
observed = data.groupby('decile')['actual'].sum()
expected = data.groupby('decile')['predicted'].sum()
total = data.groupby('decile')['actual'].count()

# Calculate the Hosmer-Lemeshow test statistic
hl_stat = ((observed - expected)**2 / (expected * (1 - expected / total))).sum()

# Degrees of freedom (number of groups - 2)
degrees_freedom = 10 - 2

# Calculate the p-value
p_value = 1 - chi2.cdf(hl_stat, degrees_freedom)

print("Hosmer-Lemeshow test statistic:", hl_stat)
print("p-value:", p_value)


#%% Residual distribution
def deviance_residuals(y, y_pred):
    return np.sign(y - y_pred) * np.sqrt(-2 * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

deviance_resid = deviance_residuals(y_train, y_pred)

# Calculate the percentiles of the deviance residuals
percentiles = np.percentile(deviance_resid, [0, 25, 50, 75, 100])

#%% residuals vs age
# Scatter plot of deviance residuals versus age
plt.figure(figsize=(10, 6))
sns.scatterplot(x=df_decoded['ADAGE42'], y=y_pred, color='grey')
plt.title('Scatter Plot of Deviance Residuals Versus Age')
plt.xlabel('Age')
plt.ylabel('Deviance Residuals')
plt.savefig("Plots/DevianceResidAge.png")
plt.show()
#%% residuals distribution

observed = y_test
expected = y_pred
deviance_resid = np.sign(observed - expected) * np.sqrt(-2 * (observed * np.log(expected) + (1 - observed) * np.log(1 - expected)))

deviance_percentiles = np.percentile(deviance_resid, [0, 25, 50, 75, 100])
deviance_percentiles

#%%
# Scatter plot of deviance residuals versus fitted values
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=deviance_resid, color='grey')
sns.regplot(x=y_pred, y=deviance_resid, scatter=False, color='grey')
plt.title('Scatter Plot of Residuals Versus Predicted Values')
plt.xlabel('Predicted Values')
plt.ylabel('Deviance Residuals')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.gca().set_xlabel("Predicted Values", fontsize=20)
plt.gca().set_ylabel("Deviance Residuals", fontsize=20)
plt.gca().set_title("Scatter Plot of Residuals Versus Predicted Values", fontsize=24)
plt.gca().set_facecolor('#E5E5E5')
plt.savefig("Plots/DevianceResiduals.png")
plt.show()

#%% standard error of predictors
predictions = res.get_prediction(X_test)
pred_summary = predictions.summary_frame(alpha = 0.5)

# Add the predictions and confidence intervals to the test data
X_test['pred.full'] = pred_summary['predicted']
X_test['se.fit'] = pred_summary['se']
X_test['ymin'] = pred_summary['ci_lower']
X_test['ymax'] = pred_summary['ci_upper']

print(X_test.head())
#%% Confusion Matrix
from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
thresh = 0.5
y_pred_binary = (y_pred > thresh).astype(int)

conf_matrix = confusion_matrix(y_test,y_pred_binary)


conf_matrix_df = pd.DataFrame(conf_matrix,
                              index=['Actual Negative', 'Actual Positive'],
                              columns=['Predicted Negative', 'Predicted Positive'])

conf_matrix_df.to_excel("Tables/ConfMatx.xlsx",engine='openpyxl')

print(conf_matrix_df)
# specificity of 0.319 which a weakness of the model.
# Accuracy of 0.779


#%% QQ Plot
from statsmodels.graphics.gofplots import qqplot
# Fit the Logit model


# Calculate deviance residuals manually
def deviance_residuals(y, y_pred):
    return np.sign(y - y_pred) * np.sqrt(-2 * (y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred)))

deviance_residuals = deviance_residuals(y_test, y_pred)



# Q-Q Plot
ptype = "ResVqq"
loc = f"Plots/{ptype}.png"
qqplot(deviance_residuals, line='45')
plt.title('Q-Q Plot')
plt.savefig(loc)
plt.show()

# Residuals vs BMI
ptype = "ResVdia"
loc = f"Plots/{ptype}.png"
sns.residplot(x=X_test['ADBMI42'], y=deviance_residuals, lowess=True)
plt.xlabel('ADBMI42')
plt.ylabel('Deviance Residuals')
plt.title('Residuals vs ADBMI42')
plt.grid(color='gray')
plt.show()
plt.savefig(loc)

