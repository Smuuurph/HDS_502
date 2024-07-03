#%% Define Utility Functions

from scipy.stats import pearsonr
import numpy as np


def corrfunc(x, y, ax=None, **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    r, _ = pearsonr(x, y)
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes, color = "red", fontsize = 15)


def qq_plot(lm,save_path, kind="Residuals"):
    # Extract residuals from the linear model
    resid = lm.resid

    # Calculate the quantiles
    y = np.quantile(resid[~np.isnan(resid)], [0.25, 0.75])
    x = stats.norm.ppf([0.25, 0.75])

    # Calculate the slope and intercept for the Q-Q plot reference line
    slope = np.diff(y) / np.diff(x)
    intercept = y[0] - slope * x[0]

    # Create the Q-Q plot
    plt.figure(figsize=(8, 6))
    stats.probplot(resid, dist="norm", plot=plt)
    plt.plot(x, slope * x + intercept, color="blue", lw=2)

    # Customize the plot
    plt.title(f"{kind} Q-Q Plot")
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.savefig(save_path)

    plt.show()

#%% Read data method 1
import pandas as pd
# df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv")

#%% Read data method 2
from ucimlrepo import fetch_ucirepo
import pandas as pd
# fetch dataset
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)

# data (as pandas dataframes)
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets

df = pd.concat([X, y], axis=1)


#%% Extract column meta data
# metadata
meta_data = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.metadata

# variable information
variables = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.variables



df["BMI"] = df["Weight"] / df["Height"]

continuous_vars = variables[(variables["type"] == "Continuous") |
                            (variables["type"] == "Integer")]["name"].to_list() + ["BMI"]


categorical_vars = variables[(variables["type"] == "Categorical") |
                             (variables["type"] == "Binary")]["name"].to_list()


#A few of the NCP values (number of meals per day) are long floats which doesnt make sence
#I rounded them and will treat as categorical vars since there are only 4 values
df["NCP"] = df["NCP"].astype(int).round()
continuous_vars.remove("NCP")
categorical_vars.append("NCP")

variables[["name","description"]]


#%% Describe the continuous variables distribution
df[continuous_vars].describe()

#%% Count the frequency of categorical varaibles
from pprint import pprint
cts = df.value_counts()
count_dict = {}
for v in categorical_vars:
    count_dict[v] = df[v].value_counts().to_dict()

pprint(count_dict)


#%% continuous scatter plots
import seaborn as sns
import matplotlib.pyplot as plt


# First plot: Continuous Data Correlations
path = "Plots/CorrelationX1.png"
main = "Weight Gain Continuous Data Correlations"

# Create the pair plot
g = sns.pairplot(df[continuous_vars])
g.map_upper(corrfunc)
plt.suptitle(main, y=1, fontsize=16)
plt.savefig(path)
plt.show()


#%% Continuous correlations
# Correlations are a little hard to read in the previous code so I made a sepearte plot of numeric correlations.
corr_matrix = df[continuous_vars].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.savefig("Plots/CorrelationNumericX1.png")
plt.show()

#%% Categorical correlations
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import itertools
import math

# Create pairs of categorical variables
pairs = list(itertools.combinations(categorical_vars, 2))

# Calculate the number of rows and columns needed for the grid
num_pairs = len(pairs)
cols = int(math.ceil(math.sqrt(num_pairs)))
rows = int(math.ceil(num_pairs / cols))

# Set up the matplotlib figure with adjusted size
fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), sharex=False, sharey=False)

# Flatten axes array for easy iteration
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

#%% Interactions
df['MTRANS'] = df['MTRANS'].astype('category')

# Create a box plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='MTRANS', y='FAF')

# Add title and labels
plt.title('Box Plot of FAF by MTRANS')
plt.xlabel('Mode of Transportation (MTRANS)')
plt.ylabel('FAF (Continuous Variable)')
plt.savefig("Plots/Box Plot of FAF by MTRANS.png")

#%% Create the model
from scipy import stats
from statsmodels.formula.api import ols
model = ols("BMI~(Gender + Age + FAVC + FCVC + CALC + SCC + FAF + CH2O + CAEC+ TUE)",data=df).fit()
m1 = model
#yes: Gender, Age, FAVC, FCVC, CALC, SCC, FAF, CAEC, TUE

vars_used = ["Gender", "Age", "FAVC", "FCVC", "CALC", "SCC", "FAF", "CAEC", "TUE"]
#no: Height, Weight, NObeyesdad, family_history_with_overweight, smoke, NCP, MTRANS


#%% Hypothesis testing
stats.skew(m1.resid)
#%%
stats.kurtosis(m1.resid)
#%%
from statsmodels.stats.anova import anova_lm
anova_lm(m1)
#%%
summary_df = pd.DataFrame({
    'Coefficient': model.params,
    't-value': model.tvalues,
    'p-value': model.pvalues,
})

# summary_df
m1.summary()
#%%
qq_plot(m1, "Plots/QQPlot.png")

#%% Check Residuals variance
import matplotlib.pyplot as plt

df_plot = pd.DataFrame({"Predictions": m1.fittedvalues,
                        "Residuals": m1.resid})

plt.figure(figsize=(10,6))
sns.scatterplot(data=df_plot, x="Predictions", y="Residuals")
plt.axhline(0, linestyle='--', color='red')
plt.title('Assumptions: Residuals vs. Predicted', fontsize=14, ha='center')
plt.xlabel('Predictions')
plt.ylabel('Residuals')

# Save the plot
plt.savefig('Plots/ResVsPred.png')
plt.show()

#%%
from statsmodels.stats.diagnostic import het_breuschpagan
bp_test = het_breuschpagan(m1.resid, model.model.exog)
bp_test_statistic = bp_test[0]
bp_test_p_value = bp_test[1]
bp_test_f_value = bp_test[2]
bp_test_f_p_value = bp_test[3]

print("bp_test_statistic: ", bp_test_statistic,
      "\nbp_test_p_value: ", bp_test_p_value,
      "\nbp_test_f_value: ", bp_test_f_value,
      "\nbp_test_f_p_value: ", bp_test_f_p_value)


#%% Levene
from scipy.stats import levene

stat, p_value = levene(*[group["BMI"].values for name, group in df.groupby("FAF")])

print(f"Levene's Test Statistic: {stat}")
print(f"Levene's Test p-value: {p_value}")

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='FAF', y='BMI')

# Add title and labels
plt.title('Box Plot of Weight by Cell')
plt.xlabel('Cell')
plt.ylabel('Weight')

# Save the plot
plt.savefig("Plots/HIVBP.png")

# Show the plot
plt.show()
#%%
for i in vars_used:
    stat, p_value = levene(*[group["BMI"].values for name, group in df.groupby(i)])
    print(f"Levene's Test for: {i}")
    print(f"Levene's Test Statistic: {stat}")
    print(f"Levene's Test p-value: {p_value}")
    print("\n")

#%%
from statsmodels.stats.diagnostic import acorr_ljungbox

# Ljung-Box test
ljung_box_result = acorr_ljungbox(m1.resid, lags=[1], return_df=True)
print(ljung_box_result)
#%%
import os

# Simulate residuals (residuals from the fitted model)
residuals = model.resid

# Plot residuals vs fitted values
ptype = "ResVpred"
ver = "version"  # Define your version variable
part = "part"    # Define your part variable
mdl = "model"    # Define your model name
loc = os.path.join("Plots", "ResVBigBin.png")

plt.figure(figsize=(10, 6))
sns.residplot(x=model.fittedvalues, y=residuals, lowess=True)
plt.xlabel('Fitted values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted values')
plt.savefig(loc)

#%%
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.api as sm

aov_table = sm.stats.anova_lm(model, typ=2)
print(aov_table)

# Perform Tukey's HSD test for pairwise comparisons
tukey = pairwise_tukeyhsd(endog=df['BMI'],
                          groups=df['FAVC'].astype(str) + " : " + df['CAEC'].astype(str),
                          alpha=0.05)

# Print Tukey's HSD results
print(tukey)

# Plot and save the Tukey HSD results
plt.figure(figsize=(10, 6))
tukey.plot_simultaneous()
plt.title('Tukey HSD Test for Pairwise Comparisons')
plt.xlabel('Mean Difference')
plt.grid(True)

# Ensure the directory exists
output_dir = "A1M1/Plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.savefig(os.path.join(output_dir, "HIVPWC.png"))
#%%
m2 = ols("BMI~(Gender + Age + FAVC + FCVC + CALC + SCC + FAF + CH2O + CAEC + TUE + FAVC*CAEC)", data=df).fit()

# Summary of the model
model_summary = m2.summary()
print(model_summary)

# Save the summary to a LaTeX file
with open("A1M1/Tables/ModelSummaryAll.tex", "w") as f:
    f.write(model_summary.as_latex())

#%%  Effect Size
import pingouin as pg

# Calculate effect sizes

# Prepare data for effect size calculation
df['Gender'] = df['Gender'].astype('category').cat.codes
df['FAVC'] = df['FAVC'].astype('category').cat.codes
df['CALC'] = df['CALC'].astype('category').cat.codes
df['SCC'] = df['SCC'].astype('category').cat.codes
df['CAEC'] = df['CAEC'].astype('category').cat.codes

# Create interaction term manually
df['FAVC_CAEC'] = df['FAVC'] * df['CAEC']

# List of predictor variables including interaction term
predictors = ['Gender', 'Age', 'FAVC', 'FCVC', 'CALC', 'SCC', 'FAF', 'CH2O', 'CAEC', 'TUE', 'FAVC_CAEC']

# Calculate effect sizes
effect_sizes = pg.linear_regression(df[predictors], df['BMI'])
print(effect_sizes)

# Save the effect sizes to a LaTeX file
with open("Tables/EffectSize.tex", "w") as f:
    f.write(effect_sizes.to_latex())
#%%
# Fit another model (for comparison)
# Compare models using ANOVA
anova_results = sm.stats.anova_lm(m1, m2)
print(anova_results)

# Save the ANOVA comparison to a LaTeX file
with open("Tables/ModelCompare.tex", "w") as f:
    f.write(anova_results.to_latex())