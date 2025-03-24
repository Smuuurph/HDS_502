import pandas as pd
from pprint import pprint
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_curve
    , auc
)
from scipy.stats import pearsonr




class ModelFitMetrics:

    """
    Class to store models and compare and their results.
    This method keeps my analysis code space clean so that i don't have a 10 output variables for each model.
    This class is a little sloppy. It would be better to break up the inner functions and make it easier to access things after running the model
    """

    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.df = pd.DataFrame(index=[
            'Train Accuracy', 'Test Accuracy',
            'Train Balanced Accuracy', 'Test Balanced Accuracy',
            'Train AUC', 'Test AUC',
            'Train Precision', 'Test Precision',
            'Train Recall', 'Test Recall',
            'Fit Time'
        ])
        self.models = {}
        self.confusion_matrices = {}
        self.feature_importances = {}
        self.best_params = {}
        self.fit_times = {}

    def append_model(self, model, name, fit_time):
        id = f"scale_pos_weight: {name}"

        """
        store model and metrics metrics which are only available at the top level
        """

        self.models[id] = model
        self.fit_times[id] = fit_time


        if hasattr(model, 'best_params_'):
            self.best_params[id] = model.best_params_


        if hasattr(model, 'feature_importances_'):
            self.feature_importances[id] = model.feature_importances_

    def generate_metrics(self):

        """
        loop through the list of models and populate each of theoutput dictionaries:
        - feature importance
        - confusion matrices
        - best parameters
        - mode fit metrics

        """

        for id, model in self.models.items():

            # Train predictions
            y_pred_train = model.predict(self.X_train)
            y_pred_train_proba = model.predict_proba(self.X_train)[:, 1] if hasattr(model, "predict_proba") else None

            # Test predictions
            y_pred_test = model.predict(self.X_test)
            y_pred_test_proba = model.predict_proba(self.X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Calculate metrics
            train_accuracy = accuracy_score(self.y_train, y_pred_train)
            test_accuracy = accuracy_score(self.y_test, y_pred_test)

            train_balanced_accuracy = balanced_accuracy_score(self.y_train, y_pred_train)
            test_balanced_accuracy = balanced_accuracy_score(self.y_test, y_pred_test)

            train_auc = roc_auc_score(self.y_train, y_pred_train_proba) if y_pred_train_proba is not None else None
            test_auc = roc_auc_score(self.y_test, y_pred_test_proba) if y_pred_test_proba is not None else None

            train_precision = precision_score(self.y_train, y_pred_train)
            test_precision = precision_score(self.y_test, y_pred_test)

            train_recall = recall_score(self.y_train, y_pred_train)
            test_recall = recall_score(self.y_test, y_pred_test)

            # Create a series with the metrics including fit time
            metrics_series = pd.Series({
                'Train Accuracy': train_accuracy,
                'Test Accuracy': test_accuracy,
                'Train Balanced Accuracy': train_balanced_accuracy,
                'Test Balanced Accuracy': test_balanced_accuracy,
                'Train AUC': train_auc,
                'Test AUC': test_auc,
                'Train Precision': train_precision,
                'Test Precision': test_precision,
                'Train Recall': train_recall,
                'Test Recall': test_recall,
                'Fit Time': self.fit_times[id]
            }, name=id)

            self.df = self.df.merge(metrics_series, left_index=True, right_index=True, how="left")

            self.confusion_matrices[id] = {
            'Train': pd.DataFrame(confusion_matrix(self.y_train, y_pred_train),
                                  index=['Actual Negative', 'Actual Positive'],
                                  columns=['Predicted Negative', 'Predicted Positive']),
            'Test': pd.DataFrame(confusion_matrix(self.y_test, y_pred_test),
                                 index=['Actual Negative', 'Actual Positive'],
                                 columns=['Predicted Negative', 'Predicted Positive'])
        }

            if hasattr(model, 'feature_importances_'):
                self.feature_importances[id] = pd.Series(model.feature_importances_, index=self.X_train.columns)

            if hasattr(model, 'best_params_'):
                self.best_params[id] = model.best_params_

    def plot_roc_curves(self, save_to):
        plt.figure(figsize=(10, 8))

        for id, model in self.models.items():
            # Check if model has predict_proba method
            if hasattr(model, "predict_proba"):
                y_pred_test_proba = model.predict_proba(self.X_test)[:, 1]
                fpr, tpr, _ = roc_curve(self.y_test, y_pred_test_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=2, label=f'{id} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for All Models')
        plt.legend(loc="lower right")
        plt.savefig(save_to)
        plt.show()


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

#%% Import data and select features
import pandas as pd
import time
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

# Fetch dataset
diabetes_130_us_hospitals_for_years_1999_2008 = fetch_ucirepo(id=296)

# Data (as pandas dataframes)
X = diabetes_130_us_hospitals_for_years_1999_2008.data.features
y = diabetes_130_us_hospitals_for_years_1999_2008.data.targets

# Select features
features = [
    "time_in_hospital",          # continuous, no missing
    "age",                       # categorical, no missing
    "num_procedures",            # continuous, no missing
    "number_inpatient",          # continuous, no missing
    "number_diagnoses",          # continuous, no missing, proxy for number of chronic conditions
    "admission_source_id",       # categorical, proxy for observation stay
    "discharge_disposition_id",  # categorical, proxy for observation stay
    "A1Cresult",                  # categorical
    "number_emergency",
    "number_outpatient",
    "diag_1",
    "num_lab_procedures",
    "admission_type_id",
    "race",
    "medical_specialty",
    "num_medications"
]



# Filter the selected features
X = X[features]


#%% miss map
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.figure(figsize=(10, 6))


cmap = sns.color_palette(["#d3d3d3", "#0000ff"], as_cmap=True)

sns.heatmap(X.isnull(), cmap=cmap, yticklabels=False, cbar=False)
plt.title('Missing Value Map')

legend_labels = {0: 'Not Missing', 1: 'Missing'}
handles = [Patch(color="#d3d3d3", label=legend_labels[0]),
           Patch(color="#0000ff", label=legend_labels[1])]

plt.legend(handles=handles, loc='center left', bbox_to_anchor=(1, 0.5), title='Legend')
plt.savefig('Plots/mis_map.png', bbox_inches='tight')
plt.show()

X.drop(columns = ["A1Cresult","medical_specialty"], inplace = True)
features.remove('A1Cresult')
features.remove("medical_specialty")

#%% transform categorical values
# Map ordinal columns
age_key = {"[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4, "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9}
# A1C_key = {"Norm": 0, ">7": 1, ">8": 2}
readmitt_key = {"NO":0, "<30":0, ">30":1}

X['age'] = X['age'].map(age_key)
# X['A1Cresult'] = X['A1Cresult'].map(A1C_key)
y['readmitted'] = y['readmitted'].map(readmitt_key)

categorical = ["admission_source_id", "discharge_disposition_id","diag_1","admission_type_id","race"]
categorical_ordinal = ["age"]
target = "readmitted"

#%%drop hospice and expired
drop_list = [11,13,14,19,20,21]

# Apply the mask to X
mask = ~X["discharge_disposition_id"].isin(drop_list)
X = X[mask]
y = y.loc[X.index]


#%% ANOVA
from scipy.stats import f_oneway

def run_anova(feature):
    df = X.copy()
    df['readmitted'] = y
    groups = [group['readmitted'].values for name, group in df.groupby(feature)]
    f_stat, p_value = f_oneway(*groups)
    return {"f_stat":f_stat, "p_value":p_value}

anova_dict = {}

anova_dict["diag_1"] = run_anova("diag_1")
anova_dict["discharge_disposition_id"] = run_anova("discharge_disposition_id")
anova_dict["admission_source_id"] = run_anova("admission_source_id")
anova_dict["admission_type_id"] = run_anova("admission_type_id")
anova_dict["race"] = run_anova("race")


df_anova = pd.DataFrame.from_dict(anova_dict).T
df_anova.to_excel("Tables/anova_dict.xlsx")

#%% mann_whitney
from scipy.stats import mannwhitneyu

def run_mann_whitney(continuous_feature):
    df = X.copy()
    df["readmitted"] = y

    group1 = df[continuous_feature][df["readmitted"] == 0]
    group2 = df[continuous_feature][df["readmitted"] == 1]
    u_stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')

    return {"u_stat": u_stat, "p_value": p_value}


mann_whitney_dict = {}

mann_whitney_dict["time_in_hospital"] = run_mann_whitney("time_in_hospital")
mann_whitney_dict["number_inpatient"] = run_mann_whitney("number_inpatient")
mann_whitney_dict["num_procedures"] = run_mann_whitney("num_procedures")
mann_whitney_dict["age"] = run_mann_whitney("age")
mann_whitney_dict["number_diagnoses"] = run_mann_whitney("number_diagnoses")
mann_whitney_dict["number_emergency"] = run_mann_whitney("number_emergency")
mann_whitney_dict["number_outpatient"] = run_mann_whitney("number_outpatient")
mann_whitney_dict["num_lab_procedures"] = run_mann_whitney("num_lab_procedures")
mann_whitney_dict["num_medications"] = run_mann_whitney("num_medications")

df_mann_whitney = pd.DataFrame.from_dict(mann_whitney_dict).T
df_mann_whitney.to_excel("Tables/mann.xlsx")

#%% split data and instantiate comparison object

X_encoded = pd.get_dummies(X, columns=categorical, drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
print(X_train.head())

model_compare = ModelFitMetrics(X_train, X_test, y_train, y_test)
seed = 123


#%% EDA

categorical_all = categorical+categorical_ordinal
get_value_counts(X,categorical_all,"Tables/cat_counts.xlsx")
get_value_counts(y,[target],"Tables/cat_counts_target.xlsx")

continuous = [i for i in features if i not in categorical_all]
df_dist = X[continuous].describe()
df_dist.to_excel("Tables/dist.xlsx")

#%% correlations
possible_correlation = ["time_in_hospital", "age","number_diagnoses","number_inpatient","num_medications"]

path = "Plots/CorrelationX1.png"
main = "Continuous Correlations"
g = sns.pairplot(X[possible_correlation])
g.map_upper(corrfunc)
plt.suptitle(main, y=1, fontsize=16)
plt.savefig(path)
plt.show()

#%% boost function
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score

def run_xgb(weight):
    # Define the parameter grid
    param_grid = {
        'max_depth': [3, 6, 9],
        'eta': [0.1, 0.3, 0.5],
        'subsample': [0.8, 1.0],
        "scale_pos_weight": [weight],
        'colsample_bytree': [0.8, 1.0],
        'n_estimators': [100, 200],
        "tree_method":["gpu_hist"]
    }

    time_start = time.time()

    xgb_model = xgb.XGBClassifier(objective='binary:logistic', seed=seed, use_label_encoder=False)

    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid,
                               scoring="balanced_accuracy", cv=3, verbose=1)

    grid_search.fit(X_train, y_train)
    time_end = time.time()
    time_elapsed_xg = time_end - time_start
    best_model_xg = grid_search.best_estimator_
    model_compare.append_model(best_model_xg,weight,time_elapsed_xg)

#%% run boost for each value of "scale_pos_weight"
scoring_list = [1,1.8,2.5,3.2]

for s in scoring_list:
    run_xgb(s)

model_compare.generate_metrics()
model_compare.plot_roc_curves("Plots/ROC_curves.png")
model_compare.df.to_excel("Tables/Compare_models.xlsx")
print("Finished!")

#%% extract confusion matrix

best_model = "scale_pos_weight: 2.5"
best_output_conf = model_compare.confusion_matrices[best_model]["Test"]
best_output_conf.to_excel("Tables/best confusion matrix.xlsx")

#%% recreate final best model for easy access
best_params = {
    'max_depth': 9,
    'eta': 0.3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'n_estimators': 100,
    'scale_pos_weight': 2.5,  # You should replace `weight` with the actual value you found most effective
    'tree_method': 'gpu_hist',  # Assuming you are using GPU acceleration; use 'hist' if on CPU
    'objective': 'binary:logistic',
    'seed': seed,
    'use_label_encoder': False
}
import xgboost as xgb
xgb_model_final = xgb.XGBClassifier(**best_params)
xgb_model_final.fit(X_train, y_train)
#%% plot trees

plt.figure(figsize=(20, 10))
xgb.plot_tree(xgb_model_final, num_trees=0)  # Last tree
plt.title("First Tree")
plt.savefig("Plots/first_tree.png", dpi = 3000)
plt.show()

plt.figure(figsize=(20, 10))
xgb.plot_tree(xgb_model_final, num_trees=best_params['n_estimators'] - 1)  # Last tree
plt.title("Last Tree")
plt.savefig("Plots/last_tree.png", dpi = 3000)
plt.show()
#%% importance

feature_importances = pd.Series(xgb_model_final.feature_importances_, index=X_train.columns)
feature_importances_df = feature_importances.reset_index()
feature_importances_df.columns = ['Feature', 'Importance']
feature_importances_df.to_excel("Tables/feature_importances.xlsx", index=False)