import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tools.eval_measures import aic

# Replace with actual DataFrame loading
# df = pd.read_csv('your_data.csv')

# Example logistic regression model fitting
# Formula: ejecta ~ Diameter + Big.Bin + Diameter:Big.Bin + Matern(1 | x + y)
# Convert to pandas DataFrame
# X = ...

# Logistic regression model for hypertension example
FData_Baseline_NOHYP = pd.DataFrame({
    'HYPERTEN': np.random.choice([0, 1], size=1000),
    'DIABETES': np.random.choice([0, 1], size=1000),
    'SEX': np.random.choice([1, 2], size=1000),
    'AGE': np.random.randint(30, 70, size=1000),
    'CIGPDAY': np.random.randint(0, 20, size=1000),
    'TOTCHOL': np.random.randint(150, 250, size=1000)
})

# Logistic Regression Model
FData_Baseline_NOHYP['SEX'] = FData_Baseline_NOHYP['SEX'].astype('category')
formula = 'HYPERTEN ~ DIABETES + C(SEX) + AGE + CIGPDAY + TOTCHOL + C(SEX)*DIABETES + C(SEX)*CIGPDAY'
LRModel = sm.formula.logit(formula, data=FData_Baseline_NOHYP).fit()

# Summary of the model
print(LRModel.summary())

# Hosmer-Lemeshow Test
from statsmodels.stats.diagnostic import linear_harvey_collier
hl_test = linear_harvey_collier(LRModel)
print("Hosmer-Lemeshow test statistic:", hl_test)

# Standardized deviance residuals
deviance_resid = LRModel.resid_dev
print("Five-number summary of deviance residuals:", np.percentile(deviance_resid, [0, 25, 50, 75, 100]))

# Scatter plot of deviance residuals versus AGE
sns.scatterplot(x=FData_Baseline_NOHYP['AGE'], y=deviance_resid)
plt.xlabel('AGE')
plt.ylabel('Deviance Residuals')
plt.title('Scatter Plot of Deviance Residuals versus AGE')
plt.show()

# Scatter plot of deviance residuals versus fitted values
sns.scatterplot(x=LRModel.fittedvalues, y=deviance_resid)
plt.xlabel('Fitted Values')
plt.ylabel('Deviance Residuals')
plt.title('Scatter Plot of Deviance Residuals versus Fitted Values')
plt.show()

# Odds ratios and confidence intervals
params = LRModel.params
conf = LRModel.conf_int()
conf['OR'] = np.exp(params)
conf.columns = ['2.5%', '97.5%', 'OR']
print("Odds Ratios and Confidence Intervals:\n", conf)

# Predicted values using the current sample
print("Summary of fitted values (predictions):", np.percentile(LRModel.fittedvalues, [0, 25, 50, 75, 100]))

# Prediction: Predicted values using new data
newPredictors = pd.DataFrame({
    'AGE': np.tile(np.arange(30, 75, 5), 4),
    'DIABETES': np.repeat([0, 1], 92),
    'SEX': np.repeat([1, 2], 92),
    'CIGPDAY': np.repeat([FData_Baseline_NOHYP['CIGPDAY'].mean()], 184),
    'TOTCHOL': np.repeat([FData_Baseline_NOHYP['TOTCHOL'].mean()], 184)
})
newPredictors['SEX'] = newPredictors['SEX'].astype('category')

predicted_values = LRModel.predict(newPredictors)
newPredictors['Predicted'] = predicted_values
print("Predicted values using new data:\n", newPredictors.head())

# Standard errors for predicted values
predictions = LRModel.get_prediction(newPredictors)
newPredictors['pred.full'] = predictions.predicted_mean
newPredictors['ymin'] = newPredictors['pred.full'] - 2 * predictions.se_mean
newPredictors['ymax'] = newPredictors['pred.full'] + 2 * predictions.se_mean

# Model diagnostics using DHARMa-like residuals (Simulated Residuals)
# Note: statsmodels does not directly provide DHARMa-like diagnostics, but similar diagnostics can be done.
# Here, we'll use residual plots and QQ plots for model diagnostics.

# Residuals vs predictors
sns.residplot(x=FData_Baseline_NOHYP['AGE'], y=deviance_resid, lowess=True)
plt.xlabel('AGE')
plt.ylabel('Deviance Residuals')
plt.title('Residuals vs AGE')
plt.show()

sns.residplot(x=FData_Baseline_NOHYP['CIGPDAY'], y=deviance_resid, lowess=True)
plt.xlabel('CIGPDAY')
plt.ylabel('Deviance Residuals')
plt.title('Residuals vs CIGPDAY')
plt.show()

# ROC Curve and AUC
fpr, tpr, thresholds = roc_curve(FData_Baseline_NOHYP['HYPERTEN'], LRModel.fittedvalues)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
predictions_binary = [1 if x > 0.5 else 0 for x in LRModel.fittedvalues]
conf_matrix = confusion_matrix(FData_Baseline_NOHYP['HYPERTEN'], predictions_binary)
print("Confusion Matrix:\n", conf_matrix)

# Summary of confusion matrix and additional metrics
print("Accuracy:", accuracy_score(FData_Baseline_NOHYP['HYPERTEN'], predictions_binary))
print("Classification Report:\n", classification_report(FData_Baseline_NOHYP['HYPERTEN'], predictions_binary))
