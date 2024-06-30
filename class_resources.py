import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

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