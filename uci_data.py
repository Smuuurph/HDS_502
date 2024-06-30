from ucimlrepo import fetch_ucirepo
import pandas as pd
# fetch dataset
estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition = fetch_ucirepo(id=544)

# data (as pandas dataframes)
X = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.features
y = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.data.targets

df = pd.concat([X, y], axis=1)


# metadata
meta_data = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.metadata

# variable information
variables = estimation_of_obesity_levels_based_on_eating_habits_and_physical_condition.variables

#%%
