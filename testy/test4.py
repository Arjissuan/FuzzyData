from src.fuzzy_functions import FuzzyMethods
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

# Define the grid search space for pi and delta
param_grid = {
    'pi': np.arange(-0.5, 0.51, 0.25),
    'delta': np.arange(-0.5, 0.51, 0.25)
}

# Define cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Function to compute G-measure (F1-score in this case)
def g_measure(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

best_score = -1
best_params = None
FM = FuzzyMethods()
for pi in param_grid['pi']:
    for delta in param_grid['delta']:
        scores = []
        for train_index, test_index in kf.split(FM.df):
            train_data, test_data = FM.df.iloc[train_index], FM.df.iloc[test_index]
            y_pred = FM.fuzzy_interfence_system(test_data, pi, delta)
            score = g_measure(FM.fuzzy_interfence_system(test_data, 0, 1), y_pred)
            scores.append(score)

        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {'pi': pi, 'delta': delta}
        print(mean_score, pi, delta)

print("Best Params:", best_params)
print("Best Consistency Score:", best_score)