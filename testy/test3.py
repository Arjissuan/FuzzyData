from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
from src.fuzzy_functions import FuzzyMethods


# Define the custom scoring function for the G-measure
def g_measure(y_true, y_pred):
    # Calculate G-measure (harmonic mean of precision and recall)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    precision = tp / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    if precision + recall == 0:
        return 0
    g_measure = (2 * precision * recall) / (precision + recall)
    return g_measure


g_scorer = make_scorer(g_measure)


class FuzzyGridSearch(BaseEstimator, ClassifierMixin):
    def __init__(self, fuzzy_methods_instance, ph_limits, pi=0, delta=0):
        self.fuzzy_methods = fuzzy_methods_instance
        self.ph_limits = ph_limits
        self.pi = pi
        self.delta = delta
        self.classes_ = np.array([0, 1])  # Assuming binary classification

    def fit(self, X, y=None):
        # This method does nothing but is required by the sklearn API
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        return np.array(self.fuzzy_methods.fuzzy_interfence_system(X, self.pi, self.delta))

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_params(self, deep=False):
        return {'fuzzy_methods_instance': self.fuzzy_methods,
                'ph_limits': self.ph_limits,
                'pi': self.pi,
                'delta': self.delta}

# Load data
file_name = "FDA_data.xls"
fuzzy_methods = FuzzyMethods(file_name=file_name)

# Set pH limits for binary classification
ph_limits = {'normal': 7.2, 'abnormal': 7.1}

# Prepare the training data
X = fuzzy_methods.df[['Percentile', 'Apgar', 'Ph']]
y = (fuzzy_methods.df['Ph'] < ph_limits['abnormal']).astype(int)

# Instantiate the grid search object using the custom FuzzyGridSearch class
param_grid = {'pi': np.arange(-0.5, 0.6, 0.25), 'delta': np.arange(-0.5, 0.6, 0.25)}
grid_search = GridSearchCV(estimator=FuzzyGridSearch(fuzzy_methods_instance=fuzzy_methods, ph_limits=ph_limits), param_grid=param_grid, scoring=g_scorer, cv=KFold(n_splits=5))

# Perform grid search with 5-fold cross-validation
grid_search.fit(X, y)

# Display the best parameters and corresponding mean G-measure
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(f"Best Parameters: {best_params}")
print(f"Mean G-measure: {best_score:.4f}")
