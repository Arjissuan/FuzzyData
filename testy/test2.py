import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the universe of discourse for each variable
percentile = ctrl.Antecedent(np.arange(0, 101, 1), 'percentile')
apgar = ctrl.Antecedent(np.arange(0, 11, 1), 'apgar')
ph = ctrl.Antecedent(np.arange(6.5, 8.0, 0.01), 'ph')
outcome = ctrl.Consequent(np.arange(0, 4, 1), 'outcome')

# Define fuzzy sets and membership functions for each input
percentile['low'] = fuzz.trapmf(percentile.universe, [0, 0, 20, 40])
percentile['medium'] = fuzz.trimf(percentile.universe, [30, 50, 70])
percentile['high'] = fuzz.trapmf(percentile.universe, [60, 80, 100, 100])

apgar['low'] = fuzz.trapmf(apgar.universe, [0, 0, 3, 5])
apgar['medium'] = fuzz.trimf(apgar.universe, [4, 6, 8])
apgar['high'] = fuzz.trapmf(apgar.universe, [7, 9, 10, 10])

ph['low'] = fuzz.trapmf(ph.universe, [6.5, 6.5, 7.0, 7.2])
ph['normal'] = fuzz.trimf(ph.universe, [7.1, 7.3, 7.4])
ph['high'] = fuzz.trapmf(ph.universe, [7.3, 7.5, 8.0, 8.0])

# Define the output fuzzy set
outcome['bad'] = fuzz.trapmf(outcome.universe, [0, 0, 1, 1.5])
outcome['average'] = fuzz.trapmf(outcome.universe, [1, 1.5, 2.5, 3])
outcome['good'] = fuzz.trapmf(outcome.universe, [2, 2.5, 3, 3])

# Define fuzzy rules
rule1 = ctrl.Rule(percentile['low'] & apgar['low'] & ph['low'], outcome['bad'])
rule2 = ctrl.Rule(percentile['medium'] & apgar['medium'] & ph['normal'], outcome['average'])
rule3 = ctrl.Rule(percentile['high'] & apgar['high'] & ph['high'], outcome['good'])
rule4 = ctrl.Rule(apgar['low'] | ph['low'], outcome['bad'])
rule5 = ctrl.Rule(apgar['high'] & ph['normal'], outcome['average'])

# Create control system and simulation
outcome_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
outcome_sim = ctrl.ControlSystemSimulation(outcome_ctrl)

# Example input
example_data = {
    'percentile': 9,
    'apgar': 9,
    'ph': 7.6
}

# Fuzzification
outcome_sim.input['percentile'] = example_data['percentile']
outcome_sim.input['apgar'] = example_data['apgar']
outcome_sim.input['ph'] = example_data['ph']

# Perform inference
outcome_sim.compute()

# Get the result
result = outcome_sim.output['outcome']
print(f"Fuzzy outcome: {result}")

# Output the crisp result
classification = 'bad' if result < 1 else 'average' if result < 2.5 else 'good'
print(f"Defuzzified result (classification): {classification}")

from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import make_scorer, precision_recall_fscore_support
from sklearn.base import BaseEstimator, ClassifierMixin
import pandas as pd

class FuzzyInferenceSystem(BaseEstimator, ClassifierMixin):
    def __init__(self, p_i=0.0, delta=0.0):
        self.p_i = p_i
        self.delta = delta

    def fit(self, X, y):
        return self

    def predict(self, X):
        results = []
        for sample in X:
            percentile, apgar, ph = sample
            outcome_sim.input['percentile'] = percentile
            outcome_sim.input['apgar'] = apgar
            outcome_sim.input['ph'] = ph
            outcome_sim.compute()
            result = outcome_sim.output['outcome']
            results.append(1 if result >= 0.5 else 0)  # Classify as good (1) or bad (0)
        return np.array(results)


def g_measure(y_true, y_pred):
    precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    return 2 * (precision * recall) / (precision + recall)


# Define the model and scoring function
scorer = make_scorer(g_measure)
param_range = np.arange(-0.50, 0.51, 0.25)
best_g_measure = 0
best_params = {'p(i)': None, 'Delta': None}

# Convert dataset to numpy arrays
data = pd.read_excel("../FDA_data.xls")
X = data[['Percentile', 'Apgar', 'Ph']].values
y = (data['Ph'] > 7.2).astype(int).values  # Binary outcome based on pH level

# Grid search
for p_i in param_range:
    for delta in param_range:
        model = FuzzyInferenceSystem(p_i=p_i, delta=delta)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
        mean_score = np.mean(scores)
        if mean_score > best_g_measure:
            best_g_measure = mean_score
            best_params = {'p(i)': p_i, 'Delta': delta}

print(f'Best parameters: {best_params}')
print(f'Highest mean G-measure: {best_g_measure}')
