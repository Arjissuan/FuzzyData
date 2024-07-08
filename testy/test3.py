from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd
import os
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

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

class FuzzyMethods:
    def __init__(self, file_name="FDA_data.xls"):
        self.df = pd.read_excel(io=os.path.join(os.getcwd(), file_name))
        self.BW = ctrl.Antecedent(np.linspace(np.min(self.df.iloc[:, 0]), np.max(self.df.iloc[:, 0]), 1000), label="Percentile")
        self.AP = ctrl.Antecedent(np.linspace(np.min(self.df.iloc[:, 1]), np.max(self.df.iloc[:, 1]), 1000), label="Apgar")
        self.ph = ctrl.Antecedent(np.linspace(np.min(self.df.iloc[:, 2]), np.max(self.df.iloc[:, 2]), 1000), label="Ph")
        self.outcome = ctrl.Consequent(np.arange(0, 3.01, 0.01), 'outcome')
        self.labels = ("Normal", "Suspicious", "Abnormal")

    def membership_fun_ph(self, pi=0):
        peha_norm = self.df.loc[self.df["Ph"] >= 7.2, "Ph"]
        peha_abnm = self.df.loc[self.df['Ph'] < 7.1, "Ph"]
        peha_sussy = self.df.loc[(self.df.loc[:, 'Ph'] >= 7.1) & (self.df.loc[:, 'Ph'] <= 7.2), "Ph"]

        mean_PH_normal = float(np.mean(peha_norm)) if not peha_norm.empty else 0
        std_PH_normal = float(np.std(peha_norm) + 1e-6) if not peha_norm.empty else 1e-6
        mean_PH_sussy = float(np.mean(peha_sussy)) if not peha_sussy.empty else 0
        std_PH_sussy = float(np.std(peha_sussy) + 1e-6) if not peha_sussy.empty else 1e-6
        mean_PH_abnormal = float(np.mean(peha_abnm)) if not peha_abnm.empty else 0
        std_PH_abnormal = float(np.std(peha_abnm) + 1e-6) if not peha_abnm.empty else 1e-6

        self.ph[self.labels[0]] = fuzz.gaussmf(self.ph.universe, mean_PH_normal + pi,  std_PH_normal)
        self.ph[self.labels[1]] = fuzz.gaussmf(self.ph.universe, mean_PH_sussy + pi, std_PH_sussy)
        self.ph[self.labels[2]] = fuzz.gaussmf(self.ph.universe, mean_PH_abnormal + pi, std_PH_abnormal)

    def membership_fun_BW(self, pi=0):
        bewu_norm = self.df.loc[self.df["Percentile"] >= 10, "Percentile"]
        bewu_abnm = self.df.loc[self.df['Percentile'] <= 5, 'Percentile']
        bewu_sussy = self.df.loc[(self.df['Percentile'] < 10) & (self.df['Percentile'] > 5), 'Percentile']

        mean_BW_normal = float(np.mean(bewu_norm)) if not bewu_norm.empty else 0
        std_BW_normal = float(np.std(bewu_norm)) if not bewu_norm.empty else 1e-6
        mean_BW_sussy = float(np.mean(bewu_sussy)) if not bewu_sussy.empty else 0
        std_BW_sussy = float(np.std(bewu_sussy)) if not bewu_sussy.empty else 1e-6
        mean_BW_abnormal = float(np.mean(bewu_abnm)) if not bewu_abnm.empty else 0
        std_BW_abnormal = float(np.std(bewu_abnm) + 1e-6) if not bewu_abnm.empty else 1e-6

        self.BW[self.labels[0]] = fuzz.gaussmf(self.BW.universe, mean_BW_normal + pi, std_BW_normal)
        self.BW[self.labels[1]] = fuzz.gaussmf(self.BW.universe, mean_BW_sussy + pi, std_BW_sussy)
        self.BW[self.labels[2]] = fuzz.gaussmf(self.BW.universe, mean_BW_abnormal + pi, std_BW_abnormal)

    def membership_fun_AP(self, pi=0):
        ape_norm = self.df.loc[self.df["Apgar"] >= 7, "Apgar"]
        ape_abnm = self.df.loc[self.df['Apgar'] < 5, 'Apgar']
        ape_sussy = self.df.loc[(self.df['Apgar'] < 7) & (self.df['Apgar'] >= 5), 'Apgar']

        mean_AP_normal = float(np.mean(ape_norm)) if not ape_norm.empty else 0
        std_AP_normal = float(np.std(ape_norm) + 1e-6) if not ape_norm.empty else 1e-6
        mean_AP_sussy = float(np.mean(ape_sussy)) if not ape_sussy.empty else 0
        std_AP_sussy = float(np.std(ape_sussy) + 1e-6) if not ape_sussy.empty else 1e-6
        mean_AP_abnormal = float(np.mean(ape_abnm)) if not ape_abnm.empty else 0
        std_AP_abnormal = float(np.std(ape_abnm) + 1e-6) if not ape_abnm.empty else 1e-6

        self.AP[self.labels[0]] = fuzz.gaussmf(self.AP.universe, mean_AP_normal + pi, std_AP_normal)
        self.AP[self.labels[1]] = fuzz.gaussmf(self.AP.universe, mean_AP_sussy + pi, std_AP_sussy)
        self.AP[self.labels[2]] = fuzz.gaussmf(self.AP.universe, mean_AP_abnormal + pi, std_AP_abnormal)

    def make_plots(self):
        self.ph.view()
        self.BW.view()
        self.AP.view()

        plt.show()

    def rules(self):
        self.outcome[self.labels[2]] = fuzz.trapmf(self.outcome.universe, [0, 0, 0.5, 1])
        self.outcome[self.labels[1]] = fuzz.trapmf(self.outcome.universe, [1, 1, 1.5, 2])
        self.outcome[self.labels[0]] = fuzz.trapmf(self.outcome.universe, [2, 2, 2.5, 3])

        rule1 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[2]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule2 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[2]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule3 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[2]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])
        rule4 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[0]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule5 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[0]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule6 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[0]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])
        rule7 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[1]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule8 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[1]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule9 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[1]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])
        rule10 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[2]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule11 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[2]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule12 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[2]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])
        rule13 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[0]] & self.ph[self.labels[2]], self.outcome[self.labels[1]])
        rule14 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[0]] & self.ph[self.labels[0]], self.outcome[self.labels[1]])
        rule15 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[0]] & self.ph[self.labels[1]], self.outcome[self.labels[1]])
        rule16 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[1]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule17 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[1]] & self.ph[self.labels[0]], self.outcome[self.labels[1]])
        rule18 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[1]] & self.ph[self.labels[1]], self.outcome[self.labels[1]])
        rule19 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[2]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule20 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[2]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule21 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[2]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])
        rule22 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[0]] & self.ph[self.labels[2]], self.outcome[self.labels[1]])
        rule23 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[0]] & self.ph[self.labels[0]], self.outcome[self.labels[1]])
        rule24 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[0]] & self.ph[self.labels[1]], self.outcome[self.labels[1]])
        rule25 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[1]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule26 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[1]] & self.ph[self.labels[0]], self.outcome[self.labels[1]])
        rule27 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[1]] & self.ph[self.labels[1]], self.outcome[self.labels[1]])
        rule28 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[0]] & self.ph[self.labels[0]], self.outcome[self.labels[1]])
        rule29 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[1]] & self.ph[self.labels[0]], self.outcome[self.labels[1]])
        rule30 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[0]] & self.ph[self.labels[2]], self.outcome[self.labels[1]])

        rules = [rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9,
                 rule10, rule11, rule12, rule13, rule14, rule15, rule16, rule17, rule18,
                 rule19, rule20, rule21, rule22, rule23, rule24, rule25, rule26, rule27,
                 rule28, rule29, rule30]

        return rules

    def train(self, input_data, pi=0, delta=0):
        self.membership_fun_ph(pi)
        self.membership_fun_AP(pi)
        self.membership_fun_BW(pi)
        rules = self.rules()
        control = ctrl.ControlSystem(rules)
        sim = ctrl.ControlSystemSimulation(control)
        output_data = np.zeros(input_data.shape[0])

        for i in range(input_data.shape[0]):
            sim.input['Percentile'] = input_data.iloc[i, 0]
            sim.input['Apgar'] = input_data.iloc[i, 1]
            sim.input['Ph'] = input_data.iloc[i, 2]

            sim.compute()
            output_data[i] = sim.output['outcome']

        # Apply delta to the outcomes
        thresholds = {
            "Normal": 2 + delta,
            "Suspicious": 1 + delta,
            "Abnormal": 0 + delta
        }
        predictions = np.digitize(output_data, bins=[thresholds["Abnormal"], thresholds["Suspicious"], thresholds["Normal"]])
        return predictions

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
        return np.array(self.fuzzy_methods.train(X, self.pi, self.delta))

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
