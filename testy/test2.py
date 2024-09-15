import simpful as sf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold


# Define G-measure function
def g_measure(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    precision = tp / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    if precision + recall == 0:
        return 0
    g_measure = (2 * precision * recall) / (precision + recall)
    return g_measure


class FuzzyInferenceSystem:
    def __init__(self, data_file):
        # Load dataset
        self.dataset = pd.read_excel(io=os.path.join(os.getcwd(), data_file))

        # Initialize the fuzzy system
        self.FS = sf.FuzzySystem()

        # Define antecedent variables (input) for Percentile, Apgar, and Ph
        self.bw_values = self.dataset["Percentile"].values
        self.ap_values = self.dataset["Apgar"].values
        self.ph_values = self.dataset["Ph"].values

        # Set the universe of discourse
        self.bw_universe = np.linspace(min(self.bw_values) - 2, max(self.bw_values) + 2, 100)
        self.ap_universe = np.linspace(min(self.ap_values) - 2, max(self.ap_values) + 2, 100)
        self.ph_universe = np.linspace(min(self.ph_values) - 0.1, max(self.ph_values) + 0.1, 100)

        # Create fuzzy sets and variables
        self._create_fuzzy_variables()

        # Define the rules
        self._add_rules()

    def sigmoid_mf(self, x, c, a):
        """Returns a sigmoid membership function."""
        return 1 / (1 + np.exp(-a * (x - c)))

    def create_crossing_suspicious(self, universe, c_normal, c_abnormal, steepness_normal, steepness_abnormal):
        """
        Creates Normal, Suspicious, and Abnormal sigmoids with Suspicious crossing Normal and Abnormal at 0.5.
        """
        normal_sigmoid = self.sigmoid_mf(universe, c_normal, steepness_normal)  # Rising sigmoid for Normal
        abnormal_sigmoid = 1 - self.sigmoid_mf(universe, c_abnormal, steepness_abnormal)  # Falling sigmoid for Abnormal

        midpoint = (c_normal + c_abnormal) / 2
        suspicious_sigmoid = self.sigmoid_mf(universe, midpoint, -steepness_abnormal) * self.sigmoid_mf(universe,
                                                                                                        midpoint,
                                                                                                        steepness_normal)

        # Normalize the suspicious_sigmoid
        suspicious_sigmoid /= np.max(suspicious_sigmoid)

        return normal_sigmoid, suspicious_sigmoid, abnormal_sigmoid

    def _create_fuzzy_variables(self):
        """Internal method to create fuzzy linguistic variables."""
        # Generate fuzzy sets for Percentile (BW), Apgar (AP), and Ph (PH)
        BW_normal, BW_suspicious, BW_abnormal = self.create_crossing_suspicious(self.bw_universe, c_normal=10,
                                                                                c_abnormal=5,
                                                                                steepness_normal=2,
                                                                                steepness_abnormal=2)
        AP_normal, AP_suspicious, AP_abnormal = self.create_crossing_suspicious(self.ap_universe, c_normal=7,
                                                                                c_abnormal=5,
                                                                                steepness_normal=2,
                                                                                steepness_abnormal=2)
        PH_normal, PH_suspicious, PH_abnormal = self.create_crossing_suspicious(self.ph_universe, c_normal=7.2,
                                                                                c_abnormal=7.1,
                                                                                steepness_normal=20,
                                                                                steepness_abnormal=20)
        # Add FuzzySet objects to simpful system
        self.FS.add_linguistic_variable("Percentile", sf.LinguisticVariable([
            sf.FuzzySet(function=sf.Sigmoid_MF(c=10, a=2), term="Normal"),
            sf.FuzzySet(function=lambda x: np.interp(x, self.bw_universe, BW_suspicious), term="Suspicious"),
            sf.FuzzySet(function=sf.Sigmoid_MF(c=5, a=-2), term="Abnormal")
        ], universe_of_discourse=[min(self.bw_values), max(self.bw_values)]))

        self.FS.add_linguistic_variable("Apgar", sf.LinguisticVariable([
            sf.FuzzySet(function=sf.Sigmoid_MF(c=7, a=2), term="Normal"),
            sf.FuzzySet(function=lambda x: np.interp(x, self.ap_universe, AP_suspicious), term="Suspicious"),
            sf.FuzzySet(function=sf.Sigmoid_MF(c=5, a=-2), term="Abnormal")
        ], universe_of_discourse=[min(self.ap_values), max(self.ap_values)]))

        self.FS.add_linguistic_variable("Ph", sf.LinguisticVariable([
            sf.FuzzySet(function=sf.Sigmoid_MF(c=7.2, a=20), term="Normal"),
            sf.FuzzySet(function=lambda x: np.interp(x, self.ph_universe, PH_suspicious), term="Suspicious"),
            sf.FuzzySet(function=sf.Sigmoid_MF(c=7.1, a=-20), term="Abnormal")
        ], universe_of_discourse=[min(self.ph_values), max(self.ph_values)]))

        # Define crisp output values for Sugeno inference
        self.FS.set_crisp_output_value("Abnormal_output", 1.0)
        self.FS.set_crisp_output_value("Suspicious_output", 0.5)
        self.FS.set_crisp_output_value("Normal_output", 0.0)

    def _add_rules(self):
        """Internal method to add rules to the fuzzy system."""
        self.FS.add_rules([
            "IF (Apgar IS Abnormal) AND (Percentile IS Abnormal) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Abnormal) AND (Percentile IS Abnormal) AND (Ph IS Normal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Abnormal) AND (Percentile IS Abnormal) AND (Ph IS Suspicious) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Abnormal) AND (Percentile IS Normal) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Abnormal) AND (Percentile IS Normal) AND (Ph IS Normal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Abnormal) AND (Percentile IS Normal) AND (Ph IS Suspicious) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Abnormal) AND (Percentile IS Suspicious) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Abnormal) AND (Percentile IS Suspicious) AND (Ph IS Normal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Abnormal) AND (Percentile IS Suspicious) AND (Ph IS Suspicious) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Normal) AND (Percentile IS Abnormal) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Normal) AND (Percentile IS Abnormal) AND (Ph IS Normal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Normal) AND (Percentile IS Abnormal) AND (Ph IS Suspicious) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Normal) AND (Percentile IS Normal) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Normal) AND (Percentile IS Normal) AND (Ph IS Normal) THEN (output IS Normal_output)",
            "IF (Apgar IS Normal) AND (Percentile IS Normal) AND (Ph IS Suspicious) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Normal) AND (Percentile IS Suspicious) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Normal) AND (Percentile IS Suspicious) AND (Ph IS Normal) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Normal) AND (Percentile IS Suspicious) AND (Ph IS Suspicious) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Abnormal) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Abnormal) AND (Ph IS Normal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Abnormal) AND (Ph IS Suspicious) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Normal) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Normal) AND (Ph IS Normal) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Normal) AND (Ph IS Suspicious) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Suspicious) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Suspicious) AND (Ph IS Normal) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Suspicious) AND (Ph IS Suspicious) THEN (output IS Abnormal_output)"
        ])

    def set_inputs(self, percentile, apgar, ph):
        """Set input values for the fuzzy system."""
        self.FS.set_variable("Percentile", percentile)
        self.FS.set_variable("Apgar", apgar)
        self.FS.set_variable("Ph", ph)

    def perform_inference(self):
        """Perform fuzzy inference and return the result."""
        result = self.FS.Sugeno_inference()
        return result['output']

    def plot_memberships(self):
        """Plot membership functions for Percentile, Apgar, and Ph."""
        # Plot for Percentile
        plt.figure()
        BW_normal, BW_suspicious, BW_abnormal = self.create_crossing_suspicious(self.bw_universe, c_normal=10,
                                                                                c_abnormal=5, steepness_normal=2,
                                                                                steepness_abnormal=2)
        plt.plot(self.bw_universe, BW_normal, label="Normal")
        plt.plot(self.bw_universe, BW_suspicious, label="Suspicious")
        plt.plot(self.bw_universe, BW_abnormal, label="Abnormal")
        plt.title("Percentile Membership Functions (Suspicious Crossing at 0.5)")
        plt.legend()

        # Plot for Apgar
        plt.figure()
        AP_normal, AP_suspicious, AP_abnormal = self.create_crossing_suspicious(self.ap_universe, c_normal=7,
                                                                                c_abnormal=5, steepness_normal=2,
                                                                                steepness_abnormal=2)
        plt.plot(self.ap_universe, AP_normal, label="Normal")
        plt.plot(self.ap_universe, AP_suspicious, label="Suspicious")
        plt.plot(self.ap_universe, AP_abnormal, label="Abnormal")
        plt.title("Apgar Membership Functions (Suspicious Crossing at 0.5)")
        plt.legend()

        # Plot for Ph
        plt.figure()
        PH_normal, PH_suspicious, PH_abnormal = self.create_crossing_suspicious(self.ph_universe, c_normal=7.2,
                                                                                c_abnormal=7.1, steepness_normal=20,
                                                                                steepness_abnormal=20)
        plt.plot(self.ph_universe, PH_normal, label="Normal")
        plt.plot(self.ph_universe, PH_suspicious, label="Suspicious")
        plt.plot(self.ph_universe, PH_abnormal, label="Abnormal")
        plt.title("Ph Membership Functions (Suspicious Crossing at 0.5)")
        plt.legend()

        plt.show()


class FuzzyGridSearch(BaseEstimator, ClassifierMixin):
    def __init__(self, fuzzy_methods_instance, ph_limits, pi=0, delta=0):
        self.fuzzy_methods = fuzzy_methods_instance
        self.ph_limits = ph_limits
        self.pi = pi
        self.delta = delta
        self.classes_ = np.array([0, 1])  # Assuming binary classification

    def fit(self, X, y=None):
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        predictions = []
        for x in X:
            self.fuzzy_methods.set_inputs(x[0], x[1], x[2])
            prediction = self.fuzzy_methods.perform_inference()
            predictions.append(prediction)
        return np.array(predictions)

    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

    def get_params(self, deep=False):
        return {'fuzzy_methods_instance': self.fuzzy_methods,
                'ph_limits': self.ph_limits,
                'pi': self.pi,
                'delta': self.delta}


def grid_search_fuzzy_system(data_file, ph_limits):
    # Load dataset
    dataset = pd.read_excel(io=data_file)

    # Create an Outcome column based on Ph values
    threshold = 7.2  # Define your threshold for abnormal outcome
    dataset['Outcome'] = (dataset['Ph'] < threshold).astype(int)  # 1 for Abnormal, 0 for Normal

    # Prepare data
    X = dataset[["Percentile", "Apgar", "Ph"]].values
    y = dataset["Outcome"].values  # Outcome column now available

    # Initialize Fuzzy Inference System
    fuzzy_system = FuzzyInferenceSystem(data_file)

    # Define the parameter grid for pi and delta
    pi_values = np.arange(-0.50, 0.51, 0.25)
    delta_values = np.arange(-0.50, 0.51, 0.25)

    best_score = 0
    best_params = {'pi': 0, 'delta': 0}

    # Perform 5-fold cross-validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for pi in pi_values:
        for delta in delta_values:
            # Initialize the grid search instance
            grid_search = FuzzyGridSearch(fuzzy_system, ph_limits, pi, delta)

            scores = []
            for train_index, test_index in kf.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                # Fit the model (not actually used but required by sklearn API)
                grid_search.fit(X_train, y_train)

                # Predict and calculate the score
                y_pred = grid_search.predict(X_test)
                score = g_measure(y_test, y_pred)
                scores.append(score)

            mean_score = np.mean(scores)
            print(f"pi: {pi}, delta: {delta}, mean G-measure: {mean_score}")

            if mean_score > best_score:
                best_score = mean_score
                best_params = {'pi': pi, 'delta': delta}

    print(f"Best parameters: {best_params}")
    print(f"Best G-measure: {best_score}")


# Example usage
ph_limits = (7.0, 7.5)  # Example pH limits; adjust as needed
grid_search_fuzzy_system("FDA_data.xls", ph_limits)
