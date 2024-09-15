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
    def __init__(self, dataset):
        # Use the provided dataset
        self.dataset = dataset

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

        # Define the rules (without pi and delta yet)
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

        # Define crisp output values for Sugeno inference (modify with delta)
        self.FS.set_crisp_output_value("Abnormal_output", 1.0)
        self.FS.set_crisp_output_value("Suspicious_output", 0.5)
        self.FS.set_crisp_output_value("Normal_output", 0.0)

    def _add_rules(self):
        """Internal method to add rules to the fuzzy system."""
        self.rules = [
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
            "IF (Apgar IS Suspicious) AND (Percentile IS Abnormal) AND (Ph IS Normal) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Abnormal) AND (Ph IS Suspicious) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Normal) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Normal) AND (Ph IS Normal) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Normal) AND (Ph IS Suspicious) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Suspicious) AND (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Suspicious) AND (Ph IS Normal) THEN (output IS Suspicious_output)",
            "IF (Apgar IS Suspicious) AND (Percentile IS Suspicious) AND (Ph IS Suspicious) THEN (output IS Suspicious_output)",
        ]
        self.FS.add_rules(self.rules)

    def infer(self, test_data, p_i=1, delta=0):
        """Perform fuzzy inference using specified parameters on test data."""
        # Prepare the adjusted rules with pi and delta modifications
        adjusted_rules = []
        for rule in self.rules:
            adjusted_rule = rule
            if "Suspicious_output" in rule:
                # Adjust pi for suspicious rules
                adjusted_rule = adjusted_rule.replace("Suspicious_output",
                                                      f"({p_i} * Suspicious_output + {delta})")
            adjusted_rules.append(adjusted_rule)

        # Clear the existing rules (if applicable)
        self.FS._rules = []  # Attempt to clear rules, verify this is supported in simpful

        # Add the adjusted rules back to the fuzzy system
        for rule in adjusted_rules:
            # Example: Add rules one by one
            self.FS.add_rules([rule])

        # Perform fuzzy inference on the test data
        results = []
        for i in range(len(test_data)):
            self.FS.set_variable("Percentile", test_data["Percentile"].values[i])
            self.FS.set_variable("Apgar", test_data["Apgar"].values[i])
            self.FS.set_variable("Ph", test_data["Ph"].values[i])

            # Get crisp output after applying delta
            try:
                output_dict = self.FS.Sugeno_inference(["output"])
                output = output_dict["output"] + delta

                # Optional: Print each output for debugging purposes
                print(f"Inference result for entry {i}: {output}")  # Debugging line
            except Exception as e:
                print(f"Error during inference: {e}")
                output = 0  # Fallback value in case of error

            results.append(output)

        return np.array(results)


class FuzzyGridSearch(BaseEstimator, ClassifierMixin):
    def __init__(self, data_file, p_i_range=None, delta_range=None):
        self.data_file = data_file
        self.p_i_range = p_i_range if p_i_range else np.linspace(0.5, 1.5, 10)
        self.delta_range = delta_range if delta_range else np.linspace(-0.1, 0.1, 5)

    def fit(self, X=None, y=None):
        # Load dataset into DataFrame
        self.dataset = pd.read_excel(io=os.path.join(os.getcwd(), self.data_file))
        return self

    def grid_search(self, k=5):
        kf = KFold(n_splits=k)
        best_score = 0
        best_p_i = None
        best_delta = None

        # Simulate target values as an array
        target_data = np.random.choice([0, 0.5, 1], size=len(self.dataset), p=[0.6, 0.3, 0.1])

        for p_i in self.p_i_range:
            for delta in self.delta_range:
                scores = []
                for train_index, test_index in kf.split(self.dataset):
                    train_data, test_data = self.dataset.iloc[train_index], self.dataset.iloc[test_index]

                    # Create Fuzzy Inference System with training data
                    fis = FuzzyInferenceSystem(train_data)

                    # Perform inference only on test data
                    y_pred = fis.infer(test_data, p_i=p_i, delta=delta)

                    # Evaluate using G-measure
                    score = g_measure(target_data[test_index], y_pred)
                    scores.append(score)

                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_p_i = p_i
                    best_delta = delta

        # Save the best parameters and score as attributes of the object
        self.best_p_i = best_p_i
        self.best_delta = best_delta
        self.best_score = best_score

        return best_p_i, best_delta, best_score

# Example usage
data_file = "FDA_data.xls"
fuzzy_grid_search = FuzzyGridSearch(data_file=data_file)
fuzzy_grid_search.fit()  # Ensure dataset is loaded
fuzzy_grid_search.grid_search()

print("Best pi:", fuzzy_grid_search.best_p_i)
print("Best delta:", fuzzy_grid_search.best_delta)
print("Best score:", fuzzy_grid_search.best_score)
