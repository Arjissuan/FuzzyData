import simpful as sf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


# Define G-measure function
def g_measure(y_true, y_pred):
    """
    Calculate G-measure directly using continuous predictions.
    """
    # Since y_true can be 0, 0.5, or 1, we can keep the prediction as continuous.
    tp = np.sum((y_true == 1) & (y_pred >= 0.5))  # True positives
    precision = tp / np.sum(y_pred >= 0.5) if np.sum(y_pred >= 0.5) > 0 else 0
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
        results = []

        # Reinitialize the fuzzy system for each inference run
        self.FS = sf.FuzzySystem()
        self._create_fuzzy_variables()  # Re-add fuzzy variables
        self._add_rules()  # Re-add rules

        # Instead of adjusting rules, adjust the output after inference
        for i in range(len(test_data)):
            # Set variables for the current instance
            self.FS.set_variable("Percentile", test_data["Percentile"].values[i])
            self.FS.set_variable("Apgar", test_data["Apgar"].values[i])
            self.FS.set_variable("Ph", test_data["Ph"].values[i])

            try:
                # Perform Sugeno inference
                output_dict = self.FS.Sugeno_inference(["output"])

                # Ensure "output" exists in the result and add delta
                if "output" in output_dict:
                    output = output_dict["output"]
                    # Apply the p_i and delta adjustment after inference
                    output = (p_i * output) + delta
                    # Debugging line to inspect the output
                    # print(f"Inference result for entry {i}: {output}")
                else:
                    print(f"Error: Output key not found in inference result: {output_dict}")
                    output = 0  # Set fallback output in case of issues
            except Exception as e:
                print(f"Error during inference: {e}")
                output = 0  # Fallback value in case of error

            results.append(output)

        return np.array(results)


class FuzzyGridSearch(BaseEstimator, ClassifierMixin):
    def __init__(self, data_file, p_i_range=None, delta_range=None):
        self.data_file = data_file
        self.p_i_range = p_i_range if p_i_range else np.arange(-0.5, 0.6, 0.25)
        self.delta_range = delta_range if delta_range else np.arange(-0.5, 0.6, 0.25)

    def fit(self, X=None, y=None):
        # Load dataset into DataFrame
        self.dataset = pd.read_excel(io=os.path.join(os.getcwd(), self.data_file))
        return self

    def grid_search(self, k=5):
        kf = KFold(n_splits=k)
        best_avg_score = 0
        best_p_i = None
        best_delta = None
        best_fold_scores = []

        # Track the maximum G-measure across all folds and parameter sets
        best_individual_g_measure = 0

        # Simulate target values as an array
        target_data = np.random.choice([0, 0.5, 1], size=len(self.dataset), p=[0.6, 0.3, 0.1])

        for p_i in self.p_i_range:
            for delta in self.delta_range:
                fold_scores = []  # To track G-measure for each fold
                for train_index, test_index in kf.split(self.dataset):
                    train_data, test_data = self.dataset.iloc[train_index], self.dataset.iloc[test_index]

                    # Create Fuzzy Inference System with training data
                    fis = FuzzyInferenceSystem(train_data)

                    # Perform inference only on test data
                    y_pred = fis.infer(test_data, p_i=p_i, delta=delta)

                    # Evaluate using G-measure
                    score = g_measure(target_data[test_index], y_pred)
                    fold_scores.append(score)

                    # Track the highest individual fold G-measure score
                    if score > best_individual_g_measure:
                        best_individual_g_measure = score

                # Calculate average G-measure for this parameter combination
                avg_score = np.mean(fold_scores)

                # If this combination yields a better average score, update best parameters and scores
                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_p_i = p_i
                    best_delta = delta
                    best_fold_scores = fold_scores

        # Save the best parameters, best average score, and the mean of fold scores for best params
        self.best_p_i = best_p_i
        self.best_delta = best_delta
        self.best_avg_score = best_avg_score  # Best average G-measure (mean across folds)
        self.best_fold_scores = best_fold_scores  # Fold-wise G-measure scores for best params
        self.best_individual_g_measure = best_individual_g_measure  # Best individual G-measure

        # Return the best parameters and scores
        return best_p_i, best_delta, best_avg_score, best_individual_g_measure


# Example usage
data_file = "FDA_data.xls"
fuzzy_grid_search = FuzzyGridSearch(data_file=data_file)
fuzzy_grid_search.fit()  # Ensure dataset is loaded
best_p_i, best_delta, best_avg_score, mean_g_measure = fuzzy_grid_search.grid_search()

print("Best pi:", best_p_i)
print("Best delta:", best_delta)
print("Best Average G-Measure Score:", best_avg_score)
print("Mean G-Measure Score for Best Parameters:", mean_g_measure)

