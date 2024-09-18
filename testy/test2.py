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

        # Define the pH variable only
        self.ph_values = self.dataset["Ph"].values

        # Set the universe of discourse for pH
        self.ph_universe = np.linspace(min(self.ph_values) - 0.1, max(self.ph_values) + 0.1, 100)

        # Create fuzzy sets and variables for pH only
        self._create_fuzzy_variables()

        # Define the rules for pH only
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
        """Internal method to create fuzzy linguistic variables for pH."""
        # Generate fuzzy sets for Ph (PH) only
        PH_normal, PH_suspicious, PH_abnormal = self.create_crossing_suspicious(self.ph_universe, c_normal=7.2,
                                                                                c_abnormal=7.1,
                                                                                steepness_normal=20,
                                                                                steepness_abnormal=20)
        # Add FuzzySet objects to simpful system
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
            "IF (Ph IS Abnormal) THEN (output IS Abnormal_output)",
            "IF (Ph IS Normal) THEN (output IS Normal_output)",
            "IF (Ph IS Suspicious) THEN (output IS Suspicious_output)",
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
            # Set the Ph variable for the current instance
            self.FS.set_variable("Ph", test_data["Ph"].values[i])

            try:
                # Perform Sugeno inference
                output_dict = self.FS.Sugeno_inference(["output"])

                # Ensure "output" exists in the result and add delta
                if "output" in output_dict:
                    output = output_dict["output"]
                    # Apply the p_i and delta adjustment after inference
                    output = (p_i * output) + delta
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

        # Create target values based on pH values

        target_data = np.where(self.dataset["Ph"] < 7.2, 1, 0)

        print(target_data)
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

                avg_score = np.mean(fold_scores)

                if avg_score > best_avg_score:
                    best_avg_score = avg_score
                    best_p_i = p_i
                    best_delta = delta
                    best_fold_scores = fold_scores

        print("Best Average G-measure:", best_avg_score)
        print("Best Individual G-measure:", best_individual_g_measure)
        print(f"Best Parameters: p_i={best_p_i}, delta={best_delta}")

        return best_avg_score, best_individual_g_measure, best_p_i, best_delta, best_fold_scores


# Example usage
data_file = "FDA_data.xls"
fuzzy_grid_search = FuzzyGridSearch(data_file=data_file)
fuzzy_grid_search.fit()  # Ensure dataset is loaded
best_avg_score, best_individual_g_measure, best_p_i, best_delta, best_fold_scores = fuzzy_grid_search.grid_search()



