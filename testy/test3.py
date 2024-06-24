import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
from fuzzy_functions import FuzzyMethods


def evaluate_fuzzy_system(fuzzy_system, input_data, pH_limits):
    # Perform fuzzy inference
    fuzzy_score = fuzzy_system.train(input_data)

    # Convert fuzzy_score to binary outcome based on pH limits
    binary_fuzzy_score = 1 if fuzzy_score < pH_limits['lower'] or fuzzy_score > pH_limits['upper'] else 0
    actual_outcome = 1 if input_data['Ph'] < pH_limits['lower'] or input_data['Ph'] > pH_limits['upper'] else 0

    return binary_fuzzy_score, actual_outcome


def grid_search_fuzzy(fuzzy_system, data, pH_limits):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_g_measure = 0
    best_params = None

    for (pi, delta) in parameter_grid:
        g_measures = []

        for train_index, test_index in kf.split(data):
            train_data = data.iloc[train_index]
            test_data = data.iloc[test_index]

            # Train fuzzy system with the current parameters pi and delta
            fuzzy_system.train(train_data, pi, delta)  # You need to implement this method

            # Evaluate on test set
            binary_predictions = []
            actual_outcomes = []

            for idx, row in test_data.iterrows():
                input_data = {'Percentile': row['Percentile'], 'Apgar': row['Apgar'], 'Ph': row['Ph']}
                binary_fuzzy_score, actual_outcome = evaluate_fuzzy_system(fuzzy_system, input_data, pH_limits)
                binary_predictions.append(binary_fuzzy_score)
                actual_outcomes.append(actual_outcome)

            # Calculate precision, recall, and G-measure
            precision = precision_score(actual_outcomes, binary_predictions)
            recall = recall_score(actual_outcomes, binary_predictions)
            g_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            g_measures.append(g_measure)

        mean_g_measure = np.mean(g_measures)
        if mean_g_measure > best_g_measure:
            best_g_measure = mean_g_measure
            best_params = (pi, delta)

    return best_params, best_g_measure

# Define the range of p(i) and ∆
pi_range = np.arange(-0.50, 0.51, 0.25)
delta_range = np.arange(-0.50, 0.51, 0.25)
parameter_grid = [(pi, delta) for pi in pi_range for delta in delta_range]

fuzzy_system = FuzzyMethods()
fuzzy_system.membership_fun_ph()
fuzzy_system.membership_fun_BW()
fuzzy_system.membership_fun_AP()

pH_limits = {'lower': 7.1, 'upper': 7.2}  # These limits should be provided by your teacher
data = fuzzy_system.df  # Assuming this is the dataset

best_params, best_g_measure = grid_search_fuzzy(fuzzy_system, data, pH_limits)
print(f"Best parameters: p(i) = {best_params[0]}, ∆ = {best_params[1]}")
print(f"Best mean G-measure: {best_g_measure}")
