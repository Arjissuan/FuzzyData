import numpy as np
import simpful as sf
import pandas as pd
import time


def g_measure(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred >= 0.5))  # True positives
    precision = tp / np.sum(y_pred >= 0.5) if np.sum(y_pred >= 0.5) > 0 else 0
    recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0

    if precision + recall == 0:
        return 0

    g_measure = (2 * precision * recall) / (precision + recall)
    return g_measure


# Fuzzy inference system with adjustable parameters
def fuzzy_inference_system(data, p_values, delta):
    # Create a new fuzzy system
    FS = sf.FuzzySystem()

    # Extract p(i) values to adjust the suspicious membership functions
    p_apgar, p_percentile, p_ph = p_values

    # Define universes of discourse based on data ranges
    bw_universe = np.linspace(min(data["Percentile"]) - 2, max(data["Percentile"]) + 2, 100)
    ap_universe = np.linspace(min(data["Apgar"]) - 2, max(data["Apgar"]) + 2, 100)
    ph_universe = np.linspace(min(data["Ph"]) - 0.1, max(data["Ph"]) + 0.1, 100)

    # Create crossing suspicious membership functions with adjusted p(i) values
    BW_normal, BW_suspicious, BW_abnormal = create_crossing_suspicious(bw_universe, 10, 5, 2, 2)
    AP_normal, AP_suspicious, AP_abnormal = create_crossing_suspicious(ap_universe, 7 + p_apgar, 5, 2, 2)
    PH_normal, PH_suspicious, PH_abnormal = create_crossing_suspicious(ph_universe, 7.2 + p_ph, 7.1, 20, 20)

    # Add linguistic variables to the fuzzy system
    FS.add_linguistic_variable("Percentile", sf.LinguisticVariable([
        sf.FuzzySet(function=sf.Sigmoid_MF(c=10, a=2), term="Normal"),
        sf.FuzzySet(function=lambda x: np.interp(x, bw_universe, BW_suspicious), term="Suspicious"),
        sf.FuzzySet(function=sf.Sigmoid_MF(c=5, a=-2), term="Abnormal")
    ], universe_of_discourse=[min(data["Percentile"]), max(data["Percentile"])]))

    FS.add_linguistic_variable("Apgar", sf.LinguisticVariable([
        sf.FuzzySet(function=sf.Sigmoid_MF(c=7, a=2), term="Normal"),
        sf.FuzzySet(function=lambda x: np.interp(x, ap_universe, AP_suspicious), term="Suspicious"),
        sf.FuzzySet(function=sf.Sigmoid_MF(c=5, a=-2), term="Abnormal")
    ], universe_of_discourse=[min(data["Apgar"]), max(data["Apgar"])]))

    FS.add_linguistic_variable("Ph", sf.LinguisticVariable([
        sf.FuzzySet(function=sf.Sigmoid_MF(c=7.2, a=20), term="Normal"),
        sf.FuzzySet(function=lambda x: np.interp(x, ph_universe, PH_suspicious), term="Suspicious"),
        sf.FuzzySet(function=sf.Sigmoid_MF(c=7.1, a=-20), term="Abnormal")
    ], universe_of_discourse=[min(data["Ph"]), max(data["Ph"])]))

    # Add output crisp values
    FS.set_crisp_output_value("Abnormal_output", 1.0)
    FS.set_crisp_output_value("Suspicious_output", 0.5)
    FS.set_crisp_output_value("Normal_output", 0.0)

    # Add fuzzy rules
    FS.add_rules([
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
    ])

    # Perform fuzzy inference for each input in the dataset
    y_pred = []
    for i in range(len(data)):
        FS.set_variable("Percentile", data["Percentile"].iloc[i])
        FS.set_variable("Apgar", data["Apgar"].iloc[i])
        FS.set_variable("Ph", data["Ph"].iloc[i])

        result = FS.Sugeno_inference()
        y_pred.append(result['output'])

    y_pred = np.array(y_pred)

    # Apply threshold Delta to classify the outcome
    y_pred_class = (y_pred >= delta).astype(int)
    return y_pred_class


# Helper function for crossing suspicious sigmoids
def create_crossing_suspicious(universe, c_normal, c_abnormal, steepness_normal, steepness_abnormal):
    normal_sigmoid = sigmoid_mf(universe, c_normal, steepness_normal)
    abnormal_sigmoid = 1 - sigmoid_mf(universe, c_abnormal, steepness_abnormal)
    midpoint = (c_normal + c_abnormal) / 2
    suspicious_sigmoid = sigmoid_mf(universe, midpoint, -steepness_abnormal) * sigmoid_mf(universe, midpoint,
                                                                                          steepness_normal)
    suspicious_sigmoid /= np.max(suspicious_sigmoid)  # Normalize
    return normal_sigmoid, suspicious_sigmoid, abnormal_sigmoid


# Sigmoid function for membership function definition
def sigmoid_mf(x, c, a):
    return 1 / (1 + np.exp(-a * (x - c)))

# Define the fitness function to evaluate the performance of each individual
def fitness_function(population, data):
    # Initialize an array to store fitness scores
    fitness_scores = np.zeros(len(population))

    # Loop through each individual in the population and calculate fitness
    for i, params in enumerate(population):
        p_values = params[:-1]  # Extract p(i) values
        delta = params[-1]  # Extract Delta

        # Call the fuzzy inference system for the entire batch of data
        y_pred_class = fuzzy_inference_system(data, p_values, delta)

        # Compute G-measure for these predictions
        y_true = data["Outcome"]
        fitness_scores[i] = g_measure(y_true, y_pred_class)
    return fitness_scores

# Evolution Strategy (ES) Algorithm
class EvolutionStategiesHelp:
    def __init__(self, mu, i, o):
        self.mu = mu  # Population size
        self.i = i  # Input data (Percentile, Apgar, Ph, etc.)
        self.o = o  # Output data (ground truth for classification)

    # Modify the objective function to use fuzzy inference and G-measure
    def objective_function(self, x):
        # Each row in x represents a set of parameters (p(i), Delta)
        fitness_scores = np.zeros(x.shape[0])
        for idx, params in enumerate(x):
            p_values = params[:-1]
            delta = params[-1]
            # Call fuzzy inference system and calculate G-measure (fitness score)
            y_pred_class = fuzzy_inference_system(self.i, p_values, delta)  # Use input data (i)
            fitness_scores[idx] = g_measure(self.o, y_pred_class)  # Compare predictions with ground truth (o)
        return fitness_scores

    # Generate the population and standard deviations (sigma)
    def generate_population(self):
        # Generate random population of size (mu, 4) -> 3 p(i) values + 1 Delta
        x = np.random.uniform(low=-0.5, high=0.5, size=(self.mu, 4))
        sigma = np.random.uniform(low=0, high=0.5, size=(self.mu, 4))
        return x, sigma

    # Mutation operator (kept the same)
    def mutation_operator(self, sigma, mod_sigma1):
        n = sigma.shape[1]
        tau2 = 1 / np.sqrt(2 * np.sqrt(n))
        mod_sigma2 = tau2 * np.random.randn(sigma.shape[0], n)
        sigma = sigma * np.exp(mod_sigma1) * np.exp(mod_sigma2)
        return sigma


class muPlusLambda:
    def __init__(self, mu, i, o, n_gen):
        self.num_generations = n_gen
        self.ES = EvolutionStategiesHelp(mu, i, o)

    def stop(self, par, off):
        return (abs(par.loc[0, 0] - off.loc[0, 0])) < 10**-5

    def strategy(self):
        start_time = time.time()

        x, sigma = self.ES.generate_population()  # Generate population
        min_fitness = np.inf
        best_gen = -1

        n = sigma.shape[1]
        tau1 = 1 / np.sqrt(2 * n)
        r_sigma1 = tau1 * np.random.randn(sigma.shape[0], n)

        fitness_scores = self.ES.objective_function(x)  # Initial fitness evaluation

        for generation in range(self.num_generations):
            offspring_x = []
            offspring_sigma = []

            for parent_idx in range(self.ES.mu):
                parent_params = x[parent_idx]
                parent_sigma = sigma[parent_idx]

                for _ in range(5):  # Generate 5 offspring per parent
                    mutated_sigma = self.ES.mutation_operator(parent_sigma, r_sigma1)
                    mutated_params = parent_params + mutated_sigma * np.random.randn(4)
                    offspring_x.append(mutated_params)
                    offspring_sigma.append(mutated_sigma)

            offspring_x = np.vstack(offspring_x)
            offspring_sigma = np.vstack(offspring_sigma)

            # Evaluate fitness for offspring
            offspring_fitness = self.ES.objective_function(offspring_x)

            # Combine parents and offspring for selection
            combined_x = np.vstack([x, offspring_x])
            combined_sigma = np.vstack([sigma, offspring_sigma])
            combined_fitness = np.concatenate([fitness_scores, offspring_fitness])

            # Sort by fitness and select the best individuals
            sorted_indices = np.argsort(combined_fitness)
            selected_indices = sorted_indices[:self.ES.mu]  # Select top mu

            x = combined_x[selected_indices]
            sigma = combined_sigma[selected_indices]
            fitness_scores = combined_fitness[selected_indices]

            # Track best solution
            best_idx = np.argmin(fitness_scores)
            if np.min(fitness_scores) < min_fitness:
                min_fitness = np.min(fitness_scores)
                best_gen = generation
                best_params = x[best_idx]

            # Stopping criteria
            parents_temp = pd.DataFrame(x)
            offspring = pd.DataFrame(offspring_x)
            if self.stop(parents_temp, offspring):
                break

        end_time = time.time()
        print(f"Best parameters: {best_params}")
        print(f"Best generation: {best_gen}")
        print(f"Best fitness (G-measure): {min_fitness}")
        print(f"Execution time: {end_time - start_time:.2f} sec")

        results = {
            'Best Parameters': [best_params],
            'Generation Number': [best_gen],
            'Best Fitness': [min_fitness],
            'Duration': [end_time - start_time]
        }
        df = pd.DataFrame(results, columns=['Best Parameters', 'Generation Number', 'Best Fitness', 'Duration'])
        return df


# Example dataset
dt = pd.read_csv("../FDA_data.csv")
data = pd.DataFrame({
    'Percentile': dt['Percentile'],
    'Apgar': dt['Apgar'],
    'Ph': dt['Ph'],
    'Outcome': np.random.randint(0, 2, len(dt["Ph"]))  # Assuming binary outcomes
})

# Run Evolution Strategy
# best_solution, best_fitness = evolution_strategy(population_size=50, generations=100, mutation_rate=0.1, data=data)
# print("Best solution:", best_solution)
# print("Best fitness (G-measure):", best_fitness)
