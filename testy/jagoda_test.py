import numpy as np
import simpful as sf
import pandas as pd

def g_measure(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred >= 0.5))  # True positives
    precision = tp / np.sum(y_pred >= 0.5) if np.sum(y_pred >= 0.5) > 0 else 0
    recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0

    if precision + recall == 0:
        return 0

    g_measure = (2 * precision * recall) / (precision + recall)
    return g_measure


# Define the fuzzy inference system with adjustable parameters
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
def evolution_strategy(population_size, generations, mutation_rate, data):
    num_params = 4  # 3 p(i) values and 1 Delta
    population = np.random.uniform(-0.5, 0.5, (population_size, num_params))  # Initialize population
    print(population)
    best_solution = None
    best_fitness = 0

    for gen in range(generations):
        print(gen)
        # Step 1: Evaluate fitness of the entire population in a batch
        fitness_scores = fitness_function(population, data)

        # Step 2: Selection (select top performers)
        top_indices = fitness_scores.argsort()[-(population_size//2):]
        selected_population = population[top_indices]

        # Step 3: Mutation (apply random changes to selected population)
        mutated_population = selected_population + np.random.normal(0, mutation_rate, selected_population.shape)

        # Step 4: Create new population (combine selected and mutated populations)
        population = np.vstack([selected_population, mutated_population])

        # Track the best solution
        current_best_fitness = np.max(fitness_scores)
        if current_best_fitness > best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[np.argmax(fitness_scores)]

        print(f"Generation {gen+1}, Best Fitness: {current_best_fitness}")

    return best_solution, best_fitness


# Example dataset
dt = pd.read_csv("../FDA_data.csv")
data = pd.DataFrame({
    'Percentile': dt['Percentile'],
    'Apgar': dt['Apgar'],
    'Ph': dt['Ph'],
    'Outcome': np.random.randint(0, 2, len(dt["Ph"]))  # Assuming binary outcomes
})

# Run Evolution Strategy
best_solution, best_fitness = evolution_strategy(population_size=50, generations=100, mutation_rate=0.1, data=data)
print("Best solution:", best_solution)
print("Best fitness (G-measure):", best_fitness)
