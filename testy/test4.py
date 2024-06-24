import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score
import random
from FuzzyData.src.fuzzy_functions import FuzzyMethods


# Assuming FuzzyMethods and other imports are already defined
def evaluate_fuzzy_system(fuzzy_system, input_data, pH_limits):
    fuzzy_score = fuzzy_system.train(input_data)
    binary_fuzzy_score = 1 if fuzzy_score < pH_limits['lower'] or fuzzy_score > pH_limits['upper'] else 0
    actual_outcome = 1 if input_data['Ph'] < pH_limits['lower'] or input_data['Ph'] > pH_limits['upper'] else 0
    return binary_fuzzy_score, actual_outcome


def fitness_function(fuzzy_system, data, pH_limits, pi, delta):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    g_measures = []

    for train_index, test_index in kf.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]
        fuzzy_system.train(train_data, pi, delta)

        binary_predictions = []
        actual_outcomes = []

        for idx, row in test_data.iterrows():
            input_data = {'Percentile': row['Percentile'], 'Apgar': row['Apgar'], 'Ph': row['Ph']}
            binary_fuzzy_score, actual_outcome = evaluate_fuzzy_system(fuzzy_system, input_data, pH_limits)
            binary_predictions.append(binary_fuzzy_score)
            actual_outcomes.append(actual_outcome)

        precision = precision_score(actual_outcomes, binary_predictions)
        recall = recall_score(actual_outcomes, binary_predictions)
        g_measure = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        g_measures.append(g_measure)

    mean_g_measure = np.mean(g_measures)
    return mean_g_measure


def initialize_population(size, pi_range, delta_range):
    return [(random.uniform(*pi_range), random.uniform(*delta_range)) for _ in range(size)]


def selection(population, fitnesses, num_parents):
    selected_indices = np.argsort(fitnesses)[-num_parents:]
    return [population[i] for i in selected_indices]


def crossover(parents, offspring_size):
    offspring = []
    for _ in range(offspring_size):
        parent1 = random.choice(parents)
        parent2 = random.choice(parents)
        child = (np.mean([parent1[0], parent2[0]]), np.mean([parent1[1], parent2[1]]))
        offspring.append(child)
    return offspring


def mutation(offspring, pi_range, delta_range, mutation_rate=0.1):
    mutated_offspring = []
    for (pi, delta) in offspring:
        if random.random() < mutation_rate:
            pi += random.uniform(-0.1, 0.1)
            delta += random.uniform(-0.1, 0.1)
        pi = np.clip(pi, pi_range[0], pi_range[1])
        delta = np.clip(delta, delta_range[0], delta_range[1])
        mutated_offspring.append((pi, delta))
    return mutated_offspring


def evolutionary_strategy(fuzzy_system, data, pH_limits, pi_range, delta_range, generations=50, population_size=20, num_parents=10, mutation_rate=0.1):
    population = initialize_population(population_size, pi_range, delta_range)
    best_params = None
    best_fitness = 0

    for generation in range(generations):
        fitnesses = [fitness_function(fuzzy_system, data, pH_limits, pi, delta) for (pi, delta) in population]
        best_gen_fitness = max(fitnesses)
        best_gen_params = population[np.argmax(fitnesses)]

        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_params = best_gen_params

        parents = selection(population, fitnesses, num_parents)
        offspring = crossover(parents, population_size - num_parents)
        offspring = mutation(offspring, pi_range, delta_range, mutation_rate)

        population = parents + offspring

        print(f"Generation {generation}: Best Fitness = {best_fitness}, Best Params = {best_params}")

    return best_params, best_fitness


if __name__ == "__main__":
    fzm = FuzzyMethods()
    fzm.membership_fun_ph()
    fzm.membership_fun_AP()
    fzm.membership_fun_BW()
    fzm.make_plots()

    pi_range = (-0.50, 0.50)
    delta_range = (-0.50, 0.50)
    pH_limits = {'lower': 7.1, 'upper': 7.2}
    data = fzm.df

    best_params, best_fitness = evolutionary_strategy(fzm, data, pH_limits, pi_range, delta_range)
    print(f"Best parameters: p(i) = {best_params[0]}, ∆ = {best_params[1]}")
    print(f"Best mean G-measure: {best_fitness}")
