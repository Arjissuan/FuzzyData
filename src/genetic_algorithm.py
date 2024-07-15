import random
import math
import copy
import numpy as np
from sklearn.model_selection import KFold

def g_measure(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    precision = tp / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    if precision + recall == 0:
        return 0
    g_measure = (2 * precision * recall) / (precision + recall)
    return g_measure

class FuzzyGA:
    def __init__(self, data, parents_population_size=250, parents_selected_percentage=0.8,
                 mutation_probability=0.2, number_of_generations=1000, visualise=True):
        self.data = data
        self.parents_population_size = parents_population_size
        self.parents_selected_percentage = parents_selected_percentage
        self.mutation_probability = mutation_probability
        self.number_of_generations = number_of_generations
        self.visualise = visualise

        self.parents_to_select = math.ceil(self.parents_population_size * self.parents_selected_percentage)
        self.offspring_to_mutate = math.ceil(self.parents_to_select * self.mutation_probability)

        self.parents = np.zeros((self.parents_population_size, len(data[0])))  # Initializing for parameters
        self.parents_fitness_array = np.zeros(self.parents_population_size)

        self.offspring = np.zeros((self.parents_to_select, len(data[0])))
        self.offspring_fitness_array = np.zeros(self.parents_to_select)

        self.parents_creation()
        self.evaluate_initial_population()

    def evaluate_initial_population(self):
        for i, parent in enumerate(self.parents):
            self.parents_fitness_array[i] = self.evaluate(parent)

    def main_loop(self):
        best_g_measure = 0
        best_params = None

        for main_loop_iterator in range(1, self.number_of_generations + 1):
            selected_parents = self.parents_selection()
            self.offspring_creation(selected_parents)
            self.mutations_application()
            for i, offspring in enumerate(self.offspring):
                self.offspring_fitness_array[i] = self.evaluate(offspring)

            parents_and_offspring = np.concatenate((self.parents, self.offspring))
            parents_and_offspring_fitness = np.concatenate(
                (self.parents_fitness_array, self.offspring_fitness_array))
            sorted_indexes = parents_and_offspring_fitness.argsort()[::-1]
            parents_and_offspring_fitness = parents_and_offspring_fitness[sorted_indexes]
            parents_and_offspring = parents_and_offspring[sorted_indexes]

            current_g_measure = parents_and_offspring_fitness[0]
            if current_g_measure > best_g_measure:
                best_g_measure = current_g_measure
                best_params = parents_and_offspring[0]

            self.parents = parents_and_offspring[:self.parents_population_size]
            self.parents_fitness_array = parents_and_offspring_fitness[:self.parents_population_size]

        return best_g_measure, best_params

    def parents_creation(self):
        self.parents = np.random.uniform(low=0, high=1, size=(self.parents_population_size, len(self.data[0])))

    def evaluate(self, params):
        y_true = self.data[:, 0]
        y_pred = self.fuzzy_system(params)
        return g_measure(y_true, y_pred)

    def fuzzy_system(self, params):
        # Placeholder for fuzzy system, generate random binary array as predictions
        y_pred = np.random.randint(0, 2, len(self.data))
        return y_pred

    def parents_selection(self):
        selected_indexes = np.random.choice(np.arange(self.parents.shape[0]), size=self.parents_to_select, replace=False)
        return self.parents[selected_indexes]

    def offspring_creation(self, selected_parents):
        offspring_creation_array = np.zeros((self.parents_to_select, len(self.data[0])))
        for i in range(0, self.parents_to_select, 2):
            parent1, parent2 = selected_parents[i], selected_parents[i + 1]
            crossover_point = np.random.randint(1, len(self.data[0]) - 1)
            offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            offspring_creation_array[i] = offspring1
            offspring_creation_array[i + 1] = offspring2
        self.offspring = offspring_creation_array

    def mutations_application(self):
        selected_indexes = np.random.choice(np.arange(self.offspring.shape[0]), size=self.offspring_to_mutate, replace=False)
        for index in selected_indexes:
            mutation_point = np.random.randint(0, len(self.data[0]))
            self.offspring[index, mutation_point] = np.random.uniform(0, 1)

def cross_validate(data, params, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)
    g_measures = []

    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        ga = FuzzyGA(train_data, *params)
        g_measure, best_params = ga.main_loop()
        g_measures.append(g_measure)

    mean_g_measure = np.mean(g_measures)
    best_fold = np.argmax(g_measures)
    best_fold_params = params[best_fold]

    return mean_g_measure, best_fold_params