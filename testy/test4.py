import math
import numpy as np
from FuzzyData.src.fuzzy_functions import FuzzyMethods


def g_measure(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    precision = tp / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
    recall = tp / np.sum(y_true == 1) if np.sum(y_true == 1) > 0 else 0
    if precision + recall == 0:
        return 0
    g_measure = (2 * precision * recall) / (precision + recall)
    return g_measure


class FuzzyGA:
    def __init__(self, fuzzy_system, parents_population_size=250, parents_selected_percentage=0.8,
                 mutation_probability=0.2, number_of_generations=1000):
        self.fuzzy_system = fuzzy_system
        self.parents_population_size = parents_population_size
        self.parents_selected_percentage = parents_selected_percentage
        self.mutation_probability = mutation_probability
        self.number_of_generations = number_of_generations

        self.parents_to_select = math.ceil(self.parents_population_size * self.parents_selected_percentage)
        self.offspring_to_mutate = math.ceil(self.parents_to_select * self.mutation_probability)

        # Initialize parent parameters with random values: [pi, delta]
        self.parents = np.random.uniform(low=-1, high=1, size=(self.parents_population_size, 2))
        self.parents_fitness_array = np.zeros(self.parents_population_size)

        self.offspring = np.zeros((self.parents_to_select, 2))
        self.offspring_fitness_array = np.zeros(self.parents_to_select)

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

    def evaluate(self, params):
        pi, delta = params
        self.fuzzy_system.membership_fun_ph(pi)
        self.fuzzy_system.membership_fun_AP(pi)
        self.fuzzy_system.membership_fun_BW(pi)

        input_data = self.fuzzy_system.df[['Percentile', 'Apgar', 'Ph']]  # Assuming input data is stored in FuzzyMethods instance
        y_true = self.fuzzy_system.df[['Ph']].values  # Assuming output data is stored in FuzzyMethods instance
        y_pred = self.fuzzy_system.fuzzy_interfence_system(input_data, pi=pi, delta=delta)
        return g_measure(y_true, y_pred)



    def parents_selection(self):
        selected_indexes = np.random.choice(np.arange(self.parents.shape[0]), size=self.parents_to_select, replace=False)
        return self.parents[selected_indexes]

    def offspring_creation(self, selected_parents):
        offspring_creation_array = np.zeros((self.parents_to_select, 2))
        for i in range(0, self.parents_to_select, 2):
            parent1, parent2 = selected_parents[i], selected_parents[i + 1]
            crossover_point = np.random.randint(1, 2)
            offspring1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
            offspring2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
            offspring_creation_array[i] = offspring1
            offspring_creation_array[i + 1] = offspring2
        self.offspring = offspring_creation_array

    def mutations_application(self):
        selected_indexes = np.random.choice(np.arange(self.offspring.shape[0]), size=self.offspring_to_mutate, replace=False)
        for index in selected_indexes:
            mutation_point = np.random.randint(0, 2)
            self.offspring[index, mutation_point] = np.random.uniform(-1, 1)


# Example usage
file_name = "FDA_data.xls"
fuzzy_system = FuzzyMethods(file_name)

ga = FuzzyGA(fuzzy_system)
best_g_measure, best_params = ga.main_loop()

print(f"Best G-measure found: {best_g_measure}")
print(f"Best parameters (pi, delta): {best_params}")

