from FuzzyData.src.fuzzy_functions import FuzzyMethods
import math
import numpy as np

from joblib import Parallel, delayed
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
import gc
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def g_measure(y_true, y_pred):
    return f1_score(y_true, y_pred, average='weighted')

class FuzzyGA:
    def __init__(self, fuzzy_system, parents_population_size=250, parents_selected_percentage=0.8,
                 mutation_probability=0.2, number_of_generations=100):
        self.fuzzy_system = fuzzy_system
        self.parents_population_size = parents_population_size
        self.parents_selected_percentage = parents_selected_percentage
        self.mutation_probability = mutation_probability
        self.number_of_generations = number_of_generations

        self.parents_to_select = math.ceil(self.parents_population_size * self.parents_selected_percentage)
        self.offspring_to_mutate = math.ceil(self.parents_to_select * self.mutation_probability)

        self.parents = np.random.uniform(low=-1, high=1, size=(self.parents_population_size, 2))
        self.parents_fitness_array = np.zeros(self.parents_population_size)
        self.offspring = np.zeros((self.parents_to_select, 2))
        self.offspring_fitness_array = np.zeros(self.parents_to_select)

        self.evaluate_initial_population()

    def evaluate_initial_population(self):
        self.parents_fitness_array = np.array(
            Parallel(n_jobs=-1)(delayed(self.evaluate)(parent) for parent in self.parents)
        )

    def main_loop(self):
        best_g_measure = 0
        best_params = None

        for main_loop_iterator in range(1, self.number_of_generations + 1):
            logger.info(f"Generation {main_loop_iterator}")
            selected_parents = self.parents_selection()
            self.offspring_creation(selected_parents)
            self.mutations_application()

            self.offspring_fitness_array = np.array(
                Parallel(n_jobs=-1)(delayed(self.evaluate)(offspring) for offspring in self.offspring)
            )

            parents_and_offspring = np.concatenate((self.parents, self.offspring))
            parents_and_offspring_fitness = np.concatenate((self.parents_fitness_array, self.offspring_fitness_array))
            sorted_indexes = parents_and_offspring_fitness.argsort()[::-1]
            parents_and_offspring_fitness = parents_and_offspring_fitness[sorted_indexes]
            parents_and_offspring = parents_and_offspring[sorted_indexes]

            current_g_measure = parents_and_offspring_fitness[0]
            if current_g_measure > best_g_measure:
                best_g_measure = current_g_measure
                best_params = parents_and_offspring[0]

            self.parents = parents_and_offspring[:self.parents_population_size]
            self.parents_fitness_array = parents_and_offspring_fitness[:self.parents_population_size]

            logger.info(f"Best G-measure after generation {main_loop_iterator}: {best_g_measure}")
            logger.info(f"Best parameters after generation {main_loop_iterator}: {best_params}")

            gc.collect()

        return best_g_measure, best_params

    def evaluate(self, params):
        pi, delta = params
        logger.info(f"Evaluating params: pi={pi}, delta={delta}")
        try:
            kf = KFold(n_splits=5)
            scores = []

            for train_index, test_index in kf.split(self.fuzzy_system.df):
                train_df = self.fuzzy_system.df.iloc[train_index]
                test_df = self.fuzzy_system.df.iloc[test_index]

                input_data_train = train_df[['Percentile', 'Apgar', 'Ph']]
                y_true_train = train_df[['Percentile', 'Apgar', 'Ph']].apply(
                    lambda row: self.classify(row['Percentile'], row['Apgar'], row['Ph']), axis=1).values

                input_data_test = test_df[['Percentile', 'Apgar', 'Ph']]
                y_true_test = test_df[['Percentile', 'Apgar', 'Ph']].apply(
                    lambda row: self.classify(row['Percentile'], row['Apgar'], row['Ph']), axis=1).values

                self.fuzzy_system.membership_fun_ph(pi)
                self.fuzzy_system.membership_fun_BW(pi)
                self.fuzzy_system.membership_fun_AP(pi)

                y_pred = self.fuzzy_system.fuzzy_interfence_system(input_data_test, pi=pi, delta=delta)

                score = g_measure(y_true_test, y_pred)
                scores.append(score)

            average_score = np.mean(scores)
            logger.info(f"Average G-measure: {average_score}")
            return average_score
        except Exception as e:
            logger.error(f"Error evaluating params {params}: {e}", exc_info=True)
            return 0

    def classify(self, percentile, apgar, ph):
        if apgar >= 7 and percentile >= 10 and ph >= 7.2:
            return 2  # Normal
        elif 5 <= apgar < 7 and 5 < percentile < 10 and 7.1 <= ph < 7.2:
            return 1  # Suspicious
        else:
            return 0  # Abnormal

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
            offspring_creation_array[i] = self.clamp(offspring1)
            offspring_creation_array[i + 1] = self.clamp(offspring2)
        self.offspring = offspring_creation_array

    def clamp(self, array):
        return np.clip(array, -1, 1)

    def mutations_application(self):
        selected_indexes = np.random.choice(np.arange(self.offspring.shape[0]), size=self.offspring_to_mutate, replace=False)
        for index in selected_indexes:
            mutation_point = np.random.randint(0, 2)
            self.offspring[index, mutation_point] = np.random.uniform(-1, 1)
            self.offspring[index] = self.clamp(self.offspring[index])

# Example usage
file_name = "FDA_data.xls"
fuzzy_system = FuzzyMethods(file_name)

ga = FuzzyGA(fuzzy_system)
best_g_measure, best_params = ga.main_loop()

print(f"Best G-measure found: {best_g_measure}")
print(f"Best parameters (pi, delta): {best_params}")

# Now, determine the number of signals with high informativeness level |y0| > 0.5
fuzzy_system.membership_fun_ph(best_params[0])
fuzzy_system.membership_fun_BW(best_params[0])
fuzzy_system.membership_fun_AP(best_params[0])

input_data = fuzzy_system.df[['Percentile', 'Apgar', 'Ph']]
y_pred = fuzzy_system.fuzzy_interfence_system(input_data, pi=best_params[0], delta=best_params[1])
high_informativeness_signals = np.sum(np.abs(y_pred) > 0.5)

print(f"Number of signals with high informativeness level: {high_informativeness_signals}")
