import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define the universe of discourse for each variable
percentile = ctrl.Antecedent(np.arange(0, 101, 1), 'percentile')
apgar = ctrl.Antecedent(np.arange(0, 11, 1), 'apgar')
ph = ctrl.Antecedent(np.arange(6.5, 8.0, 0.01), 'ph')
outcome = ctrl.Consequent(np.arange(0, 4, 1), 'outcome')

# Define fuzzy sets and membership functions for each input
percentile['low'] = fuzz.trapmf(percentile.universe, [0, 0, 20, 40])
percentile['medium'] = fuzz.trimf(percentile.universe, [30, 50, 70])
percentile['high'] = fuzz.trapmf(percentile.universe, [60, 80, 100, 100])

apgar['low'] = fuzz.trapmf(apgar.universe, [0, 0, 3, 5])
apgar['medium'] = fuzz.trimf(apgar.universe, [4, 6, 8])
apgar['high'] = fuzz.trapmf(apgar.universe, [7, 9, 10, 10])

ph['low'] = fuzz.trapmf(ph.universe, [6.5, 6.5, 7.0, 7.2])
ph['normal'] = fuzz.trimf(ph.universe, [7.1, 7.3, 7.4])
ph['high'] = fuzz.trapmf(ph.universe, [7.3, 7.5, 8.0, 8.0])

# Define the output fuzzy set
outcome['bad'] = fuzz.trapmf(outcome.universe, [0, 0, 1, 1.5])
outcome['average'] = fuzz.trapmf(outcome.universe, [1, 1.5, 2.5, 3])
outcome['good'] = fuzz.trapmf(outcome.universe, [2, 2.5, 3, 3])

# Define fuzzy rules
rule1 = ctrl.Rule(percentile['low'] & apgar['low'] & ph['low'], outcome['bad'])
rule2 = ctrl.Rule(percentile['medium'] & apgar['medium'] & ph['normal'], outcome['average'])
rule3 = ctrl.Rule(percentile['high'] & apgar['high'] & ph['high'], outcome['good'])
rule4 = ctrl.Rule(apgar['low'] | ph['low'], outcome['bad'])
rule5 = ctrl.Rule(apgar['high'] & ph['normal'], outcome['average'])

# Create control system and simulation
outcome_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5])
outcome_sim = ctrl.ControlSystemSimulation(outcome_ctrl)

# Example input
example_data = {
    'percentile': 3,
    'apgar': 5,
    'ph': 8
}

# Fuzzification
outcome_sim.input['percentile'] = example_data['percentile']
outcome_sim.input['apgar'] = example_data['apgar']
outcome_sim.input['ph'] = example_data['ph']

# Perform inference
outcome_sim.compute()

# Get the result
result = outcome_sim.output['outcome']
print(f"Fuzzy outcome: {result}")

# Output the crisp result
classification = 'bad' if result < 1 else 'average' if result < 2.5 else 'good'
print(f"Defuzzified result (classification): {classification}")
