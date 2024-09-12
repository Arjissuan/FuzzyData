import simpful as sf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

# Load dataset
dataset = pd.read_excel(io=os.path.join(os.getcwd(), "../FDA_data.xls"))

# Define antecedent variables (input) for Percentile, Apgar, and Ph
bw_values = dataset["Percentile"].values
ap_values = dataset["Apgar"].values
ph_values = dataset["Ph"].values

# Create the fuzzy system using simpful
FS = sf.FuzzySystem()


# Sigmoid function (helper function)
def sigmoid_mf(x, c, a):
    """Returns a sigmoid membership function."""
    return 1 / (1 + np.exp(-a * (x - c)))


# Adjust parameters for sigmoids
def create_sigmoids_with_suspicious(universe, c_normal, c_abnormal, steepness_normal, steepness_abnormal):
    """
    Creates Normal, Suspicious, and Abnormal sigmoids such that Suspicious crosses Normal and Abnormal at 0.5 points.
    """
    normal_sigmoid = sigmoid_mf(universe, c_normal, steepness_normal)  # Rising sigmoid for Normal
    abnormal_sigmoid = 1 - sigmoid_mf(universe, c_abnormal, steepness_abnormal)  # Falling sigmoid for Abnormal

    # Suspicious: combine two sigmoids, one rising and one falling
    suspicious_sigmoid = np.minimum(1 - sigmoid_mf(universe, (c_normal + c_abnormal) / 2, steepness_normal),
                                    sigmoid_mf(universe, (c_normal + c_abnormal) / 2, steepness_abnormal))

    return normal_sigmoid, suspicious_sigmoid, abnormal_sigmoid


# Generate universe of discourse for each variable
bw_universe = np.linspace(min(bw_values), max(bw_values), 100)
ap_universe = np.linspace(min(ap_values), max(ap_values), 100)
ph_universe = np.linspace(min(ph_values), max(ph_values), 100)

# Create sigmoids for each variable ensuring proper crossing of suspicious at 0.5 points
BW_normal, BW_suspicious, BW_abnormal = create_sigmoids_with_suspicious(bw_universe, c_normal=10, c_abnormal=5,
                                                                        steepness_normal=1, steepness_abnormal=1)
AP_normal, AP_suspicious, AP_abnormal = create_sigmoids_with_suspicious(ap_universe, c_normal=7, c_abnormal=5,
                                                                        steepness_normal=1, steepness_abnormal=1)
PH_normal, PH_suspicious, PH_abnormal = create_sigmoids_with_suspicious(ph_universe, c_normal=7.2, c_abnormal=7.1,
                                                                        steepness_normal=10, steepness_abnormal=10)

# Add FuzzySet objects to simpful system
# Percentile (BW)
FS.add_linguistic_variable("Percentile", sf.LinguisticVariable([
    sf.FuzzySet(function=sf.Sigmoid_MF(c=10, a=1), term="Normal"),
    sf.FuzzySet(function=sf.Sigmoid_MF(c=7.5, a=1), term="Suspicious"),
    sf.FuzzySet(function=sf.Sigmoid_MF(c=5, a=-1), term="Abnormal")
], universe_of_discourse=[min(bw_values), max(bw_values)]))

# Apgar (AP)
FS.add_linguistic_variable("Apgar", sf.LinguisticVariable([
    sf.FuzzySet(function=sf.Sigmoid_MF(c=7, a=1), term="Normal"),
    sf.FuzzySet(function=sf.Sigmoid_MF(c=6, a=1), term="Suspicious"),
    sf.FuzzySet(function=sf.Sigmoid_MF(c=5, a=-1), term="Abnormal")
], universe_of_discourse=[min(ap_values), max(ap_values)]))

# Ph (PH) with proper crossing in the middle
FS.add_linguistic_variable("Ph", sf.LinguisticVariable([
    sf.FuzzySet(function=sf.Sigmoid_MF(c=7.2, a=10), term="Normal"),
    sf.FuzzySet(function=sf.Sigmoid_MF(c=7.15, a=10), term="Suspicious"),
    sf.FuzzySet(function=sf.Sigmoid_MF(c=7.1, a=-10), term="Abnormal")
], universe_of_discourse=[min(ph_values), max(ph_values)]))

# Define constant output values for TSK method
high_output = 1  # High output for Normal cases
medium_output = 0.5  # Medium output for Suspicious cases
low_output = 0  # Low output for Abnormal cases

# Add fuzzy rules using TSK-style consequents (numerical outputs)
FS.add_rules([
    f"IF (Percentile IS Normal) AND (Apgar IS Normal) AND (Ph IS Normal) THEN output = {high_output}",
    f"IF (Percentile IS Suspicious) OR (Apgar IS Suspicious) OR (Ph IS Suspicious) THEN output = {medium_output}",
    f"IF (Percentile IS Abnormal) OR (Apgar IS Abnormal) OR (Ph IS Abnormal) THEN output = {low_output}"
])

# Set input values using set_variable() for an example input
FS.set_variable("Percentile", bw_values[0])
FS.set_variable("Apgar", ap_values[0])
FS.set_variable("Ph", ph_values[0])

# Perform fuzzy inference using Sugeno_inference()
try:
    result = FS.Sugeno_inference()
    print(f"Inference result: {result['output']}")
except Exception as e:
    print(f"Error during inference: {e}")

# Plotting membership functions
# Membership functions for Percentile
plt.figure()
plt.plot(bw_universe, BW_normal, label="Normal")
plt.plot(bw_universe, BW_suspicious, label="Suspicious")
plt.plot(bw_universe, BW_abnormal, label="Abnormal")
plt.title("Percentile Membership Functions (Sigmoid)")
plt.legend()

# Membership functions for Apgar
plt.figure()
plt.plot(ap_universe, AP_normal, label="Normal")
plt.plot(ap_universe, AP_suspicious, label="Suspicious")
plt.plot(ap_universe, AP_abnormal, label="Abnormal")
plt.title("Apgar Membership Functions (Sigmoid)")
plt.legend()

# Membership functions for Ph
plt.figure()
plt.plot(ph_universe, PH_normal, label="Normal")
plt.plot(ph_universe, PH_suspicious, label="Suspicious")
plt.plot(ph_universe, PH_abnormal, label="Abnormal")
plt.title("Ph Membership Functions (Sigmoid)")
plt.legend()

plt.show()
