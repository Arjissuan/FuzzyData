import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import pandas as pd
import os
# Generate universe variables
# * Quality and service on subjective ranges [0, 10]
# * Tip has a range of [0, 25] in units of percentage points
dataset = pd.read_excel(io=os.path.join(os.getcwd(), "FDA_data.xls"))
print(dataset.loc[dataset["Ph"] < 7.1, "Ph"] )
labels = ("perecentile", "apgar", "pH")
BW = ctrl.Antecedent(dataset.iloc[:, 0], "percentile")
AP = ctrl.Antecedent(dataset.iloc[:, 1], label="apgar")
ph = ctrl.Antecedent(dataset.iloc[:,2], label=labels[2])

#membership functions
BW.automf(3)
AP.automf(3)

#memberhisp functions for Ph
#since we have "labels" with which we
peha_norm = dataset.loc[dataset["Ph"] >= 7.2, "Ph"]
peha_abnm = dataset.loc[dataset['Ph'] < 7.1, "Ph"]
peha_sussy = dataset.loc[(dataset.loc[:, 'Ph'] >= 7.1) & (dataset.loc[:, 'Ph'] < 7.2), "Ph"]

print(len(peha_sussy)+ len(peha_norm)+ len(peha_abnm))

ph["normal"] = fuzz.gaussmf(ph.universe, np.mean(peha_norm), np.std(peha_norm))
ph["Suspicious"] = fuzz.gaussmf(ph.universe, np.mean(peha_sussy), np.std(peha_sussy))
ph["Abnormal"] = fuzz.gaussmf(ph.universe, np.mean(peha_abnm), np.std(peha_abnm))


ph.view()
plt.show()
