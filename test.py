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
print(dataset.loc[dataset["Ph"] < 7.1, "Ph"])

labels = ("Percentile", "Apgar", "Ph")
bw_values = dataset["Percentile"].values
ap_values = dataset["Apgar"].values
ph_values = dataset["Ph"].values

BW = ctrl.Antecedent(np.linspace(min(bw_values), max(bw_values), 100), 'Percentile')
AP = ctrl.Antecedent(np.linspace(min(ap_values), max(ap_values), 100), 'Apggar')
PH = ctrl.Antecedent(np.linspace(min(ph_values), max(ph_values), 100), 'Ph')

# membership functions for bodyweight
BW_normal = dataset.loc[dataset["Percentile"] >= 10, "Percentile"]
BW_sussy = dataset.loc[(dataset['Percentile'] < 10) & (dataset['Percentile'] > 5), 'Percentile']
BW_abnormal = dataset.loc[dataset['Percentile'] <= 5, 'Percentile']

# membership function for apgar
AP_normal = dataset.loc[dataset["Apgar"] >= 7, "Apgar"]
AP_sussy = dataset.loc[(dataset['Apgar'] < 7) & (dataset['Apgar'] >= 5), 'Apgar']
AP_abnormal = dataset.loc[dataset['Apgar'] < 5, 'Apgar']

# membership functions for Ph
# since we have "labels" with which we
PH_normal = dataset.loc[dataset["Ph"] >= 7.2, "Ph"]
PH_sussy = dataset.loc[(dataset.loc[:, 'Ph'] >= 7.1) & (dataset.loc[:, 'Ph'] < 7.2), "Ph"]
PH_abnormal = dataset.loc[dataset['Ph'] < 7.1, "Ph"]

# Ensure non-empty subsets
mean_BW_normal = float(np.mean(BW_normal)) if not BW_normal.empty else 0
std_BW_normal = float(np.std(BW_normal)) if not BW_normal.empty else 1
mean_BW_sussy = float(np.mean(BW_sussy)) if not BW_sussy.empty else 0
std_BW_sussy = float(np.std(BW_sussy)) if not BW_sussy.empty else 1
mean_BW_abnormal = float(np.mean(BW_abnormal)) if not BW_abnormal.empty else 0
std_BW_abnormal = float(np.std(BW_abnormal)) if not BW_abnormal.empty else 1

mean_AP_normal = float(np.mean(AP_normal)) if not AP_normal.empty else 0
std_AP_normal = float(np.std(AP_normal)) if not AP_normal.empty else 1
mean_AP_sussy = float(np.mean(AP_sussy)) if not AP_sussy.empty else 0
std_AP_sussy = float(np.std(AP_sussy)) if not AP_sussy.empty else 1
mean_AP_abnormal = float(np.mean(AP_abnormal)) if not AP_abnormal.empty else 0
std_AP_abnormal = float(np.std(AP_abnormal)) if not AP_abnormal.empty else 1

mean_PH_normal = float(np.mean(PH_normal)) if not PH_normal.empty else 0
std_PH_normal = float(np.std(PH_normal)) if not PH_normal.empty else 1
mean_PH_sussy = float(np.mean(PH_sussy)) if not PH_sussy.empty else 0
std_PH_sussy = float(np.std(PH_sussy)) if not PH_sussy.empty else 1
mean_PH_abnormal = float(np.mean(PH_abnormal)) if not PH_abnormal.empty else 0
std_PH_abnormal = float(np.std(PH_abnormal)) if not PH_abnormal.empty else 1


BW["Normal"] = fuzz.gaussmf(BW.universe, mean_BW_normal, std_BW_normal)
BW["Suspicious"] = fuzz.gaussmf(BW.universe, mean_BW_sussy, std_BW_sussy)
BW["Abnormal"] = fuzz.gaussmf(BW.universe, mean_BW_abnormal, std_BW_abnormal)

AP["Normal"] = fuzz.gaussmf(AP.universe, mean_AP_normal, std_AP_normal)
AP["Suspicious"] = fuzz.gaussmf(AP.universe, mean_AP_sussy, std_AP_sussy)
AP["Abnormal"] = fuzz.gaussmf(AP.universe, mean_AP_abnormal, std_AP_abnormal)

PH["Normal"] = fuzz.gaussmf(PH.universe, mean_PH_normal, std_PH_normal)
PH["Suspicious"] = fuzz.gaussmf(PH.universe, mean_PH_sussy, std_PH_sussy)
PH["Abnormal"] = fuzz.gaussmf(PH.universe, mean_PH_abnormal, std_PH_abnormal)


BW.view()
AP.view()
PH.view()
plt.show()
