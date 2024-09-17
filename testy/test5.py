#zadanie 1
import numpy as np
import pandas as pd
import glob
import os


file_path = glob.glob(f'{os.getcwd()}/FDA_data*')[0]
try:
    data = pd.read_csv(file_path)
except:
    data = pd.read_excel(file_path)


def sigmoid(x, a, b):
    return 1 / (1 + np.exp(-a * (x - b)))

#membership functions
def apgar_membership(apgar):
    normal = sigmoid(apgar, 10, 7)  # Steep slope around 7
    suspicious = sigmoid(apgar, 10, 5.5) - sigmoid(apgar, 10, 6.5)  # Suspicious range between 5 and 6
    abnormal = sigmoid(apgar, -10, 5)  # Steep slope around 5 (inverse for abnormal)
    return normal, suspicious, abnormal

def percentile_membership(percentile):
    normal = sigmoid(percentile, 10, 10)
    suspicious = sigmoid(percentile, 10, 5) - sigmoid(percentile, 10, 10)
    abnormal = sigmoid(percentile, -10, 5)
    return normal, suspicious, abnormal

def ph_membership(ph):
    normal = sigmoid(ph, 100, 7.2)
    suspicious = sigmoid(ph, 100, 7.1) - sigmoid(ph, 100, 7.2)
    abnormal = sigmoid(ph, -100, 7.1)
    return normal, suspicious, abnormal

print(ph_membership(data['Ph']))

memberships = []
for indx, row in data.iterrows():
    # print(indx, row)
    apgar_mem = apgar_membership(row['Apgar'])
    percentile_mem = percentile_membership(row['Percentile'])
    ph_mem = ph_membership(row['Ph'])

    memberships.append({
        'Apgar_Normal': apgar_mem[0],
        'Apgar_Suspicious': apgar_mem[1],
        'Apgar_Abnormal': apgar_mem[2],
        'Percentile_Normal': percentile_mem[0],
        'Percentile_Suspicious': percentile_mem[1],
        'Percentile_Abnormal': percentile_mem[2],
        'Ph_Normal': ph_mem[0],
        'Ph_Suspicious': ph_mem[1],
        'Ph_Abnormal': ph_mem[2]
    })


memberships_df = pd.DataFrame(memberships)
print(memberships_df)


def rules(ap, perc, ph):
    rules = {2:[], # normal
             1:[], # suspicious
             0:[], # abnormal
             }
