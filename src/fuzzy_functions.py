import skfuzzy as fuzz
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl


class FuzzyMethods:
    def __init__(self, file_name="FDA_data.xls"):
        self.df = pd.read_excel(io=os.path.join(os.getcwd(), file_name))
        self.BW = ctrl.Antecedent(np.linspace(np.min(self.df.iloc[:, 0]), np.max(self.df.iloc[:, 0]), 1000), label="Percentile")
        self.AP = ctrl.Antecedent(np.linspace(np.min(self.df.iloc[:, 1]), np.max(self.df.iloc[:, 1]), 1000), label="Apgar")
        self.ph = ctrl.Antecedent(np.linspace(np.min(self.df.iloc[:, 2]), np.max(self.df.iloc[:, 2]), 1000), label="Ph")
        self.outcome = ctrl.Consequent(np.arange(0, 3.01, 0.01), 'outcome')
        self.labels = ("Normal", "Suspicious", "Abnormal")

    def membership_fun_ph(self, pi=0):
        peha_norm = self.df.loc[self.df["Ph"] >= 7.2, "Ph"]
        peha_abnm = self.df.loc[self.df['Ph'] < 7.1, "Ph"]
        peha_sussy = self.df.loc[(self.df.loc[:, 'Ph'] >= 7.1) & (self.df.loc[:, 'Ph'] <= 7.2), "Ph"]

        mean_PH_normal = float(np.mean(peha_norm)) if not peha_norm.empty else 0
        std_PH_normal = float(np.std(peha_norm)) if not peha_norm.empty else 1
        mean_PH_sussy = float(np.mean(peha_sussy)) if not peha_sussy.empty else 0
        std_PH_sussy = float(np.std(peha_sussy)) if not peha_sussy.empty else 1
        mean_PH_abnormal = float(np.mean(peha_abnm)) if not peha_abnm.empty else 0
        std_PH_abnormal = float(np.std(peha_abnm)) if not peha_abnm.empty else 1

        self.ph[self.labels[0]] = fuzz.gaussmf(self.ph.universe, mean_PH_normal + pi,  std_PH_normal)
        self.ph[self.labels[1]] = fuzz.gaussmf(self.ph.universe, mean_PH_sussy + pi, std_PH_sussy)
        self.ph[self.labels[2]] = fuzz.gaussmf(self.ph.universe, mean_PH_abnormal + pi, std_PH_abnormal)

    def membership_fun_BW(self, pi=0):
        bewu_norm = self.df.loc[self.df["Percentile"] >= 10, "Percentile"]
        bewu_abnm = self.df.loc[self.df['Percentile'] <= 5, 'Percentile']
        bewu_sussy = self.df.loc[(self.df['Percentile'] < 10) & (self.df['Percentile'] > 5), 'Percentile']

        mean_BW_normal = float(np.mean(bewu_norm)) if not bewu_norm.empty else 0
        std_BW_normal = float(np.std(bewu_norm)) if not bewu_norm.empty else 1
        mean_BW_sussy = float(np.mean(bewu_sussy)) if not bewu_sussy.empty else 0
        std_BW_sussy = float(np.std(bewu_sussy)) if not bewu_sussy.empty else 1
        mean_BW_abnormal = float(np.mean(bewu_abnm)) if not bewu_abnm.empty else 0
        std_BW_abnormal = float(np.std(bewu_abnm)) if not bewu_abnm.empty else 1

        self.BW[self.labels[0]] = fuzz.gaussmf(self.BW.universe, mean_BW_normal + pi, std_BW_normal)
        self.BW[self.labels[1]] = fuzz.gaussmf(self.BW.universe, mean_BW_sussy + pi, std_BW_sussy)
        self.BW[self.labels[2]] = fuzz.gaussmf(self.BW.universe, mean_BW_abnormal + pi, std_BW_abnormal)

    def membership_fun_AP(self, pi=0):
        ape_norm = self.df.loc[self.df["Apgar"] >= 7, "Apgar"]
        ape_abnm = self.df.loc[self.df['Apgar'] < 5, 'Apgar']
        ape_sussy = self.df.loc[(self.df['Apgar'] < 7) & (self.df['Apgar'] >= 5), 'Apgar']

        mean_AP_normal = float(np.mean(ape_norm)) if not ape_norm.empty else 0
        std_AP_normal = float(np.std(ape_norm)) if not ape_norm.empty else 1
        mean_AP_sussy = float(np.mean(ape_sussy)) if not ape_sussy.empty else 0
        std_AP_sussy = float(np.std(ape_sussy)) if not ape_sussy.empty else 1
        mean_AP_abnormal = float(np.mean(ape_abnm)) if not ape_abnm.empty else 0
        std_AP_abnormal = float(np.std(ape_abnm)) if not ape_abnm.empty else 1

        self.AP[self.labels[0]] = fuzz.gaussmf(self.AP.universe, mean_AP_normal + pi, std_AP_normal)
        self.AP[self.labels[1]] = fuzz.gaussmf(self.AP.universe, mean_AP_sussy + pi, std_AP_sussy)
        self.AP[self.labels[2]] = fuzz.gaussmf(self.AP.universe, mean_AP_abnormal + pi, std_AP_abnormal)

    def make_plots(self):
        self.ph.view()
        self.BW.view()
        self.AP.view()

        plt.show()

    def rules(self):
        self.outcome[self.labels[2]] = fuzz.trapmf(self.outcome.universe, [0, 0, 0.5, 1])
        self.outcome[self.labels[1]] = fuzz.trapmf(self.outcome.universe, [1, 1, 1.5, 2])
        self.outcome[self.labels[0]] = fuzz.trapmf(self.outcome.universe, [2, 2, 2.5, 3])

        rule1 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[2]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule2 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[2]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule3 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[2]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])
        rule4 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[0]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule5 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[0]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule6 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[0]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])
        rule7 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[1]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule8 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[1]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule9 = ctrl.Rule(self.AP[self.labels[2]] & self.BW[self.labels[1]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])

        rule10 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[2]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule11 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[2]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule12 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[2]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])
        rule13 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[0]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule14 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[0]] & self.ph[self.labels[0]], self.outcome[self.labels[0]])
        rule15 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[0]] & self.ph[self.labels[1]], self.outcome[self.labels[0]])
        rule16 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[1]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule17 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[1]] & self.ph[self.labels[0]], self.outcome[self.labels[0]])
        rule18 = ctrl.Rule(self.AP[self.labels[0]] & self.BW[self.labels[1]] & self.ph[self.labels[1]], self.outcome[self.labels[1]])

        rule19 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[2]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule20 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[2]] & self.ph[self.labels[0]], self.outcome[self.labels[2]])
        rule21 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[2]] & self.ph[self.labels[1]], self.outcome[self.labels[2]])
        rule22 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[0]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule23 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[0]] & self.ph[self.labels[0]], self.outcome[self.labels[0]])
        rule24 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[0]] & self.ph[self.labels[1]], self.outcome[self.labels[0]])
        rule25 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[1]] & self.ph[self.labels[2]], self.outcome[self.labels[2]])
        rule26 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[1]] & self.ph[self.labels[0]], self.outcome[self.labels[1]])
        rule27 = ctrl.Rule(self.AP[self.labels[1]] & self.BW[self.labels[1]] & self.ph[self.labels[1]], self.outcome[self.labels[1]])

        outcome_control = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8,
                                             rule9, rule10, rule11, rule12, rule13, rule14, rule15,
                                             rule16, rule17, rule18, rule19, rule20, rule21, rule22,
                                             rule23, rule24, rule25, rule26, rule27])
        outcome_sim = ctrl.ControlSystemSimulation(outcome_control)
        return outcome_sim


    def train(self, train_data, pi=0, delta=0):
        # Adjust membership functions with pi
        self.membership_fun_ph(pi)
        self.membership_fun_BW(pi)
        self.membership_fun_AP(pi)

        # Adjust outcome membership functions with delta
        self.outcome[self.labels[2]] = fuzz.trapmf(self.outcome.universe,
                                                   [0 + delta, 0 + delta, 0.5 + delta, 1 + delta])
        self.outcome[self.labels[1]] = fuzz.trapmf(self.outcome.universe,
                                                   [1 + delta, 1 + delta, 1.5 + delta, 2 + delta])
        self.outcome[self.labels[0]] = fuzz.trapmf(self.outcome.universe,
                                                   [2 + delta, 2 + delta, 2.5 + delta, 3 + delta])

        # Define rules
        outcome_sim = self.rules()

        if isinstance(train_data, pd.DataFrame):
            # Use vectorized operations to compute the outcome
            train_data.apply(lambda row: self._compute_outcome(row, outcome_sim), axis=1)
            return train_data
        elif isinstance(train_data, dict):
            outcome_sim.input['Percentile'] = train_data['Percentile']
            outcome_sim.input['Apgar'] = train_data['Apgar']
            outcome_sim.input['Ph'] = train_data['Ph']
            outcome_sim.compute()
            return outcome_sim.output['outcome']

    def _compute_outcome(self, row, outcome_sim):
        outcome_sim.input['Percentile'] = row['Percentile']
        outcome_sim.input['Apgar'] = row['Apgar']
        outcome_sim.input['Ph'] = row['Ph']
        outcome_sim.compute()
        return outcome_sim.output['outcome']