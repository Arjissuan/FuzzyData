import skfuzzy as fuzz
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from skfuzzy import control as ctrl


class FuzzyMethods:
    def __init__(self, file_name="FDA_data.xls"):
        self.df = pd.read_excel(io=os.path.join(os.getcwd(), file_name))
        self.BW = ctrl.Antecedent(self.df.iloc[:, 0], label="Percentile")
        self.AP = ctrl.Antecedent(self.df.iloc[:, 1], label="Apgar")
        self.ph = ctrl.Antecedent(self.df.iloc[:, 2], label="Ph")
        self.labels = ("Normal", "Suspicious", "Abnormal")

    def membership_fun_ph(self):
        peha_norm = self.df.loc[self.df["Ph"] >= 7.2, "Ph"]
        peha_abnm = self.df.loc[self.df['Ph'] < 7.1, "Ph"]
        peha_sussy = self.df.loc[(self.df.loc[:, 'Ph'] >= 7.1) & (self.df.loc[:, 'Ph'] <= 7.2), "Ph"]

        self.ph[self.labels[0]] = fuzz.gaussmf(self.ph.universe, float(np.mean(peha_norm)), float(np.std(peha_norm)))
        self.ph[self.labels[2]] = fuzz.gaussmf(self.ph.universe, float(np.mean(peha_sussy)), float(np.std(peha_sussy)))
        self.ph[self.labels[1]] = fuzz.gaussmf(self.ph.universe, float(np.mean(peha_abnm)), float(np.std(peha_abnm)))

    def membership_fun_BW(self):
        bewu_norm = self.df.loc[self.df["Percentile"] >= 10, "Percentile"]
        bewu_abnm = self.df.loc[self.df['Percentile'] <= 5, 'Percentile']
        bewu_sussy = self.df.loc[(self.df['Percentile'] < 10) & (self.df['Percentile'] > 5), 'Percentile']
        self.BW[self.labels[0]] = fuzz.gaussmf(self.BW.universe, float(np.mean(bewu_norm)), float(np.std(bewu_norm)))
        self.BW[self.labels[2]] = fuzz.gaussmf(self.BW.universe, float(np.mean(bewu_sussy)), float(np.std(bewu_sussy)))
        self.BW[self.labels[1]] = fuzz.gaussmf(self.BW.universe, float(np.mean(bewu_abnm)), float(np.std(bewu_abnm)))

    def membership_fun_AP(self):
        ape_norm = self.df.loc[self.df["Apgar"] >= 7, "Apgar"]
        ape_abnm = self.df.loc[self.df['Apgar'] < 5, 'Apgar']
        ape_sussy = self.df.loc[(self.df['Apgar'] < 7) & (self.df['Apgar'] >= 5), 'Apgar']

        self.AP[self.labels[0]] = fuzz.gaussmf(self.AP.universe, float(np.mean(ape_norm)), float(np.std(ape_norm)))
        self.AP[self.labels[2]] = fuzz.gaussmf(self.AP.universe, float(np.mean(ape_abnm)), float(np.std(ape_abnm)))
        self.AP[self.labels[1]] = fuzz.gaussmf(self.AP.universe, float(np.mean(ape_sussy)), float(np.std(ape_sussy)))

    def make_plots(self):

        self.ph.view()

        self.BW.view()

        self.AP.view()

        plt.show()