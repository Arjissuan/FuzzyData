from src.fuzzy_functions import FuzzyMethods

if __name__ == "__main__":
    fzm = FuzzyMethods()
    fzm.membership_fun_ph()
    fzm.membership_fun_AP()
    fzm.membership_fun_BW()

    fzm.make_plots()
