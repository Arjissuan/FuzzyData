from src.fuzzy_functions import FuzzyMethods

if __name__ == "__main__":
    fzm = FuzzyMethods()
    fzm.membership_fun_ph()
    fzm.membership_fun_AP()
    fzm.membership_fun_BW()
    fzm.make_plots()
    # fzm.make_plots()

    example_data = {
        'Percentile': 2,
        'Apgar': 9,
        'Ph': 7.15
    }

    print(fzm.rules(example_data))
