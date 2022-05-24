from surpyval import np


class CompetingRisks():
    def __init__(self):
        self.name = 'CR'

    def fit(x, c, n, causes):
        unique_causes = set(causes)
        cause_idx_map = {state : i for i, state in enumerate(unique_states)}

        xb, cb, nb, tb = surpyval.xcnt_handler(x=x, c=c, n=n)

        CIFs = {}

        for cause in unique_causes:
            CIFs[cause] = {

            }



