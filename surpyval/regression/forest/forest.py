class RandomSurvivalForest:
    """Random Survival Forest

    Specs:
    - n_trees `Tree`'s trained, each given independently bootstrapped samples
    - Each tree is trained
    - Each predicition is evaluated for each tree, the Weibull models are
      collected, then averaged

    """

    @classmethod
    def fit(self, x, Z, c=None):
        # Make n_trees Tree's, and provide each with bootstrapped samples
        return self

    def df(self):
        pass

    def ff(self):
        pass

    def sf(self):
        pass

    def hf(self):
        pass

    def Hf(self):
        pass
