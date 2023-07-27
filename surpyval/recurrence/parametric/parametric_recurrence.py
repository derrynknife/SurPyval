class ParametricRecurrenceModel:
    def __repr__(self):
        return "Parametric Counting Model with {} CIF".format(self.dist.name)

    def cif(self, x):
        return self.dist.cif(x, *self.params)

    def iif(self, x):
        return self.dist.iif(x, *self.params)

    def rocof(self, x):
        if hasattr(self.dist, "rocof"):
            return self.dist.rocof(x, *self.params)
        else:
            raise ValueError("rocof undefined for {}".format(self.dist.name))

    def inv_cif(self, x):
        if hasattr(self.dist, "inv_cif"):
            return self.dist.inv_cif(x, *self.params)
        else:
            raise ValueError(
                "Inverse cif undefined for {}".format(self.dist.name)
            )

    # TODO: random, to T, and to N
