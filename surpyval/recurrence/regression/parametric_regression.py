import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import uniform

from surpyval.recurrence.nonparametric import NonParametricCounting


class ParametricRecurrenceRegressionModel:
    def __repr__(self):
        return "Parametric Counting Model with {} CIF".format(self.dist.name)

    def initialize_simulation(self):
        self.us = uniform.rvs(size=100_000).tolist()

    def clear_simulation(self):
        del self.us

    def get_uniform_random_number(self):
        try:
            return self.us.pop()
        except IndexError:
            self.initialize_simulation()
            return self.us.pop()

    def count_terminated_simulation(self, events, items=1):
        self.initialize_simulation()

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            j = 0
            x_prev = 0
            for j in range(0, events + 1):
                ui = self.get_uniform_random_number()
                u_adj = ui * np.exp(-self.cif(x_prev))
                xi = self.inv_cif(-np.log(u_adj)) - x_prev
                running += xi
                x_prev = running
                xicn["i"].append(i + 1)
                xicn["n"].append(1)
                xicn["x"].append(running)
                xicn["c"].append(0)

        self.clear_simulation()

        model = NonParametricCounting.fit(**xicn)

        if self.dist.name == "CoxLewis":
            model.mcf_hat += np.exp(self.params[0])

        mask = model.mcf_hat <= events
        model.x = model.x[mask]
        model.mcf_hat = model.mcf_hat[mask]
        model.var = None
        return model

    def time_terminated_simulation(self, T, items=1, tol=1e-5):
        self.initialize_simulation()
        convergence_problem = False

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            j = 0
            x_prev = 0
            while True:
                ui = self.get_uniform_random_number()
                u_adj = ui * np.exp(-self.cif(x_prev))
                xi = self.inv_cif(-np.log(u_adj)) - x_prev
                running += xi
                x_prev = running
                xicn["i"].append(i + 1)
                xicn["n"].append(1)
                if running > T:
                    xicn["x"].append(T)
                    xicn["c"].append(1)
                    break
                elif xi < tol:
                    convergence_problem = True
                    xicn["x"].append(running)
                    xicn["c"].append(0)
                    break
                else:
                    xicn["x"].append(running)
                    xicn["c"].append(0)
                    j += 1

        self.clear_simulation()

        if convergence_problem:
            print("Maybe...")

        model = NonParametricCounting.fit(**xicn)

        if self.dist.name == "CoxLewis":
            model.mcf_hat += np.exp(self.params[0])
        model.var = None

        return model

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

    def plot(self, ax=None):
        x, r, d = self.data.to_xrd()
        if ax is None:
            ax = plt.gcf().gca()

        x_plot = np.linspace(0, self.data.x.max(), 1000)

        ax.step(x, (d / r).cumsum(), color="r", where="post")
        return ax.plot(x_plot, self.cif(x_plot), color="b")

    # TODO: random, to T, and to N
