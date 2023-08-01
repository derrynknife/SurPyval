import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.stats import uniform

from surpyval import Exponential, Weibull
from surpyval.recurrence.nonparametric import NonParametricCounting
from surpyval.utils.recurrent_utils import handle_xicn

DT_WARN = "Small increment encountered, may have trouble reaching T."


class G1Renewal:
    def __init__(self, model, q):
        self.model = model
        self.q = q

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
        life, *scale = self.model.params
        q = self.q
        self.initialize_simulation()

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            for j in range(0, events + 1):
                ui = self.get_uniform_random_number()
                if self.model.dist == Exponential:
                    new_life = 1.0 / (1.0 / life * (1 + q) ** j)
                else:
                    new_life = life * (1 + q) ** j
                xi = self.model.dist.qf(ui, new_life, *scale)
                running += xi
                xicn["x"].append(running)
                xicn["i"].append(i + 1)
                xicn["c"].append(0)
                xicn["n"].append(1)

        self.clear_simulation()

        model = NonParametricCounting.fit(**xicn)
        mask = model.mcf_hat < events
        model.x = model.x[mask]
        model.mcf_hat = model.mcf_hat[mask]
        model.var = None
        return model

    def time_terminated_simulation(self, T, items=1, tol=1e-5):
        life, *scale = self.model.params
        q = self.q
        self.initialize_simulation()
        convergence_problem = False

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            j = 0
            while True:
                ui = self.get_uniform_random_number()
                if self.model.dist == Exponential:
                    new_life = 1.0 / (1.0 / life * (1 + q) ** j)
                else:
                    new_life = life * (1 + q) ** j
                xi = self.model.dist.qf(ui, new_life, *scale)
                running += xi
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
            warnings.warn(DT_WARN)

        model = NonParametricCounting.fit(**xicn)
        model.var = None

        return model

    @classmethod
    def create_negll_func(cls, x, i, c, n, dist):
        def negll_func(params):
            ll = 0
            q = params[0]
            life, *scale = params[1:]

            for item in set(i):
                mask_item = i == item
                x_item = np.atleast_1d(x[mask_item])
                c_item = np.atleast_1d(c[mask_item])
                n_item = np.atleast_1d(n[mask_item])
                for j in range(0, len(x_item)):
                    if dist == Exponential:
                        new_life = 1.0 / (1.0 / life * (1 + q) ** j)
                    else:
                        new_life = life * (1 + q) ** j
                    if c_item[j] == 0:
                        ll += n_item[j] * dist.log_df(
                            x_item[j], new_life, *scale
                        )
                    elif c_item[j] == 1:
                        ll += n_item[j] * dist.log_sf(
                            x_item[j], new_life, *scale
                        )
            return -ll

        return negll_func

    @classmethod
    def fit_from_recurrent_data(cls, data, dist=Weibull, init=None):
        if init is None:
            dist_params = dist.fit(
                data.interarrival_times, data.c, data.n
            ).params

        neg_ll = cls.create_negll_func(
            data.interarrival_times, data.i, data.c, data.n, dist
        )

        results = []
        # Iterate over different initial values for q
        # result is sensitive to initial value of q
        if init is None:
            for q_init in [0.0001, 1.0, 2.0]:
                init = [q_init, *dist_params]
                res = minimize(
                    neg_ll,
                    init,
                    bounds=[(-1, None), *dist.bounds],
                    method="Nelder-Mead",
                )
                if res.success:
                    results.append(res)

            if results == []:
                raise ValueError(
                    "Could not find a good solution. "
                    + "Try using `init` for better initial guess."
                )
            else:
                res = results[np.argmin([res.fun for res in results])]
        else:
            res = minimize(
                neg_ll,
                init,
                bounds=[(-1, None), *dist.bounds],
                method="Nelder-Mead",
            )

        underlying_model = dist.from_params(list(res.x[1:]))
        q = res.x[0]
        out = cls(underlying_model, q)
        out.res = res
        out.data = data
        return out

    @classmethod
    def fit(cls, x, i, c, n, dist=Weibull, init=None):
        # Wrangle data
        # Rest of the data assumes values are in ascending order.
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return cls.fit_from_recurrent_data(data, dist=dist, init=init)
