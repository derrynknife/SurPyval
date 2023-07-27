import numpy as np
from scipy.optimize import minimize
from scipy.stats import uniform

from surpyval import Weibull
from surpyval.recurrence.nonparametric import NonParametricCounting
from surpyval.utils.recurrent_utils import handle_xicn


class GeneralizedRenewal:
    def __init__(self, model, q, kijima_type="i"):
        self.model = model
        self.q = q
        if kijima_type == "i":
            self.virtual_age_function = self.kijima_i
        elif kijima_type == "ii":
            self.virtual_age_function = self.kijima_ii

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

    @classmethod
    def kijima_i(self, v, x, q):
        return v + q * x

    @classmethod
    def kijima_ii(self, v, x, q):
        return q * (v + x)

    def count_terminated_simulation(self, events, items=1):
        q = self.q
        self.initialize_simulation()

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            virtual_age = 0
            running = 0
            for j in range(0, events):
                ui = self.get_uniform_random_number()
                u_adj = ui * self.model.sf(virtual_age)
                xi = self.model.qf(1 - u_adj) - virtual_age
                # Update virtual age
                virtual_age = self.virtual_age_function(virtual_age, xi, q)
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

    def time_terminated_simulation(self, T, items=1, tol=1e-2):
        q = self.q
        self.initialize_simulation()
        convergence_problem = False

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            virtual_age = 0
            j = 0
            while True:
                ui = self.get_uniform_random_number()
                u_adj = ui * self.model.sf(virtual_age)
                xi = self.model.qf(1 - u_adj) - virtual_age
                # Update virtual age
                virtual_age = self.virtual_age_function(virtual_age, xi, q)
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
            print("Warning: Convergence Problem")
        model = NonParametricCounting.fit(**xicn)
        model.var = None
        return model

    @classmethod
    def create_negll_func(cls, x, i, c, n, dist, kijima="i"):
        if kijima == "i":
            virtual_age_function = cls.kijima_i
        elif kijima == "ii":
            virtual_age_function = cls.kijima_ii

        def negll_func(params):
            ll = 0
            # print(params)
            q = params[0]
            params = params[1:]

            for item in set(i):
                mask_item = i == item
                virtual_age = 0
                x_item = np.atleast_1d(x[mask_item])
                c_item = np.atleast_1d(c[mask_item])
                n_item = np.atleast_1d(n[mask_item])

                for j in range(0, len(x_item)):
                    x_j = x_item[j] + virtual_age
                    if c_item[j] == 0:
                        ll += n_item[j] * (
                            dist.log_df(x_j, *params)
                            - dist.log_sf(virtual_age, *params)
                        )
                    elif c_item[j] == 1:
                        ll += n_item[j] * (
                            dist.log_sf(x_j, *params)
                            - dist.log_sf(virtual_age, *params)
                        )
                    virtual_age = virtual_age_function(
                        virtual_age, x_item[j], q
                    )
            return -ll

        return negll_func

    @classmethod
    def fit_from_recurrent_data(
        cls, data, dist=Weibull, kijima="i", init=None
    ):
        first_events = data.get_times_to_first_events()
        if init is None:
            try:
                dist_params = dist.fit(
                    first_events.x, first_events.c, first_events.n
                ).params
            except Exception:
                dist_params = dist.fit(
                    first_events.interarrival_times,
                    first_events.c,
                    first_events.n,
                ).params

        neg_ll = cls.create_negll_func(
            data.interarrival_times,
            data.i,
            data.c,
            data.n,
            dist,
            kijima=kijima,
        )

        if init is None:
            # Iterate over different initial values for q
            # result is (very!!) sensitive to initial value of q
            results = []
            for q_init in [0.0001, 1.0, 2.0]:
                init = [q_init, *dist_params]
                res = minimize(
                    neg_ll,
                    init,
                    bounds=[(0, None), *dist.bounds],
                    method="Nelder-Mead",
                )
                if res.success:
                    results.append(res)
            res = results[np.argmin([res.fun for res in results])]
        else:
            res = minimize(
                neg_ll,
                init,
                bounds=[(0, None), *dist.bounds],
                method="Nelder-Mead",
            )

        model = dist.from_params(list(res.x[1:]))
        q = res.x[0]
        out = cls(model, q, kijima)
        out.res = res

        return out

    @classmethod
    def fit(cls, x, i, c, n, dist=Weibull, kijima="i", init=None):
        data = handle_xicn(x, i, c, n, as_recurrent_data=True)
        return cls.fit_from_recurrent_data(data, dist, kijima, init=init)
