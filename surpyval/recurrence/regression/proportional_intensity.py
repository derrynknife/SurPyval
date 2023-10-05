import numpy as np
import numpy_indexed as npi
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.special import gammaln

from surpyval.recurrence.nonparametric import NonParametricCounting
from surpyval.recurrence.parametric import CrowAMSAA
from surpyval.utils.recurrent_utils import handle_xicn


class ProportionalIntensityModel:
    def __repr__(self):
        return "Parametric Proportional Intensity Model with {} CIF".format(
            self.name
        )

    def cif(self, x, Z):
        return self.dist.cif(x, *self.params) * np.exp(Z @ self.coeffs)

    def iif(self, x, Z):
        return self.dist.iif(x, *self.params) * np.exp(Z @ self.coeffs)

    def inv_cif(self, x, Z):
        if hasattr(self.dist, "inv_cif"):
            return self.dist.inv_cif(x / np.exp(self.coeffs @ Z), *self.params)
        else:
            raise ValueError(
                "Inverse cif undefined for {}".format(self.dist.name)
            )

    def plot(self, ax=None):
        x, r, d = self.data.to_xrd()
        if ax is None:
            ax = plt.gcf().gca()

        x_plot = np.linspace(0, self.data.x.max(), 1000)
        Z_0 = self.data.Z.mean(axis=0)

        ax.step(x, (d / r).cumsum(), color="r", where="post")
        ax.plot(x_plot, self.cif(x_plot, Z_0), color="b")
        return ax

    def count_terminated_simulation(self, events, Z, items=1):
        self.initialize_simulation()

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            j = 0
            x_prev = 0
            for j in range(0, events + 1):
                ui = self.get_uniform_random_number()
                u_adj = ui * np.exp(-self.cif(x_prev, Z))
                xi = self.inv_cif(-np.log(u_adj), Z) - x_prev
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

    def time_terminated_simulation(self, T, Z, items=1, tol=1e-5):
        self.initialize_simulation()
        convergence_problem = False

        xicn = {"x": [], "i": [], "c": [], "n": []}

        for i in range(0, items):
            running = 0
            j = 0
            x_prev = 0
            while True:
                ui = self.get_uniform_random_number()
                u_adj = ui * np.exp(-self.cif(x_prev, Z))
                xi = self.inv_cif(-np.log(u_adj), Z) - x_prev
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


class ProportionalIntensityHPP:
    @classmethod
    def iif(self, x, rate):
        return np.ones_like(x) * rate

    @classmethod
    def cif(self, x, rate):
        return rate * x

    @classmethod
    def inv_cif(self, cif, rate):
        return cif / rate

    @classmethod
    def create_negll_func(cls, data):
        x, c, n = data.x, data.c, data.n
        Z = data.Z
        x_prev = data.find_x_previous()

        has_observed = True if 0 in c else False
        has_right_censoring = True if 1 in c else False
        has_left_censoring = True if -1 in c else False
        has_interval_censoring = True if x.ndim == 2 else False

        x_l = x if x.ndim == 1 else x[:, 0]
        x_r = x[:, 1] if x.ndim == 2 else None
        x_prev_r = x_prev[:, 1] if x_prev.ndim == 2 else x_prev

        # This code splits each observation type, if it exists, into its own
        # array. This is done to avoid having to simplify the log-likelihood
        # function to account for the different types of observations.

        # Further by calculating the sum of the needed arrays, we can avoid
        # having to do array sums in the log-likelihood function. This will be
        # faster, especially for large datasets.

        # Although this code is a bit more complex it results in a longer time
        # to create the log-likelihood function, but a faster time to evaluate
        # the log-likelihood function.

        # In conclusion, this is a ridiculous optimisation that is probably
        # not worth the effort that went into it.
        if has_observed:
            x_o = x_l[c == 0]
            x_prev_o = x_prev_r[c == 0]
            len_observed = len(x_o)
            # Don't change the order of the subtraction
            # Doing the analytic simplification of the log-likelihood
            # shows that this is the correct order when using "+" for the
            # specific term.
            x_o = x_prev_o - x_o
            Z_o = Z[c == 0]
        else:
            x_o = 0.0
            len_observed = 0.0
            Z_o = np.zeros((1, Z.shape[1]))

        if has_right_censoring:
            x_right = x_l[c == 1]
            x_right_prev = x_prev_r[c == 1]
            x_right = x_right_prev - x_right
            Z_right = Z[c == 1]
        else:
            Z_right = np.zeros((1, Z.shape[1]))
            x_right = 0.0

        if has_left_censoring:
            x_left = x_l[c == -1]
            n_left = n[c == -1]
            log_xl = np.log(x_left)
            n_log_x_left = n_left * log_xl
            n_log_x_left_sum = n_log_x_left.sum()
            n_left_sum = n_left.sum()
            n_l_factorial = gammaln(n_left + 1)
            n_l_factorial_sum = n_l_factorial.sum()
        else:
            n_log_x_left_sum = 0.0
            x_left = 0.0
            n_left_sum = 0.0
            n_left = 0.0
            n_l_factorial_sum = 0.0
            Z_left = np.zeros((1, Z.shape[1]))

        if has_interval_censoring:
            x_i_l = x_l[c == 2]
            x_i_r = x_r[c == 2]
            delta_xi = x_i_r - x_i_l

            n_interval = n[c == 2]
            n_interval_sum = n_interval.sum()

            n_log_x_interval_sum = (n_interval * np.log(delta_xi)).sum()
            n_i_factorial_sum = gammaln(n_interval + 1).sum()
        else:
            n_interval = 0.0
            n_interval_sum = 0.0
            n_log_x_interval_sum = 0.0
            n_i_factorial_sum = 0.0
            Z_i = np.zeros((1, Z.shape[1]))
            delta_xi = 0.0

        def negll_func(params):
            log_rate = params[0]
            rate = np.exp(log_rate)
            beta_coeffs = params[1:]

            phi_exponent_observed = np.dot(Z_o, beta_coeffs)
            ll = (
                phi_exponent_observed.sum()
                + len_observed * log_rate
                + rate * (x_o * np.exp(phi_exponent_observed)).sum()
            )

            phi_right = np.exp(np.dot(Z_right, beta_coeffs))
            ll += rate * (x_right * phi_right).sum()

            phi_exponent_left = np.dot(Z_left, beta_coeffs)
            ll += (
                (n_left * phi_exponent_left).sum()
                + log_rate * n_left_sum
                + n_log_x_left_sum
                - rate * (np.exp(phi_exponent_left) * x_left).sum()
                - n_l_factorial_sum
            )

            phi_exponent_interval = np.dot(Z_i, beta_coeffs)
            ll += (
                (n_interval * phi_exponent_interval).sum()
                + log_rate * n_interval_sum
                + n_log_x_interval_sum
                - rate * (np.exp(phi_exponent_interval) * delta_xi).sum()
                - n_i_factorial_sum
            )

            return -ll

        return negll_func

    @classmethod
    def fit(cls, x, Z, i=None, c=None, n=None, init=None):
        data = handle_xicn(x, i, c, n, Z=Z, as_recurrent_data=True)

        out = ProportionalIntensityModel()
        out.dist = cls
        out.data = data

        out.param_names = ["lambda"]
        out.bounds = ((0, None),)
        out.support = (0.0, np.inf)
        out.name = "Homogeneous Poisson Process"

        init = (data.n[data.c == 0]).sum() / npi.group_by(data.i).max(data.x)[
            1
        ].sum()

        num_covariates = Z.shape[1]
        init = np.append(np.log(init), np.zeros(num_covariates))

        neg_ll = cls.create_negll_func(data)

        res = minimize(neg_ll, [init])
        out.res = res
        out.params = np.atleast_1d(np.exp(res.x[0]))
        out.coeffs = np.atleast_1d(res.x[1:])
        out.name = "Homogeneous Poisson Process"

        return out


class ProportionalIntensityNHPP:
    @classmethod
    def iif(self, x, rate):
        return np.ones_like(x) * rate

    @classmethod
    def cif(self, x, rate):
        return rate * x

    @classmethod
    def inv_cif(self, cif, rate):
        return cif / rate

    @classmethod
    def create_negll_func(self, data, dist):
        x, c, n = data.x, data.c, data.n
        Z = data.Z
        # Covariates
        x_prev = data.find_x_previous()

        has_interval_censoring = True if 2 in c else False
        has_observed = True if 0 in c else False
        has_left_censoing = True if -1 in c else False
        has_right_censoring = True if 1 in c else False

        x_l = x if x.ndim == 1 else x[:, 0]
        x_r = x[:, 1] if x.ndim == 2 else None

        x_prev_l = x_prev if x_prev.ndim == 1 else x[:, 0]
        x_prev_r = x_prev[:, 1] if x_prev.ndim == 2 else None

        # Untangle the observed data
        x_o = x_l[c == 0] if has_observed else np.array([])
        if has_interval_censoring:
            x_o_prev = x_prev_r[c == 0] if has_observed else np.array([])
        else:
            x_o_prev = x_prev_l[c == 0] if has_observed else np.array([])
        Z_o = Z[c == 0] if has_observed else np.zeros((1, Z.shape[1]))

        # Untangle the right censored data
        x_right = x_l[c == 1] if has_right_censoring else np.array([])
        if has_interval_censoring:
            x_right_prev = (
                x_prev_r[c == 1] if has_right_censoring else np.array([])
            )
        else:
            x_right_prev = (
                x_prev_l[c == 1] if has_right_censoring else np.array([])
            )
        Z_right = (
            Z[c == 1] if has_right_censoring else np.zeros((1, Z.shape[1]))
        )

        # Untangle the left censored data
        x_left = x_l[c == -1] if has_left_censoing else np.array([])
        n_left = n[c == -1] if has_left_censoing else np.array([])
        Z_left = Z[c == -1] if has_left_censoing else np.zeros((1, Z.shape[1]))

        # Untangle the interval censored data
        x_i_l = x_l[c == 2] if has_interval_censoring else np.array([])
        x_i_r = x_r[c == 2] if has_interval_censoring else np.array([])
        n_i = n[c == 2] if has_interval_censoring else np.array([])
        Z_i = (
            Z[c == 2] if has_interval_censoring else np.zeros((1, Z.shape[1]))
        )

        # Using the empty arrays avoids the need for if statements in the
        # likelihood function. It also means that the likelihood function
        # will not encounter any invalid values since taking the log of 0
        # will not occur.

        def negll_func(params):
            dist_params = params[: len(dist.param_names)]
            beta_coeffs = params[len(dist.param_names) :]
            # ll of directly observed
            phi_exponents_observed = np.dot(Z_o, beta_coeffs)
            delta_cif_o = dist.cif(x_o_prev, *dist_params) - dist.cif(
                x_o, *dist_params
            )
            # TODO: Implement log_iif functions
            ll = (
                dist.log_iif(x_o, *dist_params)
                + phi_exponents_observed
                + (np.exp(phi_exponents_observed) * delta_cif_o)
            ).sum()

            # ll of right censored
            phi_right = np.exp(np.dot(Z_right, beta_coeffs))
            delta_cif_right = dist.cif(x_right_prev, *dist_params) - dist.cif(
                x_right, *dist_params
            )
            ll += (phi_right * delta_cif_right).sum()

            # ll of left censored
            delta_cif_left = dist.cif(x_left, *dist_params)
            phi_exponents_left = np.dot(Z_left, beta_coeffs)
            phi_left = np.exp(phi_exponents_left)
            ll += (
                n_left * phi_exponents_left
                + n_left * np.log(delta_cif_left)
                - phi_left * delta_cif_left
                - gammaln(n_left + 1)
            ).sum()

            # ll of interval censored
            delta_cif_interval = dist.cif(x_i_r, *dist_params) - dist.cif(
                x_i_l, *dist_params
            )
            phi_exponents_interval = np.dot(Z_i, beta_coeffs)
            phi_interval = np.exp(phi_exponents_interval)

            ll += (
                n_i * phi_exponents_interval
                + n_i * np.log(delta_cif_interval)
                - phi_interval * delta_cif_interval
                - gammaln(n_i + 1)
            ).sum()

            return -ll

        return negll_func

    @classmethod
    def fit_from_recurrent_data(cls, data, dist, init=None):
        out = ProportionalIntensityModel()
        out.dist = dist
        out.data = data

        init = np.ones(len(dist.param_names))

        num_covariates = data.Z.shape[1]
        init = np.append(init, np.zeros(num_covariates))

        neg_ll = cls.create_negll_func(data, dist)

        res = minimize(
            neg_ll,
            [init],
            method="Nelder-Mead",
        )
        out.res = res
        out.params = res.x[: len(dist.param_names)]
        out.coeffs = res.x[len(dist.param_names) :]
        out.name = "Non-Homogeneous Poisson Process"

        return out

    @classmethod
    def fit(cls, x, Z, i=None, c=None, n=None, dist=CrowAMSAA, init=None):
        data = handle_xicn(x, i, c, n, Z=Z, as_recurrent_data=True)
        return cls.fit_from_recurrent_data(data, dist, init)
