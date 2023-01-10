import numpy as np
import pytest

from surpyval import (
    Beta,
    ExpoWeibull,
    Gamma,
    Gumbel,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Uniform,
    Weibull,
)

DISTS = [
    Gumbel,
    Normal,
    Weibull,
    LogNormal,
    Logistic,
    LogLogistic,
    Uniform,
    Beta,
    ExpoWeibull,
    Gamma,
]

parameter_sample_random_parameters = [
    ((1, 20), (0.5, 5)),
    ((1, 100), (0.5, 100)),
    ((1, 100), (0.5, 20)),
    ((1, 3), (0.2, 1)),
    ((1, 100), (0.5, 20)),
    ((1, 100), (0.5, 20)),
    ((1, 100), (1, 100)),
    ((0.1, 30), (0.1, 30)),
    ((1, 30), (0.1, 10), (0.5, 1.5)),
    ((1, 30), (0.1, 10)),
]
FIT_SIZES = [1_000, 10_000, 100_000]


def generate_mle_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        for kind in ["full", "censored", "truncated", "interval"]:
            yield dist, random_parameters, kind


def generate_small_mle_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        for kind in ["full"]:
            yield dist, random_parameters, kind


def generate_mpp_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        for rr in ["x", "y"]:
            yield dist, random_parameters, rr


def generate_mom_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        yield dist, random_parameters


def generate_mps_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        yield dist, random_parameters


def generate_mse_test_cases():
    for idx, dist in enumerate(DISTS):
        random_parameters = parameter_sample_random_parameters[idx]
        yield dist, random_parameters


def idfunc(x):
    if type(x) is tuple:
        return "random_parameters"
    elif type(x) is str:
        return x
    else:
        return x.name


def interval_censor(x, n=100):
    n, xx = np.histogram(x, bins=n)
    x = np.vstack([xx[0:-1], xx[1::]]).T
    x = x[n > 0]
    n = n[n > 0]
    return x, n


def censor_at(x, q, where="right"):
    c = np.zeros_like(x)
    x = np.copy(x)
    if where == "right":
        x_q = np.quantile(x, 1 - q)
        mask = x > x_q
        c[mask] = 1
        x[mask] = x_q
        return x, c
    elif where == "left":
        x_q = np.quantile(x, q)
        mask = x < x_q
        c[mask] = -1
        x[mask] = x_q
        return x, c
    elif where == "both":
        x_u = np.quantile(x, 1 - q)
        x_l = np.quantile(x, q)
        mask_l = x < x_l
        mask_u = x > x_u
        c[mask_l] = -1
        c[mask_u] = 1
        x[mask_l] = x_l
        x[mask_u] = x_u
        return x, c
    else:
        raise ValueError("'where' parameter not correctly defined")


def truncate_at(x, q, where="right"):
    x = np.copy(x)
    if where == "right":
        x_q = np.quantile(x, 1 - q)
        x = x[x < x_q]
        return x, None, x_q
    elif where == "left":
        x_q = np.quantile(x, q)
        x = x[x > x_q]
        return x, x_q, None
    elif where == "both":
        x_u = np.quantile(x, 1 - q)
        x_l = np.quantile(x, q)
        x = x[x < x_u]
        x = x[x > x_l]
        return x, x_l, x_u
    else:
        raise ValueError("'where' parameter not correctly defined")


@pytest.mark.parametrize(
    "dist,random_parameters,kind", generate_mle_test_cases(), ids=idfunc
)
def test_mle_convergence(dist, random_parameters, kind):
    tol = 0.025
    for n in FIT_SIZES:
        test_params = []
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        if kind == "full":
            model = dist.fit(x)
        elif kind == "censored":
            x, c = censor_at(x, 0.025, "right")
            model = dist.fit(x, c=c)
        elif kind == "truncated":
            x, tl, tr = truncate_at(x, 0.05, "both")
            model = dist.fit(x, tl=tl, tr=tr)
        elif kind == "interval":
            x, n = interval_censor(x)
            model = dist.fit(x=x, n=n)
        if len(model.params) == 0:
            continue
        fitted_params = np.array(model.params)
        if dist.name == "Uniform":
            fitted_params = np.sort(fitted_params)
            test_params = np.sort(test_params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        # Decrease the tolerance for every parameter
        # e.g. Weibull (2 params) tol will be 5%
        # ExpoWeibull the tolerance will be 7.5%
        if (diff < tol * dist.k).all():
            break
    else:
        raise AssertionError("MLE convergence not good for %s\n" % dist.name)


@pytest.mark.parametrize(
    "dist,random_parameters,kind", generate_small_mle_test_cases(), ids=idfunc
)
def test_mle_convergence_small(dist, random_parameters, kind):
    tol = 0.03
    for n in [100, 250, 500]:
        test_params = []
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        if kind == "full":
            model = dist.fit(x)
        if len(model.params) == 0:
            continue
        fitted_params = np.array(model.params)
        if dist.name == "Uniform":
            fitted_params = np.sort(fitted_params)
            test_params = np.sort(test_params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        # Decrease the tolerance for every parameter
        # e.g. Weibull (2 params) tol will be 6%
        # ExpoWeibull the tolerance will be 9%
        if (diff < tol * dist.k).all():
            break
    else:
        raise AssertionError(
            "MLE fit for small data not good for %s\n" % dist.name
        )


@pytest.mark.parametrize(
    "dist,random_parameters,rr", generate_mpp_test_cases(), ids=idfunc
)
def test_mpp(dist, random_parameters, rr):
    if dist not in [Beta, ExpoWeibull]:
        for n in FIT_SIZES:
            test_params = []
            tol = 0.025
            for b in random_parameters:
                test_params.append(np.random.uniform(*b))
            test_params = np.array(test_params)
            x = dist.random(10000, *test_params)
            model = dist.fit(x=x, rr=rr, how="MPP", heuristic="Nelson-Aalen")
            fitted_params = np.array(model.params)
            if dist.name == "Uniform":
                fitted_params = np.sort(fitted_params)
                test_params = np.sort(test_params)
            max_params = np.max([fitted_params, test_params], axis=0)
            diff = np.abs(fitted_params - test_params) / max_params
            if (diff < tol * dist.k).all():
                break
        else:
            raise AssertionError("MPP fit not very good in %s\n" % dist.name)


@pytest.mark.parametrize(
    "dist,random_parameters", generate_mom_test_cases(), ids=idfunc
)
def test_mom(dist, random_parameters):
    for n in FIT_SIZES:
        test_params = []
        tol = 0.025
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        model = dist.fit(x=x, how="MOM")
        fitted_params = np.array(model.params)
        if dist.name == "Uniform":
            fitted_params = np.sort(fitted_params)
            test_params = np.sort(test_params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol * dist.k).all():
            break
    else:
        raise AssertionError("MOM fit not very good in %s\n" % dist.name)


@pytest.mark.parametrize(
    "dist,random_parameters", generate_mps_test_cases(), ids=idfunc
)
def test_mps(dist, random_parameters):
    for n in FIT_SIZES:
        test_params = []
        tol = 0.01
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        model = dist.fit(x=x, how="MPS")
        fitted_params = np.array(model.params)
        if dist.name == "Uniform":
            fitted_params = np.sort(fitted_params)
            test_params = np.sort(test_params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol * dist.k).all():
            break
    else:
        raise AssertionError("MPS fit not very good in %s\n" % dist.name)


@pytest.mark.parametrize(
    "dist,random_parameters", generate_mse_test_cases(), ids=idfunc
)
def test_mse(dist, random_parameters):
    for n in FIT_SIZES:
        test_params = []
        # 5% accuracy!!
        tol = 0.05
        for b in random_parameters:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        model = dist.fit(x=x, how="MSE")
        fitted_params = np.array(model.params)
        if dist.name == "Uniform":
            fitted_params = np.sort(fitted_params)
            test_params = np.sort(test_params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol).all():
            break
    else:
        raise AssertionError("MPS fit not very good in %s\n" % dist.name)
