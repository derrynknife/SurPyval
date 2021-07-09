import pytest
import numpy as np
from surpyval import Gumbel, Normal, Weibull, LogNormal
from surpyval import Logistic, LogLogistic, Uniform

DISTS = [Gumbel, Normal, Weibull, LogNormal, Logistic, LogLogistic, Uniform]
parameter_sample_bounds = [((1, 20), (0.5, 5)),
                           ((1, 100), (0.5, 100)),
                           ((1, 100), (0.5, 20)),
                           ((1, 3), (0.2, 1)),
                           ((1, 100), (0.5, 20)),
                           ((1, 100), (0.5, 20)),
                           ((1, 100), (1, 100)),
                           ]
FIT_SIZES = [5000, 10000, 20000, 50000, 100000]


def generate_mle_test_cases():
    for idx, dist in enumerate(DISTS):
        bounds = parameter_sample_bounds[idx]
        for kind in ['full', 'censored', 'truncated', 'interval']:
            yield dist, bounds, kind


def generate_mpp_test_cases():
    for idx, dist in enumerate(DISTS):
        bounds = parameter_sample_bounds[idx]
        for rr in ['x', 'y']:
            yield dist, bounds, rr


def generate_mom_test_cases():
    for idx, dist in enumerate(DISTS):
        bounds = parameter_sample_bounds[idx]
        yield dist, bounds


def generate_mps_test_cases():
    for idx, dist in enumerate(DISTS):
        bounds = parameter_sample_bounds[idx]
        yield dist, bounds

def generate_mse_test_cases():
    for idx, dist in enumerate(DISTS):
        bounds = parameter_sample_bounds[idx]
        yield dist, bounds


def idfunc(x):
    if type(x) is tuple:
        return 'bounds'
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


def censor_at(x, q, where='right'):
    c = np.zeros_like(x)
    x = np.copy(x)
    if where == 'right':
        x_q = np.quantile(x, 1 - q)
        mask = x > x_q
        c[mask] = 1
        x[mask] = x_q
        return x, c
    elif where == 'left':
        x_q = np.quantile(x, q)
        mask = x < x_q
        c[mask] = -1
        x[mask] = x_q
        return x, c
    elif where == 'both':
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


def truncate_at(x, q, where='right'):
    x = np.copy(x)
    if where == 'right':
        x_q = np.quantile(x, 1 - q)
        x = x[x < x_q]
        return x, None, x_q
    elif where == 'left':
        x_q = np.quantile(x, q)
        x = x[x > x_q]
        return x, x_q, None
    elif where == 'both':
        x_u = np.quantile(x, 1 - q)
        x_l = np.quantile(x, q)
        x = x[x < x_u]
        x = x[x > x_l]
        return x, x_l, x_u
    else:
        raise ValueError("'where' parameter not correctly defined")


@pytest.mark.parametrize("dist,bounds,kind",
                         generate_mle_test_cases(),
                         ids=idfunc)
def test_mle(dist, bounds, kind):
    for n in FIT_SIZES:
        test_params = []
        for b in bounds:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        if kind == 'full':
            model = dist.fit(x)
            tol = 0.1
        elif kind == 'censored':
            x, c = censor_at(x, 0.025, 'right')
            tol = 0.1
            model = dist.fit(x, c=c)
        elif kind == 'truncated':
            x, tl, tr = truncate_at(x, 0.05, 'both')
            model = dist.fit(x, tl=tl, tr=tr)
            tol = 0.15
        elif kind == 'interval':
            x, n = interval_censor(x)
            model = dist.fit(x=x, n=n)
            tol = 0.15
        if model.params == []:
            continue
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol).all():
            break
    else:
        raise AssertionError('MLE fit not very good in %s\n' % dist.name)


@pytest.mark.parametrize("dist,bounds,rr",
                         generate_mpp_test_cases(),
                         ids=idfunc)
def test_mpp(dist, bounds, rr):
    for n in FIT_SIZES:
        test_params = []
        tol = 0.1
        for b in bounds:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(10000, *test_params)
        model = dist.fit(x=x, rr=rr, how='MPP')
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol).all():
            break
    else:
        raise AssertionError('MPP fit not very good in %s\n' % dist.name)


@pytest.mark.parametrize("dist,bounds",
                         generate_mom_test_cases(),
                         ids=idfunc)
def test_mom(dist, bounds):
    for n in FIT_SIZES:
        test_params = []
        # 1% accuracy!!
        tol = 0.01
        for b in bounds:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        model = dist.fit(x=x, how='MOM')
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol).all():
            break
    else:
        raise AssertionError('MOM fit not very good in %s\n' % dist.name)


@pytest.mark.parametrize("dist,bounds",
                         generate_mps_test_cases(),
                         ids=idfunc)
def test_mps(dist, bounds):
    for n in FIT_SIZES:
        test_params = []
        # 1% accuracy!!
        tol = 0.01
        for b in bounds:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        model = dist.fit(x=x, how='MPS')
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol).all():
            break
    else:
        raise AssertionError('MPS fit not very good in %s\n' % dist.name)

@pytest.mark.parametrize("dist,bounds",
                         generate_mse_test_cases(),
                         ids=idfunc)
def test_mse(dist, bounds):
    for n in FIT_SIZES:
        test_params = []
        # 1% accuracy!!
        tol = 0.01
        for b in bounds:
            test_params.append(np.random.uniform(*b))
        test_params = np.array(test_params)
        x = dist.random(n, *test_params)
        model = dist.fit(x=x, how='MSE')
        fitted_params = np.array(model.params)
        max_params = np.max([fitted_params, test_params], axis=0)
        diff = np.abs(fitted_params - test_params) / max_params
        if (diff < tol).all():
            break
    else:
        raise AssertionError('MPS fit not very good in %s\n' % dist.name)

