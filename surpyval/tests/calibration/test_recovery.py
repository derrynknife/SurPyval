"""Parameter recovery: bias relative to the estimator's spread, and the
reported standard errors against that spread.

For each model, ``reps`` data sets are simulated from known parameters and
refitted. Per parameter, two checks (see ``_montecarlo.check_bias``):

- the standardised bias ``(mean estimate - truth) / sd`` is within
  ``3 / sqrt(reps) + 0.2``. A maximum likelihood estimator's bias is O(1/n)
  against an O(1/sqrt(n)) spread, so in standard deviations it is
  O(1/sqrt(n)) and not negligible at these sizes: the Weibull shape's is
  about ``1.4 beta / r`` for ``r`` failures, 0.18 of a standard deviation
  at r = 100 (the limited-failure study sees 0.23 at n = 150, and an exact
  maximisation of the same likelihood gives the same estimates). A fault
  in a likelihood -- truncation ignored, say -- is several standard
  deviations; the left-truncation study checks that it would be seen;
- where the model reports standard errors, their root mean square is
  within ``3 / sqrt(2 reps) + 0.1`` of the actual spread (so a 95% Wald
  interval built on them would be within about a point of nominal).

The data are simulated directly from the model's definition, never through
the package's own simulators.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.recurrent import ARA, GeneralizedOneRenewal
from surpyval.tests.calibration._montecarlo import check_bias

# --- univariate MLE with truncation and interval censoring ---------------


def _left_truncated(rng, n):
    # Delayed entry: a unit is seen only if it survives to its entry time.
    true = sp.Weibull.from_params([10.0, 2.0])
    x, tl = np.empty(0), np.empty(0)
    while x.size < n:
        t = true.qf(rng.uniform(size=n))
        entry = rng.uniform(0, 8.0, n)
        keep = t > entry
        x, tl = np.append(x, t[keep]), np.append(tl, entry[keep])
    return dict(x=x[:n], tl=tl[:n])


def _interval_censored(rng, n):
    # Inspections every 3 time units: each failure is known to an interval.
    t = sp.Weibull.from_params([10.0, 2.0]).qf(rng.uniform(size=n))
    left = 3.0 * np.floor(t / 3.0)
    return dict(xl=left, xr=left + 3.0)


def _right_truncated_censored(rng, n):
    # Right truncation at a per-unit horizon (only failures before it are
    # recorded), then exact observation.
    true = sp.LogNormal.from_params([2.0, 0.5])
    x, tr = np.empty(0), np.empty(0)
    while x.size < n:
        t = true.qf(rng.uniform(size=n))
        horizon = rng.uniform(8.0, 20.0, n)
        keep = t < horizon
        x, tr = np.append(x, t[keep]), np.append(tr, horizon[keep])
    return dict(x=x[:n], tr=tr[:n])


def _limited_failure(rng, n):
    # 70% of units can fail (Weibull(10, 2)); the rest never do. Followed
    # to 30, well past where the susceptible ones have failed.
    t = sp.Weibull.from_params([10.0, 2.0]).qf(rng.uniform(size=n))
    t[rng.uniform(size=n) > 0.7] = np.inf
    c = (t > 30.0).astype(int)
    return dict(x=np.minimum(t, 30.0), c=c)


UNIVARIATE = {
    "Weibull left-truncated": (
        sp.Weibull,
        _left_truncated,
        {},
        [10.0, 2.0],
        150,
        801,
    ),
    "Weibull interval-censored": (
        sp.Weibull,
        _interval_censored,
        {},
        [10.0, 2.0],
        150,
        802,
    ),
    "LogNormal right-truncated": (
        sp.LogNormal,
        _right_truncated_censored,
        {},
        [2.0, 0.5],
        150,
        803,
    ),
    "Weibull limited-failure": (
        sp.Weibull,
        _limited_failure,
        {"lfp": True},
        [10.0, 2.0, 0.7],
        150,
        804,
    ),
}


@pytest.mark.parametrize("name", list(UNIVARIATE))
def test_univariate_mle_recovery(name):
    dist, simulate, options, truth, n, seed = UNIVARIATE[name]
    rng = np.random.default_rng(seed)
    reps = 500
    k = len(truth)
    est, se = np.full((reps, k), np.nan), np.full((reps, k), np.nan)
    naive = np.full((reps, k), np.nan)
    for r in range(reps):
        data = simulate(rng, n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = dist.fit(**data, **options)
            if "tl" in data:
                naive[r] = dist.fit(x=data["x"]).params
        params = list(model.params)
        cov = np.diag(model.hess_inv)
        if options.get("lfp"):
            params.append(model.p)
            cov = np.diag(model.cov_matrix)
        est[r] = params
        se[r] = np.sqrt(cov)
    check_bias(est, truth, name, standard_errors=se)
    if np.isfinite(naive).all():
        # Design check: the same data fitted as if there were no delayed
        # entry is biased by far more than the tolerance.
        z = (naive.mean(axis=0) - truth) / naive.std(axis=0, ddof=1)
        print("ignoring the truncation: standardised bias", z.round(2))
        assert np.abs(z).max() > 3 / np.sqrt(reps) + 0.2


# --- regression families ----------------------------------------------------


def _design(rng, n):
    return np.column_stack([rng.binomial(1, 0.5, n), rng.normal(0, 1, n)])


def _censored(t, rng, c_max):
    cens = rng.uniform(0, c_max, t.size)
    return np.minimum(t, cens), (cens < t).astype(int)


def _cox_data(rng, n, beta):
    Z = _design(rng, n)
    t = 10.0 * (rng.exponential(size=n) / np.exp(Z @ beta)) ** (1 / 1.5)
    x, c = _censored(t, rng, 25.0)
    return x, Z, c


def test_cox_recovery_with_delayed_entry():
    # Left truncation: a unit enters the risk set at its entry time.
    rng = np.random.default_rng(811)
    beta = np.array([0.7, -0.5])
    reps, n = 500, 200
    est, se = np.empty((reps, 2)), np.empty((reps, 2))
    for r in range(reps):
        xs, Zs, cs, tls = [], [], [], []
        while sum(len(a) for a in xs) < n:
            x, Z, c = _cox_data(rng, n, beta)
            entry = rng.uniform(0, 5.0, n)
            keep = x > entry
            xs.append(x[keep])
            Zs.append(Z[keep])
            cs.append(c[keep])
            tls.append(entry[keep])
        x, Z = np.concatenate(xs)[:n], np.concatenate(Zs)[:n]
        c, tl = np.concatenate(cs)[:n], np.concatenate(tls)[:n]
        model = sp.CoxPH.fit(x=x, Z=Z, c=c, tl=tl)
        est[r] = model.params
        se[r] = np.sqrt(np.diag(np.linalg.inv(model.jac(model.params)[1])))
    check_bias(est, beta, "CoxPH (left-truncated)", standard_errors=se)


def test_buckley_james_recovery():
    # log T = 2 - (0.7 z1 - 0.5 z2) + N(0, 0.5): AFT coefficients +0.7,
    # -0.5 in surpyval's sign (a positive coefficient shortens life). No
    # standard errors: Buckley-James has only a bootstrap.
    rng = np.random.default_rng(812)
    beta = np.array([0.7, -0.5])
    reps, n = 500, 200
    est = np.empty((reps, 2))
    for r in range(reps):
        Z = _design(rng, n)
        t = np.exp(2.0 - Z @ beta + rng.normal(0, 0.5, n))
        x, c = _censored(t, rng, 25.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            est[r] = sp.BuckleyJames.fit(x=x, Z=Z, c=c).params
    check_bias(est, beta, "BuckleyJames")


def test_lin_ying_recovery():
    # h(t | Z) = 0.1 + 0.05 z1 + 0.03 |z2|, a constant baseline.
    rng = np.random.default_rng(813)
    beta = np.array([0.05, 0.03])
    reps, n = 500, 300
    est, se = np.empty((reps, 2)), np.empty((reps, 2))
    for r in range(reps):
        Z = _design(rng, n)
        Z[:, 1] = np.abs(Z[:, 1])
        t = rng.exponential(1 / (0.1 + Z @ beta))
        x, c = _censored(t, rng, 25.0)
        model = sp.AdditiveHazards.fit(x=x, Z=Z, c=c)
        est[r] = model.params
        se[r] = model.standard_errors()
    check_bias(est, beta, "AdditiveHazards (Lin-Ying)", standard_errors=se)


def test_frailty_recovery():
    # 60 groups of 5, gamma frailty of variance 0.5, Weibull(10, 1.5)
    # baseline, a unit-level binary covariate with coefficient 0.6.
    rng = np.random.default_rng(814)
    groups = np.repeat(np.arange(60), 5)
    truth = np.array([10.0, 1.5, 0.6, 0.5])
    reps = 300
    est, se = np.empty((reps, 4)), np.empty((reps, 4))
    for r in range(reps):
        u = rng.gamma(2.0, 0.5, 60)[groups]
        Z = rng.binomial(1, 0.5, (groups.size, 1))
        H = rng.exponential(size=groups.size) / (u * np.exp(0.6 * Z[:, 0]))
        t = 10.0 * H ** (1 / 1.5)
        x, c = _censored(t, rng, 30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sp.WeibullFrailty.fit(x, Z=Z, c=c, groups=groups)
        est[r] = model._param_vector()
        errors = model.standard_errors()
        se[r] = [errors[p] for p in model.parameter_names]
    check_bias(est, truth, "WeibullFrailty", standard_errors=se)


def test_lognormal_frailty_recovery():
    # As above with a log-normal frailty (#343): u = exp(w), w ~ N(0, 0.5),
    # so the baseline is that of a group of median frailty.
    rng = np.random.default_rng(343)
    groups = np.repeat(np.arange(60), 5)
    truth = np.array([10.0, 1.5, 0.6, 0.5])
    fitter = sp.Frailty(sp.Weibull, family="lognormal")
    reps = 300
    est, se = np.empty((reps, 4)), np.empty((reps, 4))
    for r in range(reps):
        u = np.exp(rng.normal(0.0, np.sqrt(0.5), 60))[groups]
        Z = rng.binomial(1, 0.5, (groups.size, 1))
        H = rng.exponential(size=groups.size) / (u * np.exp(0.6 * Z[:, 0]))
        t = 10.0 * H ** (1 / 1.5)
        x, c = _censored(t, rng, 30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fitter.fit(x, Z=Z, c=c, groups=groups)
        est[r] = model.params
        errors = model.standard_errors()
        se[r] = [errors[p] for p in model.parameter_names]
    check_bias(est, truth, "WeibullFrailty[lognormal]", standard_errors=se)


def test_cox_frailty_recovery():
    # The gamma frailty of test_frailty_recovery with the baseline left to
    # the data (#342): the coefficient and theta, whose standard error is
    # the profile likelihood's curvature.
    rng = np.random.default_rng(342)
    groups = np.repeat(np.arange(60), 5)
    truth = np.array([0.6, 0.5])
    reps = 200
    est, se = np.empty((reps, 2)), np.empty((reps, 2))
    for r in range(reps):
        u = rng.gamma(2.0, 0.5, 60)[groups]
        Z = rng.binomial(1, 0.5, (groups.size, 1))
        H = rng.exponential(size=groups.size) / (u * np.exp(0.6 * Z[:, 0]))
        t = 10.0 * H ** (1 / 1.5)
        x, c = _censored(t, rng, 30.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sp.CoxFrailty.fit(x, Z=Z, c=c, groups=groups)
        est[r] = model.params
        errors = model.standard_errors()
        se[r] = [errors[p] for p in model.parameter_names]
    check_bias(est, truth, "CoxFrailty", standard_errors=se)


# --- renewal models ---------------------------------------------------------


def _renewal_data(rng, items, t_end, next_failure):
    """Time-terminated renewal data; ``next_failure(history, u)`` returns the
    next failure time given the failure times so far."""
    xs, ids, cs = [], [], []
    for k in range(items):
        history = []
        while True:
            t = next_failure(history, rng.uniform())
            if t > t_end:
                break
            history.append(t)
        xs.extend([*history, t_end])
        ids.extend([k] * (len(history) + 1))
        cs.extend([0] * len(history) + [1])
    return np.array(xs), np.array(ids), np.array(cs)


def _weibull_after(age, u, alpha=10.0, beta=2.0):
    """The failure age of a Weibull unit that has survived to ``age``."""
    return alpha * ((age / alpha) ** beta - np.log(u)) ** (1 / beta)


def _g1(q):
    # The j-th interarrival (from 0) is (1 + q)^j times a Weibull(10, 2).
    def next_failure(history, u):
        last = history[-1] if history else 0.0
        return last + (1 + q) ** len(history) * _weibull_after(0.0, u)

    return next_failure


def _ara1(rho):
    # Arithmetic reduction of age, memory 1: the virtual age after the
    # repair at T is (1 - rho) T, and the unit ages from there.
    def next_failure(history, u):
        last = history[-1] if history else 0.0
        v = (1 - rho) * last
        return last + _weibull_after(v, u) - v

    return next_failure


RENEWAL = {
    "G1 (q = 0.1)": (
        lambda x, i, c: GeneralizedOneRenewal.fit(x, i, c=c),
        _g1(0.1),
        {"q": 0.1, "alpha": 10.0, "beta": 2.0},
        821,
    ),
    "ARA1 (rho = 0.5)": (
        lambda x, i, c: ARA.fit(x, i, c=c, m=1),
        _ara1(0.5),
        {"rho": 0.5, "alpha": 10.0, "beta": 2.0},
        822,
    ),
}


@pytest.mark.parametrize("name", list(RENEWAL))
def test_renewal_recovery(name):
    fit, next_failure, truth, seed = RENEWAL[name]
    rng = np.random.default_rng(seed)
    reps = 200
    names = list(truth)
    est, se = np.full((reps, 3), np.nan), np.full((reps, 3), np.nan)
    for r in range(reps):
        x, i, c = _renewal_data(rng, 15, 50.0, next_failure)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fit(x, i, c)
        order = [model.parameter_names.index(p) for p in names]
        est[r] = np.asarray(model._mle)[order]
        se[r] = np.asarray(model.standard_errors())[order]
    check_bias(est, [truth[p] for p in names], name, standard_errors=se)
