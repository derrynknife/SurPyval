"""Coverage of the parametric confidence bounds (``param_cb`` and ``cb``).

Data: n = 100 from the named distribution, right-censored by an independent
uniform censoring time (about 30% censored). At each replicate the Wald and
likelihood-ratio bounds on every parameter, the bounds on ``sf`` at the
true 10%, 50% and 90% quantiles, and the Wald bounds on the 10% quantile
(the B10 life) and the mean, are checked against the truth.

Two-sided bounds must cover 95% of the time and a one-sided lower bound on
``sf`` 95% too: a one-sided bound that put only ``alpha / 2`` in its tail
(or a two-sided one that put ``alpha`` in each) is off by five points,
which these checks see at every sample size used here.

Slack is 0.01 (see ``_montecarlo``): at n = 100 with 30% censoring the
Wald bounds on the log scale are within a point of nominal for these two
families, and the likelihood-ratio bounds closer still.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage

# (distribution, true parameters, censoring upper limit, seed)
CASES = {
    "Weibull": (sp.Weibull, [10.0, 2.0], 30.0, 101),
    "LogNormal": (sp.LogNormal, [2.0, 0.5], 25.0, 102),
}
N = 100


def _sample(dist, params, c_max, rng):
    true = dist.from_params(params)
    t = true.qf(rng.uniform(size=N))
    cens = rng.uniform(0, c_max, N)
    return np.minimum(t, cens), (cens < t).astype(int)


def _study(name, method, reps):
    dist, params, c_max, seed = CASES[name]
    rng = np.random.default_rng(seed + (0 if method == "wald" else 1000))
    true = dist.from_params(params)
    t_eval = true.qf(np.array([0.1, 0.5, 0.9]))
    sf_true = true.sf(t_eval)
    k = len(params)
    p_lo, p_hi = np.empty((reps, k)), np.empty((reps, k))
    s_lo, s_hi = np.empty((reps, 3)), np.empty((reps, 3))
    s_one = np.empty((reps, 3))
    # the B10 life and the mean (#494)
    q_lo, q_hi = np.empty(reps), np.empty(reps)
    m_lo, m_hi = np.empty(reps), np.empty(reps)
    censored = 0.0
    for r in range(reps):
        x, c = _sample(dist, params, c_max, rng)
        censored += c.mean() / reps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = dist.fit(x, c=c)
            for j, name_j in enumerate(dist.param_names):
                p_lo[r, j], p_hi[r, j] = model.param_cb(name_j, method=method)
            band = model.cb(t_eval, on="sf", method=method)
            s_lo[r], s_hi[r] = band[:, 0], band[:, 1]
            s_one[r] = model.cb(t_eval, on="sf", bound="lower", method=method)
            if method == "wald":
                q_lo[r], q_hi[r] = model.quantile_cb(0.1)
                m_lo[r], m_hi[r] = model.mean_cb()
    print("{} ({}): {:.0%} censored".format(name, method, censored))
    label = "{} {}".format(name, method)
    check_coverage(p_lo, p_hi, np.asarray(params), 0.95, label + " param_cb")
    check_coverage(s_lo, s_hi, sf_true, 0.95, label + " cb(sf)")
    check_coverage(
        s_one, np.inf, sf_true, 0.95, label + " cb(sf, bound='lower')"
    )
    if method == "wald":
        # (the likelihood-ratio quantile and mean bounds are left out:
        # 300 replicates of them took over 50 minutes)
        check_coverage(
            q_lo, q_hi, t_eval[0], 0.95, label + " quantile_cb(0.1)"
        )
        check_coverage(m_lo, m_hi, true.mean(), 0.95, label + " mean_cb")


@pytest.mark.parametrize("name", sorted(CASES))
def test_wald_coverage(name):
    _study(name, "wald", reps=1000)


@pytest.mark.parametrize("name", sorted(CASES))
def test_likelihood_ratio_coverage(name):
    # The profile bounds re-optimise at every candidate, ~0.8 s a replicate.
    _study(name, "lr", reps=300)
