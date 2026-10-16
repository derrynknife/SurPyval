"""Coverage of the likelihood-ratio bounds of offset, limited-failure and
zero-inflated models (``param_cb`` and ``cb`` with ``method="lr"``).

Data: n = 100 Weibull lifetimes (scale 10), right censored by an
independent uniform time, with

- ``lfp``: shape 2, 40% of units that never fail (``lfp_p`` = 0.6);
- ``lfp_edge``: shape 2, every unit can fail (``lfp_p`` = 1, the edge of
  its space), fitted with ``lfp=True``;
- ``zi``: shape 2, 10% dead on arrival (``f0`` = 0.1);
- ``offset``: shape 3, offset 5 (a shape above 2, where the offset's
  likelihood is regular; Smith, 1985).

At each replicate the two-sided 95% bounds on every fitted parameter,
the extra one among them, and on ``sf`` at three true quantiles are
checked against the truth. Slack is 0.01, as for the plain models
(``test_coverage_parametric``).

On the edge, ``lfp_p`` = 1 is in the interval exactly when its profile
deviance there is within the critical value, and it is 0 there whenever
the estimate is on the edge -- asymptotically half the time (Self and
Liang, 1987), and more often at 100 units (82% of 2,000 replicates): the
two-sided 95% interval covers the edge 97.5% of the time asymptotically,
and the one-sided 95% upper bound (towards the edge) 95%. At 100 units
2,000 replicates of the deviance on the edge gave 98.5% and 97.2%, inside
the tolerances here (the study's 300 replicates: 98.7% and 98.3%).

Each study's replicates run on every core (the searches take about two
seconds a replicate); the studies run in a group of their own in CI.
"""

import os
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.calibration._montecarlo import check_coverage

N = 100
REPS = 300

# (seed, truth: parameters by name, the model, quantiles where sf is
# bounded)
STUDIES = {
    "lfp": (
        301,
        {"alpha": 10.0, "beta": 2.0, "lfp_p": 0.6},
        (0.1, 0.3, 0.5),
    ),
    "lfp_edge": (
        302,
        {"alpha": 10.0, "beta": 2.0, "lfp_p": 1.0},
        (0.1, 0.3, 0.5),
    ),
    "zi": (303, {"alpha": 10.0, "beta": 2.0, "f0": 0.1}, (0.2, 0.4, 0.6)),
    "offset": (
        304,
        {"alpha": 10.0, "beta": 3.0, "gamma": 5.0},
        (0.1, 0.3, 0.5),
    ),
}


def _truth(name):
    _, truth, _ = STUDIES[name]
    extras = {k: v for k, v in truth.items() if k in ("gamma", "lfp_p", "f0")}
    return sp.Weibull.from_params([truth["alpha"], truth["beta"]], **extras)


def _sample(name, rng):
    _, truth, _ = STUDIES[name]
    t = truth["alpha"] * rng.weibull(truth["beta"], N)
    if "lfp_p" in truth:
        t = np.where(rng.uniform(size=N) > truth["lfp_p"], np.inf, t)
        cens = rng.uniform(0, 30, N)
    elif "f0" in truth:
        t = np.where(rng.uniform(size=N) < truth["f0"], 0.0, t)
        cens = rng.uniform(0, 30, N)
    else:
        t = truth["gamma"] + t
        cens = truth["gamma"] + rng.uniform(0, 25, N)
    return np.minimum(t, cens), (cens < t).astype(int)


def _replicate(job):
    name, x, c = job
    _, truth, probs = STUDIES[name]
    kind = {"lfp": "lfp", "lfp_edge": "lfp", "zi": "zi", "offset": "offset"}
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        model = sp.Weibull.fit(x, c=c, **{kind[name]: True})
        params = [model.param_cb(k, method="lr") for k in truth]
        band = model.cb(_truth(name).qf(np.array(probs)), method="lr")
        upper = np.nan
        if name == "lfp_edge":
            upper = model.param_cb("lfp_p", bound="upper", method="lr")[0]
    return np.array(params), band, upper


def _study(name):
    seed, truth, probs = STUDIES[name]
    rng = np.random.default_rng(seed)
    jobs = [(name, *_sample(name, rng)) for _ in range(REPS)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        out = list(pool.map(_replicate, jobs))
    params = np.array([o[0] for o in out])
    band = np.array([o[1] for o in out])
    one_sided = np.array([o[2] for o in out])
    true = _truth(name)
    return truth, params, band, one_sided, true.sf(true.qf(np.array(probs)))


@pytest.mark.parametrize("name", ["lfp", "zi", "offset"])
def test_likelihood_ratio_coverage(name):
    truth, params, band, _, sf_true = _study(name)
    names = list(truth)
    values = np.array([truth[k] for k in names])
    check_coverage(
        params[:, :, 0], params[:, :, 1], values, 0.95, f"{name} param_cb"
    )
    check_coverage(band[:, :, 0], band[:, :, 1], sf_true, 0.95, f"{name} cb")


def test_likelihood_ratio_coverage_on_the_edge():
    truth, params, band, one_sided, sf_true = _study("lfp_edge")
    values = np.array([truth["alpha"], truth["beta"]])
    check_coverage(
        params[:, :2, 0], params[:, :2, 1], values, 0.95, "edge param_cb"
    )
    check_coverage(
        params[:, 2, 0], params[:, 2, 1], 1.0, 0.975, "edge lfp_p two-sided"
    )
    check_coverage(
        np.full_like(one_sided, -np.inf), one_sided, 1.0, 0.95, "edge upper"
    )
    check_coverage(band[:, :, 0], band[:, :, 1], sf_true, 0.95, "edge cb")
