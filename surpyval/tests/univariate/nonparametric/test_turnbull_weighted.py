"""The Turnbull estimate with sampling weights ``w``.

The point estimate is the weighted NPMLE: each row counts ``n w`` in the
EM. It is checked against R's ``icenReg::ic_np(..., weights = w)``
(scripts/reference/reference_r_weighted_turnbull.R) and by its identities:
whole-number weights give the counted estimate, the scale of the weights
changes nothing, ``w = 1`` gives the unweighted estimate, and on right
censored (and left truncated) data the Kaplan-Meier option is the weighted
Kaplan-Meier. Its bounds are the weighted case bootstrap's
(``bootstrap_cb``), seeded and free of the weights' scale; their coverage
is a study (scripts/studies/weighted_turnbull/coverage.py).
"""

import json
import pickle

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import quietly, weighted_reference_data
from surpyval.utils.weights import WeightedRefusal

KM = "Kaplan-Meier"

# The data of reference_r_weighted_turnbull.R: (l, u] intervals, u = inf
# right censored.
L = np.array(
    [6, 3, 6, 2, 6, 0, 0, 1, 4, 17, 10, 10, 0, 11, 2, 1, 7, 2, 7, 7, 14, 18]
    + [0, 6, 22, 0, 12, 0, 1, 2],
    dtype=float,
)
U = np.array(
    [8, np.inf, 11, 6, 10, 6, 3, 7, 8, np.inf, 12, 14, 5, 18, 4, np.inf, 13]
    + [8, 12, 12, 20, np.inf, np.inf, 10, np.inf, np.inf, 15, 8, 8, 6],
    dtype=float,
)
W = np.array(
    [2.25, 1.03, 2.19, 2.25, 0.71, 0.49, 0.83, 2.86, 0.98, 1.52, 1.88]
    + [0.65, 1.61, 2.66, 2.74, 2.81, 2.69, 2.75, 2.02, 2.18, 1.11, 2.4]
    + [2.9, 1.9, 1.69, 1.42, 2.65, 0.44, 1.65, 2.38]
)
# icenReg's Turnbull intervals (l, u] and their masses, from R 4.6.2,
# icenReg 2.0.16 (ic_np, B = c(0, 1), tol = 1e-12)
R_LOWER = np.array([2, 3, 4, 6, 7, 10, 11, 12, 14, 17, 18, 22], dtype=float)
R_UPPER = np.array([3, 4, 5, 7, 8, 11, 12, 13, 15, 18, 20, np.inf])
R_P_WEIGHTED = np.array(
    [0.3204996160475181988, 0, 0, 0, 0.3292425750792612460, 0]
    + [0.1230203352107951953, 0.0060676849164775504, 0.1017933053341916888]
    + [0, 0, 0.1193764834117561208]
)
R_P_UNWEIGHTED = np.array(
    [0.312377026485816112, 0, 0, 0, 0.341049506803491109, 0]
    + [0.135602130867590848, 0, 0.086482935685562395, 0]
    + [0.019002732425694377, 0.105485667731845159]
)


def _interval(w=None, **kwargs):
    return quietly(
        sp.Turnbull.fit, xl=L, xr=U, w=w, turnbull_estimator=KM, **kwargs
    )


@pytest.mark.parametrize(
    "w, p", [(W, R_P_WEIGHTED), (None, R_P_UNWEIGHTED)], ids=["w", "none"]
)
def test_against_icenreg(w, p):
    # The survival at each end of icenReg's Turnbull intervals: 1 minus
    # the mass up to there. R stops at its own tolerance (1e-12 on its
    # criterion), which leaves the masses 3e-10 apart; 1e-8 here.
    model = _interval(w)
    assert model.converged
    finite = np.isfinite(R_UPPER)
    assert_allclose(
        model.sf(R_UPPER[finite]), 1 - np.cumsum(p)[finite], atol=1e-8
    )
    assert_allclose(
        model.sf(R_LOWER), np.r_[1.0, 1 - np.cumsum(p)][:-1], atol=1e-8
    )


@pytest.mark.parametrize("algorithm", ["EMICM", "EM"])
def test_integer_weights_are_counts(algorithm):
    k = np.arange(L.size) % 3 + 1
    weighted = _interval(
        k.astype(float),
        turnbull_algorithm=algorithm,
        tol=1e-12,
        max_iter=10**5,
    )
    counted = quietly(
        sp.Turnbull.fit,
        xl=L,
        xr=U,
        n=k,
        turnbull_estimator=KM,
        turnbull_algorithm=algorithm,
        tol=1e-12,
        max_iter=10**5,
    )
    # To the algorithm's tolerance
    assert_allclose(weighted.R, counted.R, atol=1e-8)
    assert_allclose(weighted.x, counted.x)


@pytest.mark.parametrize(
    "estimator", ["Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington"]
)
@pytest.mark.parametrize("scale", [0.01, 37.0])
def test_scale_of_the_weights_changes_nothing(estimator, scale):
    base = quietly(
        sp.Turnbull.fit, xl=L, xr=U, w=W, turnbull_estimator=estimator
    )
    scaled = quietly(
        sp.Turnbull.fit, xl=L, xr=U, w=W * scale, turnbull_estimator=estimator
    )
    assert_allclose(scaled.R, base.R, atol=1e-10)
    assert_allclose(scaled.r, base.r, rtol=1e-9)


@pytest.mark.parametrize(
    "estimator", ["Kaplan-Meier", "Nelson-Aalen", "Fleming-Harrington"]
)
@pytest.mark.parametrize("algorithm", ["auto", "EM"])
def test_unit_weights_are_the_unweighted_estimate(estimator, algorithm):
    kwargs = dict(turnbull_estimator=estimator, turnbull_algorithm=algorithm)
    weighted = quietly(
        sp.Turnbull.fit, xl=L, xr=U, w=np.ones(L.size), **kwargs
    )
    plain = quietly(sp.Turnbull.fit, xl=L, xr=U, **kwargs)
    assert weighted.weighted and not plain.weighted
    assert_allclose(weighted.R, plain.R, atol=1e-12)


@pytest.mark.parametrize("truncated", [False, True])
def test_right_censored_is_the_weighted_kaplan_meier(truncated):
    data = weighted_reference_data()
    tl = data["tl"] if truncated else None
    args = (data["x"], data["c"])
    turnbull = quietly(
        sp.Turnbull.fit, *args, w=data["w"], tl=tl, turnbull_estimator=KM
    )
    km = sp.KaplanMeier.fit(*args, w=data["w"], tl=tl)
    q = np.linspace(0.1, np.max(data["x"]) - 0.1, 50)
    # To the EM's tolerance
    assert_allclose(turnbull.sf(q), km.sf(q), atol=1e-8)


def test_truncated_scale_and_unit_weights():
    data = weighted_reference_data()
    args = (data["x"], data["c"])
    base = quietly(sp.Turnbull.fit, *args, w=data["w"], tl=data["tl"])
    scaled = quietly(sp.Turnbull.fit, *args, w=data["w"] * 9, tl=data["tl"])
    assert_allclose(scaled.R, base.R, atol=1e-10)
    ones = quietly(
        sp.Turnbull.fit, *args, w=np.ones(len(args[0])), tl=data["tl"]
    )
    plain = quietly(sp.Turnbull.fit, *args, tl=data["tl"])
    assert_allclose(ones.R, plain.R, atol=1e-12)


def test_bootstrap_is_seeded_and_scale_free():
    a = _interval(W)
    b = _interval(W * 7.3)
    q = [3.0, 7.0, 12.0]
    first = a.bootstrap_cb(q, n_boot=40, random_state=5)
    assert np.isfinite(first).all()
    assert (first[:, 0] <= a.sf(q)).all() and (a.sf(q) <= first[:, 1]).all()
    assert_allclose(a.bootstrap_cb(q, n_boot=40, random_state=5), first)
    assert_allclose(
        b.bootstrap_cb(q, n_boot=40, random_state=5), first, atol=1e-8
    )
    # A row with n = 2 is two units: the same resamples as two rows
    rows = np.r_[np.arange(L.size), [0]]
    n = np.ones(L.size, dtype=int)
    n[0] = 2
    counted = _interval(W, n=n)
    written = quietly(
        sp.Turnbull.fit,
        xl=L[rows],
        xr=U[rows],
        w=W[rows],
        turnbull_estimator=KM,
    )
    assert_allclose(counted.R, written.R, atol=1e-10)


def test_bounds_on_the_variance_refuse_and_name_the_bootstrap():
    model = _interval(W)
    calls = {
        "cb": lambda: model.cb([3.0]),
        "quantile_cb": lambda: model.quantile_cb(0.5),
        "rmst": lambda: model.rmst(10.0),
        "sf_diff": lambda: sp.sf_diff(model, _interval(), 5.0),
        "band": lambda: model.band(),
    }
    for key, call in calls.items():
        with pytest.raises(WeightedRefusal) as info:
            call()
        assert "Use" in str(info.value), key
    assert "bootstrap_cb" in str(pytest.raises(WeightedRefusal, model.cb, 3.0))
    # The curve is still drawn, without bounds
    assert model.get_plot_data()["cbs"] is None
    assert "bounds by bootstrap_cb" in repr(model)


def test_the_other_weighted_estimates_still_refuse_bootstrap_cb():
    data = weighted_reference_data()
    km = sp.KaplanMeier.fit(data["x"], data["c"], w=data["w"])
    with pytest.raises(WeightedRefusal, match="cb\\(\\)"):
        km.bootstrap_cb([2.0], random_state=1)


def test_round_trip():
    model = _interval(W)
    d = model.to_dict(with_data=True)
    assert d["schema"] == 4 and d["weighted"] is True
    restored = sp.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    q = [3.0, 7.0]
    for again in (restored, pickle.loads(pickle.dumps(model))):
        assert_allclose(again.sf(q), model.sf(q))
        assert_allclose(
            again.bootstrap_cb(q, n_boot=10, random_state=2),
            model.bootstrap_cb(q, n_boot=10, random_state=2),
        )
        with pytest.raises(WeightedRefusal):
            again.cb(q)


def test_effective_sample_size_and_its_warning():
    model = _interval(W)
    assert_allclose(model.effective_n, W.sum() ** 2 / np.sum(W**2))
    w = np.r_[np.full(L.size - 1, 1.0), 1000.0]
    with pytest.warns(UserWarning, match="effective"):
        sp.Turnbull.fit(xl=L, xr=U, w=w)


def test_weighted_probability_plot_uses_the_weighted_turnbull():
    model = quietly(sp.Weibull.fit, xl=L, xr=U, w=W)
    data = model.get_plot_data()
    scaled = quietly(sp.Weibull.fit, xl=L, xr=U, w=W * 11).get_plot_data()
    assert_allclose(scaled["F"], data["F"], atol=1e-10)
    turnbull = quietly(sp.Turnbull.fit, xl=L, xr=U, w=W).F
    assert np.isin(np.round(data["F"], 10), np.round(turnbull, 10)).all()
    # The rank heuristics still refuse
    with pytest.raises(WeightedRefusal, match="Turnbull"):
        model.get_plot_data(heuristic="Blom")
