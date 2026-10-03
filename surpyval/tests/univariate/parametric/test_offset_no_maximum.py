"""An offset fit whose offset runs onto the first failure has no maximum
(#487).

As the offset ``gamma`` approaches the smallest exact observation, that
observation's density term is the base density at its origin. Where that is
infinite (a Weibull, Gamma or LogLogistic shape below 1), or 0 and reached
only along a path on which the likelihood grows without bound (the
LogNormal), the likelihood has no finite maximum. The fit warns "No finite
maximum" once, returns where its search stopped, and recommends maximum
product of spacings; a density finite at its origin (the Exponential) has a
genuine maximum at ``gamma = x(1)`` and does not warn.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.utils.surpyval_data import SurpyvalData

NO_MAXIMUM = "No finite maximum: the offset gamma"
# The issue's sample: the 3-parameter Weibull profile likelihood rises all
# the way to gamma = 55.
X = [55, 60, 70, 80, 95, 120, 140]


def _fit(dist, x=X, **kwargs):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = dist.fit(x, offset=True, **kwargs)
    return model, rec


@pytest.mark.parametrize("dist", [sp.Weibull, sp.Gamma, sp.LogLogistic])
def test_an_offset_run_onto_the_first_failure_warns_once(dist):
    # Before: the Weibull warned "MLE Failed" and returned its start, the
    # Gamma "Precision was lost" at gamma = 55 with shape 0.027, and the
    # LogLogistic "MLE Failed" -- none said the likelihood is unbounded.
    model, rec = _fit(dist)
    assert len(rec) == 1, [str(w.message)[:60] for w in rec]
    message = str(rec[0].message)
    assert message.startswith(NO_MAXIMUM)
    assert "how='MPS'" in message
    assert rec[0].filename == __file__
    # the answer is where the search stopped: on the first failure
    assert model.gamma == pytest.approx(55.0, abs=1e-5)
    assert model.gamma <= 55.0
    # with the density infinite at the origin: a shape below 1
    assert dist.df(np.array([0.0]), *model.params)[0] == np.inf


def test_the_weibull_profile_likelihood_rises_to_the_first_failure():
    # The profile over gamma has no interior maximum on this sample.
    data = np.asarray(X, dtype=float)
    profile = [
        -sp.Weibull.fit(data - g)._neg_ll for g in (0, 40, 54, 54.99, 54.9999)
    ]
    assert np.all(np.diff(profile) > 0)


def test_no_raw_numpy_warning_escapes():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always", UserWarning)
            sp.Weibull.fit(X, offset=True)
    assert [str(w.message)[:18] for w in rec] == ["No finite maximum:"]


def test_mps_has_an_interior_offset():
    model, rec = _fit(sp.Weibull, how="MPS")
    assert not [w for w in rec if "finite maximum" in str(w.message)]
    assert model.gamma < 55.0 - 1.0


def test_the_exponential_boundary_maximum_does_not_warn():
    # The two-parameter Exponential's MLE is gamma = x(1): its density is
    # finite at the origin, so the likelihood is bounded there.
    model, rec = _fit(sp.Exponential)
    assert not [w for w in rec if "finite maximum" in str(w.message)]
    assert model.gamma == pytest.approx(55.0, abs=1e-4)


@pytest.mark.parametrize(
    "dist", [sp.Weibull, sp.Gamma, sp.LogNormal, sp.LogLogistic]
)
def test_an_interior_offset_fit_does_not_warn(dist):
    rng = np.random.default_rng(3)
    x = 40.0 + sp.Weibull.random(300, 60.0, 2.5, random_state=rng)
    model, rec = _fit(dist, x=x)
    assert not [w for w in rec if "finite maximum" in str(w.message)]
    assert model.gamma < x.min()


def test_a_lognormal_run_into_the_corner_warns():
    # sigma runs up as gamma closes on the first failure (Hill, 1963): the
    # LogNormal density is 0 at its origin, so the fit only gets there
    # along a path on which the likelihood grows without bound.
    fitter: object = sp.LogNormal
    data = SurpyvalData(np.asarray(X, dtype=float))
    results = {"params": np.array([-0.1, 12.4]), "gamma": 55.0 - 1e-12}
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        flagged = fitter._warn_if_offset_at_limit(
            data, results, {}, False, False
        )
    assert flagged
    assert "density is zero at its origin" in str(rec[0].message)


# #599: the conformance fixture shifted by 5, with an observation at -1
# below the rest (9 to 22): a long left tail, which no offset LogNormal or
# Gamma has.
X_599 = np.array(
    [-1.0, 8.84, 9.956, 10.953, 11.903, 12.846, 13.816, 14.85, 15.997]
    + [17.347, 19.096, 21.954]
)
C_599 = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0])
N_599 = np.array([1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1])


@pytest.mark.parametrize("dist", [sp.LogNormal, sp.Gamma])
def test_599_an_offset_running_to_the_normal_limit_warns_once(
    dist, monkeypatch
):
    # The offset runs down towards -inf, the shape making up for it, and
    # the likelihood rises towards the Normal's (42.2653) only as
    # 1 / |gamma|: it never looked flat to the verification, every rung
    # of the ladder ran, and the fit ended "unverified" after 4-17 s.
    # Now the first rung that stops short of a maximum (or BFGS's watch
    # of its iterates) sees the Normal fit the data at least as well,
    # and the fit ends there.
    from surpyval.univariate.parametric.fitters import mle

    rungs = []
    run_rung = mle._run_rung

    def counted(fun, method, *args, **kwargs):
        rungs.append(method)
        return run_rung(fun, method, *args, **kwargs)

    monkeypatch.setattr(mle, "_run_rung", counted)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        model = dist.fit(X_599, C_599, N_599, offset=True)
    assert len(rec) == 1, [str(w.message)[:60] for w in rec]
    message = str(rec[0].message)
    assert message.startswith("No finite maximum: the " + dist.name)
    assert "approaches a Normal distribution" in message
    assert "surpyval.Normal" in message
    assert rec[0].filename == __file__
    assert model.maximum == "no finite maximum"
    assert set(rungs) == {"BFGS"}
    # On the way to the limit: below where it started, and no better
    # than the Normal
    assert model.gamma < -10
    normal = sp.Normal.fit(X_599, C_599, N_599)
    assert model.neg_ll() >= normal.neg_ll()


def test_599_the_profile_likelihood_falls_to_the_normal_limit():
    # The LogNormal profile over the offset rises all the way to the
    # Normal's likelihood as gamma -> -inf: no finite maximum.
    profile = [
        sp.LogNormal.fit(X_599 - g, C_599, N_599).neg_ll()
        for g in (-1.5, -10, -100, -1e3, -1e4)
    ]
    assert np.all(np.diff(profile) < 0)
    normal = sp.Normal.fit(X_599, C_599, N_599).neg_ll()
    assert profile[-1] - normal == pytest.approx(0, abs=1e-2)
    assert np.all(np.asarray(profile) > normal)
