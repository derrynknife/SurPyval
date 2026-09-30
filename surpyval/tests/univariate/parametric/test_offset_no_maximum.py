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
        flagged = fitter._warn_if_offset_at_limit(  # type: ignore
            data, results, {}, False, False
        )
    assert flagged
    assert "density is zero at its origin" in str(rec[0].message)
