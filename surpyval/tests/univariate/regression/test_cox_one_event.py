"""A Cox fit to one event: separated data with no finite maximum.

With a single event the partial likelihood increases without bound
along a direction of the coefficients that ranks the event's row above
every other at-risk row. The fit should say so (``maximum == "no
finite maximum"``, nan standard errors, one warning) and return, as the
parametric families do and as ``center=True`` does. With two continuous
covariates at ``center=False`` the runaway coefficients make
``exp(beta'Z)`` overflow on the rows as given, so the baseline is kept
at the covariate means, with a warning, rather than moved to ``Z = 0``.
"""

import warnings

import numpy as np
import pytest

import surpyval
from surpyval import CoxPH

NO_MAX = "No finite maximum"
KEPT = "The baseline is kept at the covariate means"


def _data():
    rng = np.random.default_rng(1)
    x = rng.weibull(2.0, 30) * 100
    Z = rng.normal(size=(30, 2))
    c = np.r_[0, np.ones(29)]  # one event, x = 103.587..., rank 19 of 30
    return x, Z, c


def _messages(record):
    return [str(w.message) for w in record]


@pytest.mark.parametrize("tie_method", ["efron", "breslow"])
def test_one_event_two_covariates_returns(tie_method):
    """The coefficients run off, so the baseline cannot be moved to Z = 0
    (``beta'Z`` reaches about -1047 on the rows as given): it is kept at
    the covariate means, as with ``center=True``, with one warning saying
    so, and the model is the centred fit's. It used to warn and then
    raise from the baseline move.
    """
    x, Z, c = _data()
    with pytest.warns(UserWarning, match=NO_MAX):
        m = CoxPH.fit(x, Z, c, tie_method=tie_method)
    assert m.maximum == "no finite maximum"
    assert np.all(np.isnan(np.diag(m.covariance())))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        centred = CoxPH.fit(x, Z, c, tie_method=tie_method, center=True)
    np.testing.assert_array_equal(m.center, centred.center)
    np.testing.assert_array_equal(m.params, centred.params)
    np.testing.assert_array_equal(m.h0, centred.h0)
    q = [50.0, 100.0, 150.0]
    np.testing.assert_array_equal(m.sf(q, Z[:3]), centred.sf(q, Z[:3]))


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"w": np.full(30, 1.5)}],
    ids=["unweighted", "weighted"],
)
def test_one_event_kept_at_centre_warns_once(kwargs):
    x, Z, c = _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m = CoxPH.fit(x, Z, c, **kwargs)
    messages = _messages(record)
    assert sum(s.startswith(NO_MAX) for s in messages) == 1
    kept = [s for s in messages if s.startswith(KEPT)]
    assert len(kept) == 1
    assert "center=True" in kept[0] and "penalised" in kept[0]
    assert m.maximum == "no finite maximum"
    np.testing.assert_array_equal(m.center, m._fit_center)
    # It survives a round trip with its centre.
    back = surpyval.from_dict(m.to_dict())
    np.testing.assert_array_equal(back.center, m.center)


def test_one_event_stratified_moves_when_it_can():
    # Stratified, the runaway coefficients stay small enough for the move
    # to Z = 0: the baselines are moved and nothing extra is said.
    x, Z, c = _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m = CoxPH.fit(x, Z, c, strata=np.arange(30) % 2)
    assert m.maximum == "no finite maximum"
    np.testing.assert_array_equal(m.center, [0.0, 0.0])
    assert not any(s.startswith(KEPT) for s in _messages(record))


def test_one_event_two_covariates_centred_returns():
    x, Z, c = _data()
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        m = CoxPH.fit(x, Z, c, center=True)
    assert m.maximum == "no finite maximum"
    assert sum(s.startswith(NO_MAX) for s in _messages(record)) == 1
    assert np.all(np.isnan(np.diag(m.covariance())))


def test_one_event_binary_covariate_returns():
    x, Z, c = _data()
    with pytest.warns(UserWarning, match=NO_MAX):
        m = CoxPH.fit(x, (Z[:, :1] > 0).astype(float), c)
    assert m.maximum == "no finite maximum"
    # Observed -35.02 at e3412770: running off towards -inf.
    assert m.params[0] < -20
    assert np.all(np.isnan(np.diag(m.covariance())))


def test_one_event_one_covariate_has_a_maximum():
    x, Z, c = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = CoxPH.fit(x, Z[:, :1], c)
    assert m.maximum == "verified"
    # Observed at e3412770, pinned as a regression value.
    assert m.params[0] == pytest.approx(-1.97905, rel=1e-4)
