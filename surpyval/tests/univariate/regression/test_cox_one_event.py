"""A Cox fit to one event: separated data with no finite maximum.

With a single event the partial likelihood increases without bound
along a direction of the coefficients that ranks the event's row above
every other at-risk row. The fit should say so (``maximum == "no
finite maximum"``, nan standard errors, one warning) and return, as the
parametric families do and as ``center=True`` already does. With two
continuous covariates at ``center=False`` it warns and then raises from
the move of the baseline to ``Z = 0``
(``_fit_skeleton.py::baseline_at_origin_error``).
"""

import warnings

import numpy as np
import pytest

from surpyval import CoxPH

NO_MAX = "No finite maximum"


def _data():
    rng = np.random.default_rng(1)
    x = rng.weibull(2.0, 30) * 100
    Z = rng.normal(size=(30, 2))
    c = np.r_[0, np.ones(29)]  # one event, x = 103.587..., rank 19 of 30
    return x, Z, c


def _messages(record):
    return [str(w.message) for w in record]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "a separated Cox fit with 2 continuous covariates raises from the "
        "baseline move after warning 'No finite maximum'; with "
        "center=True it returns"
    ),
)
@pytest.mark.parametrize("tie_method", ["efron", "breslow"])
def test_one_event_two_covariates_returns(tie_method):
    """Observed at e3412770: ``CoxPH.fit(x, Z, c)`` warns "No finite
    maximum: the partial likelihood keeps increasing as coefficients
    [0, 1] grow without bound together" and then raises ``ValueError:
    The baseline hazard at Z = 0 cannot be represented for these
    covariates ... (beta'Z reaches -1047 ...)``. ``tie_method="breslow"``
    raises the same way. Expected: a model with ``maximum == "no finite
    maximum"`` and nan standard errors, as with ``center=True``.
    """
    x, Z, c = _data()
    with pytest.warns(UserWarning, match=NO_MAX):
        m = CoxPH.fit(x, Z, c, tie_method=tie_method)
    assert m.maximum == "no finite maximum"
    assert np.all(np.isnan(np.diag(m.covariance())))


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
