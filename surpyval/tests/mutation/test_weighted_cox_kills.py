"""Checks that kill mutants of the weighted Cox fit (``CoxPH.fit(...,
w=w)``, ``proportional_hazards/_weighted_cox.py``) that the suite let
survive (#396).

Each test names the mutants it kills: the function, then the edit. The
values are closed forms, R's (``survival``'s ``coxph`` with ``weights``
and ``robust = TRUE``, as stored in ``test_cox_weighted.py``) or the
unweighted fit, which a weighted fit with unit weights must reproduce.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.stats import norm

from surpyval import CoxPH
from surpyval.tests._helpers import weighted_reference_data

# R 4.5.3, survival 3.8.12: coxph(Surv(x, d) ~ z1 + z2, weights = w,
# robust = TRUE) (Efron), as stored in reference/test_weights.py
R_COEF = np.array([1.06791948026886, 1.01251989144677])
R_SE = np.array([0.472898791484031, 0.239872432814770])


@pytest.fixture
def data():
    return weighted_reference_data()


def _fit(data, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return CoxPH.fit(data["x"], data["Z"], c=data["c"], **kwargs)


def test_p_values_are_the_robust_wald_tests(data):
    """Kills, in ``fit_weighted_cox``, the p-value made ``3 * sf``, ``2 /
    sf``, ``sf(|beta * se|)``, ``None`` and ``expand(None, ...)``: no test
    read a weighted fit's p-values. R's ``Pr(>|z|)`` is the two-sided
    normal tail of ``coef / robust se``."""
    m = _fit(data, w=data["w"])
    assert_allclose(m.p_values, 2 * norm.sf(np.abs(R_COEF / R_SE)), rtol=1e-6)


def test_a_converged_fit_is_a_verified_maximum(data):
    """Kills, in ``fit_weighted_cox``, ``"verified" if converged and
    False``: every weighted fit said ``"unverified"`` (without the
    warning that goes with it)."""
    m = _fit(data, w=data["w"])
    assert m.maximum == "verified"


def test_a_fit_stopped_short_is_unverified(data):
    """Kills, in ``fit_weighted_cox``, ``"verified" if converged or
    True``, and in ``newton`` the iteration limit edited (101, ``range``
    ends): a search stopped at its limit of 100 Newton steps (here by a
    tolerance no step meets) says so, once, and is not a verified
    maximum."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = CoxPH.fit(data["x"], data["Z"], c=data["c"], w=data["w"], tol=-1)
    said = [str(w.message) for w in caught]
    assert len(said) == 1 and "did not converge in 100 Newton steps" in said[0]
    assert m.maximum == "unverified"


def test_effective_sample_size_is_kish(data):
    """Kills, in ``fit_weighted_cox``, ``effective_n`` set to ``None`` or
    computed without the counts or the weights: Kish's ``(sum n w)**2 /
    sum n w**2``, here with counts of 2 on every other row."""
    n = np.where(np.arange(40) % 2 == 0, 2.0, 1.0)
    w = np.asarray(data["w"], dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = CoxPH.fit(data["x"], data["Z"], c=data["c"], n=n, w=w)
    assert_allclose(m.effective_n, (n * w).sum() ** 2 / (n * w**2).sum())


def test_one_dimensional_covariate(data):
    """Kills, in ``fit_weighted_cox``, the reshape of a 1-d ``Z`` to one
    column made ``(-2, 1)``, ``(-1, 2)``, ``(+1, 1)``, ``(1)`` or
    ``(None, 1)``: no test gave a weighted fit a 1-d ``Z``. It is one
    covariate, the same fit as the column."""
    z = np.asarray(data["Z"], dtype=float)[:, 1]
    flat = _fit({**data, "Z": z}, w=data["w"])
    column = _fit({**data, "Z": z[:, None]}, w=data["w"])
    assert_allclose(flat.params, column.params, rtol=1e-10)
    assert_allclose(
        flat.standard_errors(), column.standard_errors(), rtol=1e-10
    )


def test_breslow_with_an_aliased_column(data):
    """Kills, in ``WeightedCox.columns``, the copy without the aliased
    columns made Efron whatever the fit's ties (``efron if ... or True``,
    ``"XXbreslowXX"``, ``"BRESLOW"``): a Breslow fit with an aliased
    column was fitted by Efron's likelihood. The data have tied deaths, so
    the two differ; the fit must be the Breslow fit of the other
    columns."""
    Z = np.asarray(data["Z"], dtype=float)
    x = np.round(np.asarray(data["x"], dtype=float))  # ties
    with_constant = np.column_stack([Z, np.ones(len(x))])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the aliased column's warning
        m = CoxPH.fit(
            x, with_constant, c=data["c"], w=data["w"], tie_method="breslow"
        )
    ref = _fit({**data, "x": x}, w=data["w"], tie_method="breslow")
    efron = _fit({**data, "x": x}, w=data["w"], tie_method="efron")
    assert not np.allclose(ref.params, efron.params, rtol=1e-4)
    assert_allclose(m.params[:2], ref.params, rtol=1e-8)
    assert np.isnan(m.params[2])


def test_every_column_aliased():
    """Kills, in ``fit_weighted_cox``, the branch where every column is
    aliased (nothing to estimate): its log-likelihood taken at
    ``zeros(1)``, as the score (``[1]``) or ``None``, ``beta_k`` of
    ``zeros(1)``, and the empty covariance ``(1, 0)`` or ``(0, 1)``. No
    test fitted a weighted model with nothing to estimate.

    With untied deaths, no entry and ``beta`` irrelevant, the weighted
    partial log-likelihood is ``-sum_j w_j log(sum_{x_i >= x_j} w_i)``:
    deaths at 1, 2 and 4 (weights 2, 1, 3), a censoring at 3 (weight
    0.5); risk sets {all}, {2, 3, 4}, {4}.
    """
    x = np.array([1.0, 2.0, 3.0, 4.0])
    c = np.array([0, 0, 1, 0])
    w = np.array([2.0, 1.0, 0.5, 3.0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # aliased, few effective units
        m = CoxPH.fit(x, np.ones(4), c=c, w=w)
    ll = -(2 * np.log(6.5) + 1 * np.log(4.5) + 3 * np.log(3.0))
    assert np.isnan(m.params).all()
    assert_allclose(m.neg_ll(), -ll, rtol=1e-12)
    assert m.maximum == "verified"


def test_weighted_round_trip_keeps_ties_and_entry(data):
    """Kills, in ``fit_weighted_cox``, ``model.tie_method = None`` and
    ``model.tl`` set to ``None`` (or without ``dtype``): ``to_dict`` saves
    both, and the restored model lost Breslow's ties and the entry
    times."""
    tl = np.asarray(data["tl"], dtype=float)
    m = _fit(data, w=data["w"], tl=tl, tie_method="breslow")
    saved = m.to_dict()
    assert saved["tie_method"] == "breslow"
    assert_allclose(saved["tl"], tl)


def test_ties_at_the_last_death_time_under_efron():
    """Kills, in ``WeightedCox._score_residuals``, the clamp of each row's
    own event time made ``k - 2``: a death at the last event time was
    then taken as at risk there but not dying, which changes its score
    residual only when deaths tie there under Efron (the Efron share
    ``1 - k/m``). The weighted score residuals sum to the score, 0 at the
    fit, so the robust covariance is a sum of zero-mean terms; checked
    here against the Efron fit with unit weights, whose robust variance
    the unweighted fit gives (``robust_summary``)."""
    x = np.array([1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 6.0, 6.0])
    c = np.array([0, 0, 1, 0, 0, 1, 0, 0, 0])
    z = np.array([0.3, -1.2, 0.8, 1.5, -0.4, 0.1, 0.9, -0.7, 0.2])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # few effective units
        weighted = CoxPH.fit(x, z, c=c, w=np.ones(9))
    plain = CoxPH.fit(x, z, c=c)
    assert_allclose(weighted.params, plain.params, rtol=1e-8)
    assert_allclose(
        weighted.standard_errors(), plain.robust_summary()["se"], rtol=1e-8
    )
