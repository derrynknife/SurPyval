"""The weighted Cox fit (``CoxPH.fit(..., w=w)``) where it meets the rest
of the Cox model: the robust summaries, the likelihood-ratio test, the
residuals and diagnostics, the baseline at ``Z = 0``, aliased columns and
the saved baseline method. Each used to take the weighted fit for an
unweighted one, or skip what the unweighted fit does. (The fit itself is
checked against R in ``reference/test_weights.py``.)"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp
from surpyval.diagnostics import diagnose as registry_diagnose
from surpyval.tests._helpers import weighted_reference_data
from surpyval.univariate.regression.proportional_hazards import diagnostics
from surpyval.utils.weights import WeightedRefusal

# R 4.5.3, survival 3.8.12: coxph(Surv(x, d) ~ z1 + z2, weights = w,
# robust = TRUE) (Efron), as stored in reference/test_weights.py
R_COEF = [1.06791948026886, 1.01251989144677]
R_SE = [0.472898791484031, 0.239872432814770]


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


@pytest.fixture(scope="module")
def model(data):
    return sp.CoxPH.fit(data["x"], data["Z"], c=data["c"], w=data["w"])


# -- A, B: confirmed on dev (fixed with the weights-interaction fixes) -------


def test_robust_summary_is_the_weighted_sandwich(model):
    assert_allclose(model.summary(robust=True)["se(coef)"], R_SE, rtol=1e-6)
    assert_allclose(model.robust_summary()["se"], R_SE, rtol=1e-6)
    assert_allclose(diagnostics.robust_summary(model)["se"], R_SE, rtol=1e-6)


def test_likelihood_ratio_test_refuses(model):
    with pytest.raises(WeightedRefusal, match="not chi-squared"):
        model.likelihood_ratio_test()


# -- C: every diagnostic path refuses ----------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda m: m.diagnose(),
        lambda m: registry_diagnose(m),
        lambda m: diagnostics.check_ph(m),
        lambda m: diagnostics.zph_score_test(m),
        lambda m: diagnostics.compute_residuals(m, "martingale"),
        lambda m: diagnostics.compute_residuals(m, "dfbeta"),
        lambda m: diagnostics.added_covariate_score_test(m, np.ones(40)),
        lambda m: diagnostics.robust_covariance(m),
        lambda m: diagnostics.cox_fit_data(m),
        lambda m: m.compute_residuals("schoenfeld"),
        lambda m: m.check_ph(),
    ],
    ids=[
        "diagnose",
        "registry.diagnose",
        "check_ph",
        "zph_score_test",
        "martingale",
        "dfbeta",
        "added_covariate",
        "module.robust_covariance",
        "cox_fit_data",
        "method.residuals",
        "method.check_ph",
    ],
)
@pytest.mark.filterwarnings("ignore:check_ph is deprecated:DeprecationWarning")
def test_diagnostics_refuse_a_weighted_fit(model, call):
    # diagnose() and the module's functions used to diagnose the weighted
    # fit as if its rows were unweighted units (the residuals of the
    # unweighted data at the weighted coefficients), with no sign of it.
    with pytest.raises(WeightedRefusal) as err:
        call(model)
    message = str(err.value)
    assert "weights (w)" in message
    assert "robust" in message and "covariance" in message


def test_unweighted_diagnostics_still_run(data):
    model = sp.CoxPH.fit(data["x"], data["Z"], c=data["c"])
    assert model.diagnose() is not None
    assert diagnostics.compute_residuals(model, "martingale").shape == (40,)


# -- D: the baseline moved to Z = 0 as the unweighted fit moves it ----------


@pytest.mark.parametrize("weighted", [False, True])
def test_far_covariates_refuse_the_move_to_zero(data, weighted):
    # A covariate far from 0 (a calendar year): the baseline at Z = 0 is
    # exp(-beta' mean) times the one at the mean, which underflows. The
    # weighted fit used to store h0 = 0 and give sf = 1 at every time and
    # covariate, with no warning; both fits now refuse, pointing to
    # center=True.
    Z = np.column_stack([data["Z"][:, 0], 2000 + data["Z"][:, 1]])
    w = data["w"] if weighted else None
    with pytest.raises(ValueError, match="center=True"):
        sp.CoxPH.fit(data["x"], Z, c=data["c"], w=w)
    centred = sp.CoxPH.fit(data["x"], Z, c=data["c"], w=w, center=True)
    sf = centred.sf(np.array([2.0, 6.0]), np.array([0.0, 2000.0]))
    assert np.all((sf > 0) & (sf < 1))


def test_weighted_baseline_moves_like_the_unweighted_one(data):
    # Weights of one are the unweighted fit: its baseline and risk
    # weights at Z = 0, moved from the centre the same way.
    Z = data["Z"] + np.array([3.0, -2.0])
    ones = np.ones(len(data["x"]))
    plain = sp.CoxPH.fit(data["x"], Z, c=data["c"])
    weighted = sp.CoxPH.fit(data["x"], Z, c=data["c"], w=ones)
    assert_allclose(weighted.params, plain.params, rtol=1e-9)
    # The unweighted baseline has a step (of 0) at every time, the
    # weighted one at the event times
    at = plain.d > 0
    assert_array_equal(weighted.x, plain.x[at])
    assert_allclose(weighted.h0, plain.h0[at], rtol=1e-9)
    assert_allclose(weighted.r, plain.r[at], rtol=1e-9)
    assert_allclose(weighted.d, plain.d[at], rtol=0)
    t = np.array([1.0, 3.3, 8.0])
    assert_allclose(weighted.sf(t, Z[:3]), plain.sf(t, Z[:3]), rtol=1e-9)
    # Breslow's increment is the deaths over the risk weight
    breslow = sp.CoxPH.fit(
        data["x"], Z, c=data["c"], w=data["w"], tie_method="breslow"
    )
    assert_allclose(breslow.h0, breslow.d / breslow.r, rtol=1e-12)


# -- E: aliased columns, as the unweighted fit aliases them -----------------


@pytest.mark.parametrize(
    "extra",
    [
        lambda Z: Z[:, 0] + Z[:, 1],  # a combination of the others
        # singular only to rounding: np.linalg.inv does not fail on it
        lambda Z: 1e-3 * Z[:, 0] + 7.1 * Z[:, 1],
        lambda Z: np.full(Z.shape[0], 5.0),  # constant
    ],
    ids=["sum", "combination", "constant"],
)
@pytest.mark.parametrize("weighted", [False, True])
def test_aliased_column_is_nan(data, extra, weighted):
    # The weighted fit did not check for collinear columns: with an
    # information singular to rounding it refused ("drop it and fit
    # again"), and with one that np.linalg.inv could invert it returned
    # nonsense (coefficients 1.06, -83.0 and 11.8 for the second case,
    # with no warning). Now, as the unweighted fit (and R's coxph, with
    # weights or without), the column is aliased: nan, with a warning,
    # and the others are the fit without it.
    Z = np.column_stack([data["Z"], extra(data["Z"])])
    w = data["w"] if weighted else None
    with pytest.warns(UserWarning, match=r"column\(s\) 2 of Z"):
        model = sp.CoxPH.fit(data["x"], Z, c=data["c"], w=w)
    without = sp.CoxPH.fit(data["x"], data["Z"], c=data["c"], w=w)
    assert np.isnan(model.params[2]) and np.isnan(model.p_values[2])
    assert np.isnan(model.standard_errors()[2])
    assert_allclose(model.params[:2], without.params, rtol=1e-9)
    assert_allclose(
        model.standard_errors()[:2], without.standard_errors(), rtol=1e-7
    )
    assert_allclose(
        model.sf(np.array([2.0, 6.0]), Z[:2]),
        without.sf(np.array([2.0, 6.0]), data["Z"][:2]),
        rtol=1e-9,
    )
    if not weighted:
        return
    # R's coxph(..., weights = w, robust = TRUE): NA for the column, the
    # others as without it
    assert_allclose(model.params[:2], R_COEF, rtol=0, atol=1e-7)
    assert_allclose(model.standard_errors()[:2], R_SE, rtol=1e-6)
    robust = model.robust_covariance()
    assert np.isnan(robust[2]).all() and np.isnan(robust[:, 2]).all()
    assert_allclose(robust[:2, :2], without.covariance(), rtol=1e-7)
    cluster = np.arange(len(data["x"])) // 4
    assert_allclose(
        model.robust_covariance(cluster)[:2, :2],
        without.robust_covariance(cluster),
        rtol=1e-7,
    )
    table = model.summary(robust=True)
    assert np.isnan(table["se(coef)"].iloc[2])


def test_weighted_fit_without_aliasing_warns_nothing(data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.CoxPH.fit(data["x"], data["Z"], c=data["c"], w=data["w"])


# -- F: the baseline method's name --------------------------------------------


@pytest.mark.parametrize("ties", ["efron", "breslow"])
def test_baseline_method_matches_the_unweighted_fit(data, ties):
    plain = sp.CoxPH.fit(data["x"], data["Z"], c=data["c"], tie_method=ties)
    weighted = sp.CoxPH.fit(
        data["x"], data["Z"], c=data["c"], w=data["w"], tie_method=ties
    )
    assert weighted.baseline_method == plain.baseline_method == ties
    restored = type(weighted).from_dict(weighted.to_dict())
    assert restored.baseline_method == ties
    assert_array_equal(restored.h0, weighted.h0)
