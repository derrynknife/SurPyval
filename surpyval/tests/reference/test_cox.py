"""SurPyval's Cox model against stored R ``coxph`` results (#379).

Covers the Breslow and Efron tie methods, R's ``ties="exact"`` (the
discrete, conditional-logistic likelihood: SurPyval's
``method="kalbfleisch-prentice"``), strata, left truncation and start-stop
data, with scikit-survival as a second reference for the tie methods.

Tolerances: both programs solve the score equations to convergence (R
stops when the log-likelihood changes by less than 1e-9 relatively,
SurPyval's Newton-Raphson when a step is below ``tol=1e-10`` standard
errors), so the coefficients agree to
about 1e-9 on these data; ``atol=1e-7`` leaves room for that and nothing
else. The standard errors and the Breslow baseline are smooth functions of
the coefficients and agree to the same order (``rtol=1e-6``).
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

COEF = dict(rtol=0, atol=1e-7)
DERIVED = dict(rtol=1e-6, atol=1e-10)


def _se(model):
    """Model-based standard errors, the inverse observed information."""
    info = model.jac(model.params)[1]
    return np.sqrt(np.diag(np.linalg.inv(info)))


def _check(model, ref, p):
    assert_allclose(model.params, np.atleast_1d(ref["coef"]), **COEF)
    assert_allclose(_se(model), np.atleast_1d(ref["se"]), **DERIVED)
    # coxph's loglik is (null, fitted); the null is the same function at 0.
    assert_allclose(-model.neg_ll_of(model.params), ref["loglik"][1], **COEF)
    assert_allclose(-model.neg_ll_of(np.zeros(p)), ref["loglik"][0], **COEF)
    # logLik(fit) and AIC(fit): the partial likelihood, k the coefficients
    # (#604)
    assert_allclose(model.log_likelihood, ref["loglik"][1], **COEF)
    assert_allclose(model.aic(), 2 * p - 2 * ref["loglik"][1], **COEF)
    # The Breslow estimator of the uncentred baseline (survfit.coxph with
    # ctype = 1 at covariates 0), which is what SurPyval reports for every
    # tie method.
    t = ref["baseline_time"]
    assert_allclose(
        model.Hf(t, np.zeros(p)), ref["baseline_cumhaz"], **DERIVED
    )


def _lung(drop_missing=True):
    d = fixture("lung")
    Z = np.column_stack([d["age"], d["sex"], d["ph_ecog"]])
    keep = ~np.isnan(d["ph_ecog"]) if drop_missing else slice(None)
    return d["time"][keep], Z[keep], d["c"][keep]


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_lung(ties):
    x, Z, c = _lung()
    model = sp.CoxPH.fit(x, Z, c=c, tie_method=ties)
    _check(model, values("r_survival", "cox_lung_" + ties), 3)


def test_cox_lung_missing_covariate_row_is_dropped_like_na_omit():
    # One patient has ph.ecog missing. R drops the row (na.omit); SurPyval
    # drops it with a warning, and the fit must be the same.
    x, Z, c = _lung(drop_missing=False)
    with pytest.warns(UserWarning):
        model = sp.CoxPH.fit(x, Z, c=c, tie_method="efron")
    _check(model, values("r_survival", "cox_lung_efron"), 3)


@pytest.mark.parametrize(
    "ties, method",
    [
        ("breslow", "breslow"),
        ("efron", "efron"),
        # R's "exact" is the discrete (Kalbfleisch-Prentice, conditional
        # logistic) likelihood. SurPyval's "exact" is the continuous-time
        # average over orderings, which R does not implement.
        ("exact", "kalbfleisch-prentice"),
    ],
)
def test_cox_ties(ties, method):
    d = fixture("ties")
    Z = np.column_stack([d["z1"], d["z2"]])
    model = sp.CoxPH.fit(d["x"], Z, c=d["c"], tie_method=method)
    _check(model, values("r_survival", "cox_ties_" + ties), 2)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_ties_matches_scikit_survival(ties):
    d = fixture("ties")
    Z = np.column_stack([d["z1"], d["z2"]])
    model = sp.CoxPH.fit(d["x"], Z, c=d["c"], tie_method=ties)
    ref = values("py_sksurv", "cox_ties_" + ties)
    assert_allclose(model.params, ref["coef"], **COEF)


def test_cox_exact_ties_differ_from_discrete_likelihood():
    # A reminder, with numbers, that SurPyval's "exact" is not R's: on the
    # tied fixture R's exact coefficients are (0.4498, 0.8578) and the
    # average-over-orderings likelihood gives (0.4253, 0.8075).
    d = fixture("ties")
    Z = np.column_stack([d["z1"], d["z2"]])
    model = sp.CoxPH.fit(d["x"], Z, c=d["c"], tie_method="exact")
    ref = values("r_survival", "cox_ties_exact")
    assert np.max(np.abs(model.params - ref["coef"])) > 1e-2


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_left_truncation(ties):
    d = fixture("left_truncation")
    model = sp.CoxPH.fit(
        d["x"], d["z"][:, None], c=d["c"], tl=d["tl"], tie_method=ties
    )
    _check(model, values("r_survival", "cox_left_truncation_" + ties), 1)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_start_stop_heart(ties):
    # Start-stop rows are left-truncated rows to the partial likelihood.
    d = fixture("heart")
    Z = np.column_stack([d["age"], d["year"], d["surgery"], d["transplant"]])
    model = sp.CoxPH.fit(
        d["stop"], Z, c=d["c"], tl=d["start"], tie_method=ties
    )
    _check(model, values("r_survival", "cox_heart_" + ties), 4)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_stratified(ties):
    d = fixture("lung")
    keep = ~np.isnan(d["ph_ecog"])
    Z = np.column_stack([d["age"], d["ph_ecog"]])[keep]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = sp.CoxPH.fit(
            d["time"][keep],
            Z,
            c=d["c"][keep],
            tie_method=ties,
            strata=d["sex"][keep],
        )
    ref = values("r_survival", "cox_lung_strata_" + ties)
    assert_allclose(model.params, ref["coef"], **COEF)
    assert_allclose(_se(model), ref["se"], **DERIVED)
    for sex in (1, 2):
        t = ref["baseline_time_sex{}".format(sex)]
        assert_allclose(
            model.Hf(t, np.zeros(2), stratum=float(sex)),
            ref["baseline_cumhaz_sex{}".format(sex)],
            **DERIVED,
        )
