"""Cox fits to case-cohort and nested case-control samples
(``CoxPH.fit_case_cohort``, ``CoxPH.fit_nested_case_control``).

Checked against R's ``survival`` 3.8.12 (``cch``, ``coxph`` with
Barlow's weights, ``clogit``) on the Wilms tumour cohort ``nwtco`` and
``infert``, stored by ``scripts/reference/reference_r_case_cohort.R`` in
``tests/reference/data/case_cohort_r.json``; and against the ordinary Cox
fit, to which every case-cohort estimator reduces when the subcohort is
the whole cohort.

The JSON embeds R's ``survival::nwtco`` (the National Wilms Tumor Study;
Breslow and Chatterjee 1999; ``survival`` is LGPL), whole, because the
whole-cohort test needs every child, and ``datasets::infert``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from surpyval import CoxPH
from surpyval.univariate.regression.proportional_hazards.case_cohort import (
    CaseCohortModel,
    NestedCaseControlModel,
)

DATA = Path(__file__).resolve().parents[2] / "reference" / "data"
REF = json.loads((DATA / "case_cohort_r.json").read_text())
R = REF["references"]

#: Both fits solve the same estimating equations to Newton convergence
#: (R's coxph to a relative change in the log-likelihood of 1e-9, here
#: to a step of 1e-10 standard errors), so they agree to about 1e-9.
COEF_ATOL = 1e-6
SE_RTOL = 1e-5


def _cohort():
    co = {k: np.asarray(v) for k, v in REF["nwtco"].items()}
    Z = np.column_stack(
        [
            co["stage"] == 2,
            co["stage"] == 3,
            co["stage"] == 4,
            co["histol"] == 2,
            co["age_months"] / 12,
        ]
    ).astype(float)
    x = co["edrel"].astype(float)
    c = 1 - co["rel"]
    sub = co["in_subcohort"].astype(bool)
    return co, x, Z, c, sub


CO, X, Z, C, SUB = _cohort()
SEL = (CO["rel"] == 1) | SUB
ENTRY = np.floor(X / 4)
SIZES = {k: int(np.sum(CO["instit"] == k)) for k in (1, 2)}

CASES = [
    ("prentice", "cch_Prentice"),
    ("self-prentice", "cch_SelfPrentice"),
    ("lin-ying", "cch_LinYing"),
    ("barlow", "barlow"),
    ("borgan-i", "cch_I_Borgan"),
    ("borgan-ii", "cch_II_Borgan"),
]


def _check(model, key):
    ref = R[key]
    beta = np.asarray(ref["coef"])
    V = np.asarray(ref["var"])
    np.testing.assert_allclose(model.params, beta, rtol=0, atol=COEF_ATOL)
    np.testing.assert_allclose(
        model.standard_errors(), np.sqrt(np.diag(V)), rtol=SE_RTOL
    )
    np.testing.assert_allclose(model.covariance(), V, rtol=SE_RTOL, atol=1e-10)


def _fit(method, entry=False, **kwargs):
    if method.startswith("borgan"):
        kwargs.update(sampling_strata=CO["instit"][SEL], cohort_size=SIZES)
    else:
        kwargs.setdefault("cohort_size", 4028)
    return CoxPH.fit_case_cohort(
        X[SEL],
        Z[SEL],
        subcohort=SUB[SEL],
        c=C[SEL],
        tl=ENTRY[SEL] if entry else None,
        method=method,
        **kwargs,
    )


@pytest.mark.parametrize("entry", [False, True], ids=["right", "entry"])
@pytest.mark.parametrize("method,key", CASES)
def test_matches_r(method, key, entry):
    model = _fit(method, entry)
    assert isinstance(model, CaseCohortModel)
    _check(model, key + ("_entry" if entry else ""))


def test_lin_ying_robust_matches_r():
    _check(_fit("lin-ying", robust=True), "cch_LinYing_robust")


@pytest.mark.parametrize(
    "method,key",
    [
        ("prentice", "Prentice"),
        ("self-prentice", "SelfPrentice"),
        ("lin-ying", "LinYing"),
    ],
)
def test_whole_cohort_as_subcohort_matches_r(method, key):
    model = CoxPH.fit_case_cohort(
        X,
        Z,
        subcohort=np.ones(X.size, bool),
        c=C,
        cohort_size=X.size,
        method=method,
    )
    _check(model, "full_cohort_" + key)


# -- the whole cohort as the subcohort is the ordinary Cox fit -------------
@pytest.fixture(scope="module")
def cohort():
    rng = np.random.default_rng(20261127)
    N = 400
    Zs = np.column_stack([rng.normal(size=N), rng.integers(0, 2, N)])
    t = rng.weibull(1.5, N) * 10 * np.exp(-(Zs @ [0.5, -0.4]) / 1.5)
    # Rounded to whole units, so there are ties
    t = np.ceil(t)
    cens = np.ceil(rng.uniform(5, 25, N))
    x = np.minimum(t, cens)
    c = (t > cens).astype(int)
    tl = np.where(rng.random(N) < 0.3, np.floor(x / 3), 0.0)
    return x, Zs, c, tl


@pytest.mark.parametrize(
    "method,ties",
    [
        ("prentice", "efron"),
        ("lin-ying", "efron"),
        ("barlow", "efron"),
        ("prentice", "breslow"),
        ("lin-ying", "breslow"),
        ("self-prentice", "breslow"),
    ],
)
def test_whole_cohort_is_the_cox_fit(cohort, method, ties):
    x, Zs, c, tl = cohort
    kw = {"tie_method": ties} if method != "self-prentice" else {}
    model = CoxPH.fit_case_cohort(
        x,
        Zs,
        subcohort=np.ones(x.size, bool),
        c=c,
        tl=tl,
        cohort_size=x.size,
        method=method,
        **kw,
    )
    cox = CoxPH.fit(x, Zs, c=c, tl=tl, tie_method=ties)
    np.testing.assert_allclose(model.params, cox.params, atol=1e-8)
    if method == "barlow":
        # The robust (sandwich) variance, as a weighted fit's
        ref = CoxPH.fit(x, Zs, c=c, tl=tl, w=np.ones(x.size), tie_method=ties)
        np.testing.assert_allclose(
            model.covariance(), ref.covariance(), rtol=1e-6
        )
    elif not (method == "prentice" and ties == "efron"):
        # The phase-two term is 0: the model-based variance, of the
        # estimating equation the variance is taken from (Prentice's
        # is Self-Prentice's, whose risk sets leave the tied case out:
        # Breslow's)
        np.testing.assert_allclose(
            model.covariance(), cox.covariance(), rtol=1e-6
        )
    # The baseline: Breslow's on the cohort's risk sets
    breslow = CoxPH.fit(x, Zs, c=c, tl=tl, tie_method="breslow")
    if ties == "breslow":
        np.testing.assert_allclose(model.x, breslow.x)
        np.testing.assert_allclose(model.H0, breslow.H0, rtol=1e-6)
        np.testing.assert_allclose(
            model.sf([3.0, 10.0], [0.2, 1.0]),
            breslow.sf([3.0, 10.0], [0.2, 1.0]),
            rtol=1e-6,
        )


def test_self_prentice_whole_cohort_is_breslow_with_ties(cohort):
    x, Zs, c, tl = cohort
    model = CoxPH.fit_case_cohort(
        x,
        Zs,
        subcohort=np.ones(x.size, bool),
        c=c,
        tl=tl,
        cohort_size=x.size,
        method="self-prentice",
    )
    efron = CoxPH.fit(x, Zs, c=c, tl=tl, tie_method="efron")
    breslow = CoxPH.fit(x, Zs, c=c, tl=tl, tie_method="breslow")
    np.testing.assert_allclose(model.params, breslow.params, atol=1e-8)
    assert np.max(np.abs(model.params - efron.params)) > 1e-4


def test_baseline_estimates_the_cohorts():
    """A seeded case-cohort sample's weighted baseline is near the full
    cohort's (fixed before the result was seen: 4000 units, 10%
    subcohort, within 15% at three times)."""
    rng = np.random.default_rng(7)
    N = 4000
    z = rng.normal(size=N)
    t = rng.exponential(1 / (0.01 * np.exp(0.6 * z)))
    x, c = np.minimum(t, 20.0), (t > 20.0).astype(int)
    sub = rng.random(N) < 0.10
    keep = sub | (c == 0)
    cox = CoxPH.fit(x, z, c=c, tie_method="breslow")
    for method in ("prentice", "lin-ying", "barlow"):
        model = CoxPH.fit_case_cohort(
            x[keep],
            z[keep],
            subcohort=sub[keep],
            c=c[keep],
            cohort_size=N,
            method=method,
        )
        times = [5.0, 10.0, 19.0]
        np.testing.assert_allclose(
            model.Hf(times, [0.0]), cox.Hf(times, [0.0]), rtol=0.15
        )


# -- refusals --------------------------------------------------------------
def test_refuses_what_a_pseudo_likelihood_cannot_give():
    model = _fit("prentice")
    for call in (
        model.neg_ll,
        model.aic,
        model.bic,
        model.likelihood_ratio_test,
        model.compute_residuals,
        model.diagnose,
        model.robust_covariance,
        model.to_dict,
        lambda: model.cb([100.0], Z[0]),
        lambda: model.log_likelihood,
    ):
        with pytest.raises(ValueError, match="case-cohort sample"):
            call()
    # What it does give
    assert model.summary().shape[0] == 5
    assert "Design              : case-cohort" in repr(model)
    assert np.all(np.diff(model.sf([100.0, 1000.0, 3000.0], Z[0])) < 0)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"method": "cch"}, "not a case-cohort estimator"),
        ({"tie_method": "exact"}, "'efron'"),
        ({"robust": True}, "Lin-Ying"),
        ({"sampling_strata": np.ones(4)}, "borgan"),
        ({"cohort_size": 3}, "smaller than the sample"),
        ({"cohort_size": {1: 10}}, "one number"),
    ],
)
def test_argument_refusals(kwargs, match):
    x = np.array([1.0, 2.0, 3.0, 4.0])
    kw = {"cohort_size": 10, **kwargs}
    with pytest.raises(ValueError, match=match):
        CoxPH.fit_case_cohort(
            x,
            [0.1, 0.5, -0.2, 0.3],
            subcohort=[1, 1, 1, 0],
            c=[0, 1, 1, 0],
            **kw,
        )


def test_censored_unit_outside_subcohort_is_refused():
    with pytest.raises(ValueError, match="outside the subcohort"):
        CoxPH.fit_case_cohort(
            [1.0, 2.0, 3.0, 4.0],
            [0.1, 0.5, -0.2, 0.3],
            subcohort=[1, 1, 0, 1],
            c=[0, 1, 1, 0],
            cohort_size=10,
        )


def test_borgan_needs_strata():
    with pytest.raises(ValueError, match="sampling_strata"):
        CoxPH.fit_case_cohort(
            X[SEL],
            Z[SEL],
            subcohort=SUB[SEL],
            c=C[SEL],
            cohort_size=SIZES,
            method="borgan-i",
        )


def test_dataframe_names_the_coefficients():
    import pandas as pd

    df = pd.DataFrame(Z[SEL], columns=REF["columns"])
    model = CoxPH.fit_case_cohort(
        X[SEL], df, subcohort=SUB[SEL], c=C[SEL], cohort_size=4028
    )
    assert list(model.summary().index) == REF["columns"]
    _check(model, "cch_Prentice")


# -- nested case-control ---------------------------------------------------
NCC = REF["ncc_sample"]


@pytest.mark.parametrize("ties", ["exact", "breslow"])
def test_nested_case_control_matches_clogit(ties):
    rows = np.asarray(NCC["row"]) - 1
    model = CoxPH.fit_nested_case_control(
        Z[rows], NCC["case"], NCC["set"], tie_method=ties
    )
    assert isinstance(model, NestedCaseControlModel)
    _check(model, "ncc_" + ties)


def test_infert_matches_clogit():
    inf = REF["infert"]
    model = CoxPH.fit_nested_case_control(
        np.column_stack([inf["spontaneous"], inf["induced"]]),
        inf["case"],
        inf["stratum"],
    )
    _check(model, "infert")
    # One case per set: the exact likelihood is Breslow's and Efron's
    for ties in ("efron", "breslow"):
        other = CoxPH.fit_nested_case_control(
            np.column_stack([inf["spontaneous"], inf["induced"]]),
            inf["case"],
            inf["stratum"],
            tie_method=ties,
        )
        np.testing.assert_allclose(other.params, model.params, atol=1e-8)
    # A true conditional likelihood: its test is offered
    assert model.likelihood_ratio_test()["df"] == 2


def test_nested_case_control_refuses_curves():
    inf = REF["infert"]
    model = CoxPH.fit_nested_case_control(
        inf["spontaneous"], inf["case"], inf["stratum"]
    )
    for name in ("sf", "ff", "Hf", "hf", "df", "qf"):
        with pytest.raises(ValueError, match="no baseline"):
            getattr(model, name)([1.0], [1.0])
    with pytest.raises(ValueError, match="no baseline"):
        model.cb([1.0], [1.0])
    assert "nested case-control, 83 matched sets" in repr(model)


def test_nested_case_control_warns_on_sets_without_a_control():
    Zs = np.array([1.2, 0.1, -0.4, 0.8, 0.3, -1.0, 0.0, 0.5, -0.2, 0.7])
    case = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1])
    sets = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 4])
    with pytest.warns(UserWarning, match="1 of 4 matched sets"):
        model = CoxPH.fit_nested_case_control(Zs, case, sets)
    ref = CoxPH.fit_nested_case_control(Zs[:9], case[:9], sets[:9])
    np.testing.assert_allclose(model.params, ref.params)
    assert model.sets == 3


def test_lin_ying_needs_non_cases_in_the_subcohort():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    z = np.array([0.1, 0.5, -0.2, 0.3, 0.9, -0.4])
    # Every subcohort member a case
    for method in ("lin-ying", "borgan-ii"):
        kw = (
            {"sampling_strata": np.ones(6), "cohort_size": {1.0: 20}}
            if method == "borgan-ii"
            else {"cohort_size": 20}
        )
        with pytest.raises(ValueError, match="not cases"):
            CoxPH.fit_case_cohort(
                x,
                z,
                subcohort=[1, 1, 1, 0, 0, 0],
                c=np.zeros(6),
                method=method,
                **kw,
            )


def test_self_prentice_needs_a_subcohort_member_at_risk():
    # The last case (outside the subcohort) outlives every subcohort member
    x = np.array([1.0, 2.0, 3.0, 4.0, 9.0])
    z = np.array([0.1, 0.5, -0.2, 0.3, 0.9])
    for method in ("prentice", "self-prentice"):
        with pytest.raises(ValueError, match="no subcohort member is at risk"):
            CoxPH.fit_case_cohort(
                x,
                z,
                subcohort=[1, 1, 1, 1, 0],
                c=[0, 1, 0, 1, 0],
                cohort_size=20,
                method=method,
            )
    # Barlow's and Lin-Ying's risk sets hold the case itself
    model = CoxPH.fit_case_cohort(
        x,
        z,
        subcohort=[1, 1, 1, 1, 0],
        c=[0, 1, 0, 1, 0],
        cohort_size=20,
        method="barlow",
    )
    assert np.isfinite(model.params).all()


def test_nested_case_control_refuses_missing_covariates():
    Zs = np.array([1.2, np.nan, -0.4, 0.8, 0.3, -1.0])
    with pytest.raises(ValueError, match="missing or infinite"):
        CoxPH.fit_nested_case_control(
            Zs, [1, 0, 0, 1, 0, 0], [1, 1, 1, 2, 2, 2]
        )
