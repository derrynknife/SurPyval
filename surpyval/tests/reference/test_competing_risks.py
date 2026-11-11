"""SurPyval's competing-risks estimators against stored R ``cmprsk``
results (#379): the Aalen-Johansen cumulative incidence (``cuminc``),
Gray's test (``cuminc``'s ``Tests``) and the Fine-Gray model (``crr``), on
the PBC trial (death and transplant) and on a fixture whose causes tie
with each other and with censorings.

Tolerances: the cumulative incidence is a closed-form product-limit sum
(``rtol=1e-9``). ``crr`` solves the Fine-Gray score equations by
Newton-Raphson and SurPyval maximises the same likelihood with BFGS; they
agree to ~2e-6 relatively on the coefficients, hence ``rtol=1e-5``, and
the quantities derived from them (standard errors, predicted incidence) to
the same order.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.univariate.competing_risks import CompetingRisks, FineGray
from surpyval.univariate.competing_risks.aalen_johansen import (
    aalen_johansen_variance,
)

from ._data import fixture, values

EXACT = dict(rtol=1e-9, atol=1e-12)
FIT = dict(rtol=1e-5, atol=1e-8)

# name of the stored cuminc result -> (fixture, time column, group column)
CUMINC = {
    "cuminc_competing": ("competing", "x", "group"),
    "cuminc_pbc": ("pbc", "years", "drug"),
}


def _causes(cause_codes):
    """cmprsk's 0 / 1 / 2 cause codes as SurPyval labels (None censored)."""
    return np.array(
        [None if k == 0 else int(k) for k in cause_codes], dtype=object
    )


@pytest.mark.parametrize("ref_id", sorted(CUMINC))
def test_cumulative_incidence_matches_cuminc(ref_id):
    name, time, group = CUMINC[ref_id]
    d = fixture(name)
    e = _causes(d["cause"])
    ref = values("r_cmprsk", ref_id)
    for row, curve in enumerate(ref["curves"]):
        g, cause = curve.split()
        keep = d[group] == float(g)
        model = CompetingRisks.fit(d[time][keep], e[keep])
        est = ref["est"][row]
        # timepoints() gives NA past a group's last time; SurPyval holds
        # the last value there. Compare where cmprsk answers.
        known = ~np.isnan(est)
        assert_allclose(
            model.cif(ref["times"][known], int(cause)), est[known], **EXACT
        )


def _cuminc_curves(ref_id):
    """(model, cause, stored row) of each of a stored cuminc's curves."""
    if ref_id == "cuminc_competing_pooled":
        name, time, group = "competing", "x", None
    else:
        name, time, group = CUMINC[ref_id]
    d = fixture(name)
    e = _causes(d["cause"])
    for row, curve in enumerate(values("r_cmprsk", ref_id)["curves"]):
        g, cause = curve.split()
        keep = np.ones(e.size, bool) if group is None else d[group] == float(g)
        yield CompetingRisks.fit(d[time][keep], e[keep]), int(cause), row


@pytest.mark.parametrize(
    "ref_id", sorted(CUMINC) + ["cuminc_competing_pooled"]
)
def test_cumulative_incidence_variance_matches_cuminc(ref_id):
    # cuminc's var: Aalen's (1978) variance, with its tie correction (the
    # competing fixture ties causes with each other and with censorings).
    # The normal bound is the estimate +- z sqrt(var).
    from scipy.stats import norm

    ref = values("r_cmprsk", ref_id)
    times = ref["times"]
    for model, cause, row in _cuminc_curves(ref_id):
        known = ~np.isnan(ref["var"][row])
        k = model.event_idx_map[cause]
        var = aalen_johansen_variance(model.r, model.d, model.d_e[k])
        idx = np.searchsorted(model.x, times[known], side="right") - 1
        got = np.where(idx < 0, 0.0, var[np.maximum(idx, 0)])
        assert_allclose(got, ref["var"][row][known], **EXACT)
        b = model.cb(times[known], cause, bound_type="normal")
        half = norm.ppf(0.975) * np.sqrt(ref["var"][row][known])
        est = ref["est"][row][known]
        assert_allclose(b, np.column_stack([est - half, est + half]), **EXACT)


def test_pooled_cumulative_incidence_matches_cuminc():
    d = fixture("competing")
    ref = values("r_cmprsk", "cuminc_competing_pooled")
    model = CompetingRisks.fit(d["x"], _causes(d["cause"]))
    for row, curve in enumerate(ref["curves"]):
        cause = int(curve.split()[1])
        assert_allclose(model.cif(ref["times"], cause), ref["est"][row])


# Gray's test: the score and variance are cmprsk's crst routine (#380: the
# variance used to be SurPyval's own linearisation, 6.162 against cmprsk's
# 5.065 on the tied fixture, cause 2, and the rho = 1 score differed in the
# pooled F^0 of the weight).
@pytest.mark.parametrize(
    "ref_id, rho",
    [
        ("cuminc_competing", 0),
        ("cuminc_pbc", 0),
        ("cuminc_competing_rho1", 1),
    ],
)
def test_gray_test_matches_cuminc(ref_id, rho):
    name, time, group = CUMINC[ref_id.replace("_rho1", "")]
    d = fixture(name)
    e = _causes(d["cause"])
    ref = values("r_cmprsk", ref_id)
    for k, cause in enumerate(ref["gray_cause"]):
        res = sp.gray_test(d[time], e, d[group], event=int(cause), rho=rho)
        assert res.df == ref["gray_df"][k]
        assert_allclose(res.statistic, ref["gray_stat"][k], rtol=1e-9)


CRR = {
    "crr_competing_cause1": ("competing", "x", ["group", "z"], 1),
    "crr_competing_cause2": ("competing", "x", ["group", "z"], 2),
    "crr_pbc_death": (
        "pbc",
        "years",
        ["drug", "age", "log_bili", "female"],
        2,
    ),
}


@pytest.mark.parametrize("ref_id", sorted(CRR))
def test_fine_gray_matches_crr(ref_id):
    name, time, covariates, cause = CRR[ref_id]
    d = fixture(name)
    Z = np.column_stack([d[k] for k in covariates])
    ref = values("r_cmprsk", ref_id)
    model = FineGray.fit(d[time], Z, _causes(d["cause"]), event=cause)
    assert_allclose(model.beta, ref["coef"], **FIT)
    # The stored optimum of the weighted partial likelihood.
    assert_allclose(-model._neg_ll, ref["loglik"], rtol=0, atol=1e-6)
    # The default covariance is Fine and Gray's robust sandwich (crr's
    # var); robust=False gives the model-based one (crr's invinf).
    assert_allclose(model.covariance(), ref["var_sandwich"], **FIT)
    assert_allclose(
        model.standard_errors(), np.sqrt(np.diag(ref["var_sandwich"])), **FIT
    )
    assert_allclose(model.covariance(robust=False), ref["var_naive"], **FIT)
    assert_allclose(
        model.standard_errors(robust=False),
        np.sqrt(np.diag(ref["var_naive"])),
        **FIT,
    )
    # crr's own table: z and p on the robust standard errors.
    table = model.summary()
    z = np.asarray(ref["coef"]) / np.sqrt(np.diag(ref["var_sandwich"]))
    assert_allclose(table["z"], z, rtol=1e-4)
    assert_allclose(model.p_values, table["p"], rtol=1e-12)
    for j, z in enumerate(ref["z_new"]):
        assert_allclose(model.cif(ref["times"], z), ref["cif"][:, j], **FIT)


def test_crr_robust_errors_are_smaller_and_larger_than_model_based():
    """The range the theory page quotes: over crr's stored fits the
    robust standard errors are from 25% smaller to 17% larger than the
    model-based ones."""
    ratios = []
    for source, ref_ids in (
        ("r_cmprsk", sorted(CRR)),
        (
            "r_fine_gray_variance",
            [
                "pbc_half_years_cause1",
                "pbc_half_years_cause2",
                "competing_counts_cause1",
                "competing_counts_cause2",
                "competing_no_censoring_cause1",
            ],
        ),
    ):
        for ref_id in ref_ids:
            ref = values(source, ref_id)
            ratios += list(
                np.sqrt(
                    np.diag(ref["var_sandwich"]) / np.diag(ref["var_naive"])
                )
            )
    assert round(min(ratios), 2) == 0.75, min(ratios)
    assert round(max(ratios), 2) == 1.17, max(ratios)


# The cases of scripts/reference/reference_r_fine_gray_variance.R, which
# stress the censoring term of Fine and Gray's sandwich: heavy ties
# between events of both causes and censorings (pbc in half years), counts
# (crr given each row n times) and no censoring at all (the term is zero).
VARIANCE = "r_fine_gray_variance"


def _variance_case(ref_id):
    if ref_id.startswith("pbc_half_years"):
        d = fixture("pbc")
        x = np.ceil(d["years"] * 2) / 2
        Z = np.column_stack(
            [d[k] for k in ("drug", "age", "log_bili", "female")]
        )
        return x, Z, d["cause"], None
    d = fixture("competing")
    Z = np.column_stack([d["group"], d["z"]])
    if ref_id.startswith("competing_counts"):
        n = 1 + (np.arange(1, d["x"].size + 1) % 3)
        return d["x"], Z, d["cause"], n
    return d["x"], Z, np.where(d["cause"] == 0, 2, d["cause"]), None


@pytest.mark.parametrize(
    "ref_id",
    [
        "pbc_half_years_cause1",
        "pbc_half_years_cause2",
        "competing_counts_cause1",
        "competing_counts_cause2",
        "competing_no_censoring_cause1",
    ],
)
def test_fine_gray_variances_match_crr(ref_id):
    x, Z, cause, n = _variance_case(ref_id)
    ref = values(VARIANCE, ref_id)
    assert ref["converged"]
    model = FineGray.fit(x, Z, _causes(cause), n=n, event=int(ref_id[-1]))
    assert_allclose(model.beta, ref["coef"], **FIT)
    assert_allclose(-model._neg_ll, ref["loglik"], rtol=0, atol=1e-6)
    # About 4e-8 (relative) on the sandwich, an optimiser's answer.
    assert_allclose(model.covariance(), ref["var_sandwich"], **FIT)
    assert_allclose(model.covariance(robust=False), ref["var_naive"], **FIT)
    # The two differ by far more than the tolerance (3% to 25% here).
    assert not np.allclose(
        model.standard_errors(),
        model.standard_errors(robust=False),
        rtol=1e-2,
    )
