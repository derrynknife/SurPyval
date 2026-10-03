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
    # SurPyval's standard errors are the model-based ones (crr's invinf),
    # documented as such: Fine and Gray's sandwich (crr's var) is not
    # implemented.
    assert_allclose(model.se, np.sqrt(np.diag(ref["var_naive"])), **FIT)
    assert not np.allclose(
        model.se, np.sqrt(np.diag(ref["var_sandwich"])), rtol=1e-3
    )
    for j, z in enumerate(ref["z_new"]):
        assert_allclose(model.cif(ref["times"], z), ref["cif"][:, j], **FIT)
