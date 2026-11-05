"""The cause-specific Cox model with delayed entry
(``CompetingRisksProportionalHazards.fit(..., tl=)``) against stored R
``survival`` results (``scripts/reference/reference_r_competing_entry.R``).

Each cause's ``coxph(Surv(tl, x, cause == k) ~ ...)``, the other causes
censored, under Efron and Breslow ties: the coefficients, their standard
errors, the partial log-likelihood and the baseline cumulative hazard at
``Z = 0``; and, under Breslow ties, the cumulative incidences of the
multi-state ``coxph(Surv(tl, x, factor(cause)) ~ ..., id = id)``,
``survfit(fit, newdata)``'s ``pstate``. On ``pbc`` (entries rounded to
0.01 years) and the synthetic ``competing`` set, whose integer times put
many entries exactly on an event time. The coefficients agree with R's to
about 7e-9 (relative; an optimiser's answer), the standard errors to 1e-9,
the baselines to 2e-8 (relative) and the incidences to 6e-11 (absolute);
they are tested to ``rtol=1e-6``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
)

from ._data import fixture, values

SOURCE = "r_competing_entry"
TOL = dict(rtol=1e-6, atol=1e-9)
NEWDATA = {
    "pbc": [[45.0, 0.0, 1.0], [60.0, 1.5, 0.0]],
    "competing": [[0.0, 0.0], [1.0, 1.0]],
}


def _data(name):
    d = fixture(name)
    e = np.where(d["cause"] == 0, None, d["cause"].astype(int))
    e = e.astype(object)
    if name == "pbc":
        x = d["years"]
        tl = values(SOURCE, "pbc_entry")["tl"]
        Z = np.column_stack([d["age"], d["log_bili"], d["female"]])
    else:
        x = d["x"]
        row = np.arange(1, x.size + 1)
        tl = x * (row % 3) / 4
        Z = np.column_stack([d["z"], d["group"]])
    return x, Z, e, tl


def _fit(name, ties):
    x, Z, e, tl = _data(name)
    return CompetingRisksProportionalHazards.fit(
        x, Z, e, tl=tl, tie_method=ties
    )


@pytest.mark.parametrize("ties", ["efron", "breslow"])
@pytest.mark.parametrize("name", ["pbc", "competing"])
def test_cause_specific_cox_with_entry_matches_coxph(name, ties):
    model = _fit(name, ties)
    p = model.betas.shape[1]
    se = model.standard_errors().reshape(-1, p)
    for k in (1, 2):
        ref = values(SOURCE, f"{name}_cause{k}_{ties}")
        i = model.event_idx_map[k]
        assert_allclose(model.betas[i], ref["coef"], **TOL)
        assert_allclose(se[i], ref["se"], **TOL)
        assert model.models is not None
        assert_allclose(-model.models[k].neg_ll(), ref["loglik"], **TOL)
        assert_allclose(
            model.Hf(ref["time"], np.zeros(p), event=k), ref["cumhaz"], **TOL
        )


@pytest.mark.parametrize("name", ["pbc", "competing"])
def test_cause_specific_cif_with_entry_matches_multistate_survfit(name):
    model = _fit(name, "breslow")
    ref = values(SOURCE, f"{name}_cif_breslow")
    for j, z in enumerate(NEWDATA[name], start=1):
        for k in (1, 2):
            assert_allclose(
                model.cif(ref["times"], z, k), ref[f"cif{k}_row{j}"], **TOL
            )


def test_the_summed_partial_likelihood_is_the_causes():
    model = _fit("pbc", "efron")
    total = sum(
        values(SOURCE, f"pbc_cause{k}_efron")["loglik"] for k in (1, 2)
    )
    assert_allclose(model.log_likelihood, total, **TOL)
