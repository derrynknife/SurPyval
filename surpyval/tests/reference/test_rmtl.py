"""The restricted mean time lost (RMTL) to a cause, ``CompetingRisks.rmtl``
and ``surpyval.rmtl_diff``, against R.

R 4.6.2, ``survival`` 3.8.12, ``cmprsk`` 2.2.12, ``survRM2`` 1.0.4 and
``GFDrmtl`` 0.1.0,
stored in ``data/r_rmtl.json`` by ``scripts/reference/reference_r_rmtl.R``,
on the ``competing`` fixture (all rows, and each group) and ``pbc``, each
cause, at three horizons:

- the time lost is the area under the step Aalen-Johansen incidence of
  ``survfit(Surv(x, factor(cause)) ~ 1)``, under ``cmprsk::cuminc``'s, and
  survival's own ``summary(fit, rmean = tau)``: all three, to rounding;
- the standard error matches ``GFDrmtl``'s asymptotic one (Munko, Dobler
  and Ditzhaus; its internal ``RMTL`` function, which its ``RMTL.test``
  uses) and SurPyval's delta-method formula written out in R, to
  rounding. survival's ``se(rmean)`` is the infinitesimal-jackknife
  standard error, a different estimator of the same variance, so it is
  compared loosely: equal to rounding for the second group of
  ``competing``, within 3e-5 (relative) for ``pbc`` and within 0.83% for
  ``competing``'s first group and all rows (stored, checked to 1%);
- with every failure taken as one cause the time lost is ``tau`` less the
  restricted mean survival time, and its standard error the RMST's:
  ``survRM2``'s ``rmst1`` (its one-arm function, which reports the time
  lost too), to rounding.

Tolerance: closed forms of the same counts, equal to rounding
(``rtol=1e-10``) except the jackknife comparison above.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

CASES = ["competing_all", "competing_group0", "competing_group1", "pbc"]


def _data(case):
    if case == "pbc":
        d = fixture("pbc")
        return d["years"], d["cause"]
    d = fixture("competing")
    if case == "competing_all":
        return d["x"], d["cause"]
    keep = d["group"] == float(case[-1])
    return d["x"][keep], d["cause"][keep]


def _model(x, cause):
    e = np.where(cause == 0, None, cause.astype(int)).astype(object)
    return sp.CompetingRisks.fit(x, e)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("k", [1, 2])
def test_rmtl_and_its_standard_error_match_r(case, k):
    ref = values("r_rmtl", case)
    model = _model(*_data(case))
    for i, tau in enumerate(ref["taus"]):
        res = model.rmtl(k, tau=tau)
        for source in ("survfit", "cuminc"):
            assert_allclose(
                res["rmtl"], ref[f"rmtl_{source}_{k}"][i], rtol=1e-10
            )
        assert_allclose(res["rmtl"], ref[f"rmean_survival_{k}"][i], rtol=1e-10)
        assert_allclose(res["se"], ref[f"se_delta_{k}"][i], rtol=1e-10)
        assert_allclose(res["se"], ref[f"se_gfdrmtl_{k}"][i], rtol=1e-10)
        assert_allclose(res["rmtl"], ref[f"rmtl_gfdrmtl_{k}"][i], rtol=1e-10)
        assert_allclose(res["se"], ref[f"se_rmean_survival_{k}"][i], rtol=0.01)
        if case == "competing_group1":
            assert_allclose(
                res["se"], ref[f"se_rmean_survival_{k}"][i], rtol=1e-10
            )


@pytest.mark.parametrize("case", CASES)
def test_one_cause_time_lost_is_tau_less_the_rmst(case):
    ref = values("r_rmtl", case)
    x, cause = _data(case)
    one = sp.CompetingRisks.fit(
        x, np.where(cause == 0, None, "failure").astype(object)
    )
    km = sp.KaplanMeier.fit(x, c=(cause == 0).astype(int))
    for i, tau in enumerate(ref["taus"]):
        res = one.rmtl("failure", tau=tau)
        assert_allclose(res["rmtl"], ref["one_cause_rmtl"][i], rtol=1e-10)
        assert_allclose(res["se"], ref["one_cause_se"][i], rtol=1e-10)
        rmst = km.rmst(tau=tau)
        assert_allclose(res["rmtl"], tau - rmst["rmst"], rtol=1e-12)
        assert_allclose(res["se"], rmst["se"], rtol=1e-12)


def test_rmtl_diff_of_the_two_groups():
    ref0 = values("r_rmtl", "competing_group0")
    ref1 = values("r_rmtl", "competing_group1")
    a = _model(*_data("competing_group1"))
    b = _model(*_data("competing_group0"))
    for i, tau in enumerate(ref0["taus"]):
        res = sp.rmtl_diff(a, b, 1, tau=tau)
        diff = ref1["rmtl_survfit_1"][i] - ref0["rmtl_survfit_1"][i]
        se = np.hypot(ref1["se_delta_1"][i], ref0["se_delta_1"][i])
        assert_allclose(res["difference"], diff, rtol=1e-10)
        assert_allclose(res["se"], se, rtol=1e-10)
        assert_allclose(
            res["ratio"],
            ref1["rmtl_survfit_1"][i] / ref0["rmtl_survfit_1"][i],
            rtol=1e-10,
        )
