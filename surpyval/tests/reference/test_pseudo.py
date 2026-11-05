"""Pseudo-observations and the regressions on them against stored R
results (``scripts/reference/reference_r_pseudo.R``).

* The pseudo-values against ``pseudo::pseudosurv``, ``pseudomean`` and
  ``pseudoci`` on ``lung``, ``ovarian``, the PBC trial and the seeded
  competing-risks fixture (integer times, with causes tied with each other
  and with censorings). Both sides are the same closed-form product-limit
  sums: ``atol=1e-8``.
* The regressions against ``geepack::geeglm(..., corstr =
  "independence")`` on those pseudo-values (gaussian variance, robust
  sandwich standard errors). Both solve the same estimating equations to
  their convergence tolerance: coefficients and standard errors to
  ``rtol=1e-6``. SurPyval's survival regressions model ``1 - S(t)``, as
  the R script's ``1 - pseudosurv``; its RMST coefficients have the
  opposite sign of R's (a positive coefficient means a shorter life).
* ``survRM2::rmst2(covariates=)`` is an IPCW regression, a different
  estimator: only a loose check that the RMST difference agrees to within
  a quarter of its standard error.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval import (
    pseudo_cif,
    pseudo_regression,
    pseudo_rmst,
    pseudo_survival,
)

from ._data import fixture, values

PSEUDO = dict(rtol=0, atol=1e-8)
GEE = dict(rtol=1e-6, atol=1e-10)


def _causes(codes):
    return np.array([None if k == 0 else int(k) for k in codes], dtype=object)


def _lung():
    d = fixture("lung")
    return d["time"], d["c"], np.c_[d["sex"], d["age"]]


def _ovarian():
    d = fixture("ovarian")
    return d["futime"], 1 - d["fustat"], np.c_[d["rx"], d["age"]]


def test_pseudosurv_lung():
    x, c, _ = _lung()
    ref = values("r_pseudo", "pseudosurv_lung")
    assert_allclose(
        pseudo_survival(x, ref["time"], c), ref["pseudo"], **PSEUDO
    )


def test_pseudosurv_ovarian():
    x, c, _ = _ovarian()
    ref = values("r_pseudo", "pseudosurv_ovarian")
    assert_allclose(
        pseudo_survival(x, ref["time"], c), ref["pseudo"], **PSEUDO
    )


@pytest.mark.parametrize(
    "ref_id, data, tau",
    [("pseudomean_lung", _lung, 365), ("pseudomean_ovarian", _ovarian, 1000)],
)
def test_pseudomean(ref_id, data, tau):
    x, c, _ = data()
    ref = values("r_pseudo", ref_id)
    assert_allclose(pseudo_rmst(x, tau, c), ref["pseudo"], **PSEUDO)


@pytest.mark.parametrize(
    "ref_id, name, time",
    [
        ("pseudoci_competing", "competing", "x"),
        ("pseudoci_pbc", "pbc", "years"),
    ],
)
@pytest.mark.parametrize("cause", [1, 2])
def test_pseudoci(ref_id, name, time, cause):
    d = fixture(name)
    ref = values("r_pseudo", ref_id)
    got = pseudo_cif(d[time], _causes(d["cause"]), ref["time"], cause)
    assert_allclose(got, ref[f"cause{cause}"], **PSEUDO)


def _check(ref_id, res, n_covariates, sign=1.0):
    ref = values("r_pseudo", ref_id)
    coef = np.array(ref["coef"])
    coef[-n_covariates:] *= sign
    assert_allclose(res.coef, coef, **GEE)
    assert_allclose(res.se, ref["se"], **GEE)
    assert_allclose(res.p, ref["p"], rtol=1e-5, atol=1e-12)


@pytest.mark.parametrize("link", ["cloglog", "logit"])
def test_survival_regression_lung(link):
    x, c, Z = _lung()
    res = pseudo_regression(x, Z, c, t=[180, 365, 540], link=link)
    _check(f"gee_survival_lung_{link}", res, 2)


def test_survival_regression_one_time():
    x, c, Z = _lung()
    res = pseudo_regression(x, Z, c, t=365)
    assert res.names[0] == "Intercept"
    _check("gee_survival_lung_365_cloglog", res, 2)


def test_survival_regression_ovarian():
    x, c, Z = _ovarian()
    res = pseudo_regression(x, Z, c, t=[365, 730])
    _check("gee_survival_ovarian_cloglog", res, 2)


@pytest.mark.parametrize("link", ["identity", "log"])
@pytest.mark.parametrize(
    "name, data, tau", [("lung", _lung, 365), ("ovarian", _ovarian, 1000)]
)
def test_rmst_regression(link, name, data, tau):
    x, c, Z = data()
    res = pseudo_regression(x, Z, c, quantity="rmst", tau=tau, link=link)
    # R's coefficients have the sign of the RMST; SurPyval's the opposite.
    _check(f"gee_rmst_{name}_{link}", res, 2, sign=-1.0)


@pytest.mark.parametrize("link", ["cloglog", "logit"])
@pytest.mark.parametrize("cause", [1, 2])
def test_cif_regression_competing(link, cause):
    d = fixture("competing")
    res = pseudo_regression(
        d["x"],
        np.c_[d["group"], d["z"]],
        quantity="cif",
        e=_causes(d["cause"]),
        cause=cause,
        t=[2, 4, 8],
        link=link,
    )
    _check(f"gee_cif_competing_cause{cause}_{link}", res, 2)


def test_cif_regression_pbc():
    d = fixture("pbc")
    res = pseudo_regression(
        d["years"],
        np.c_[d["drug"], d["age"], d["log_bili"]],
        quantity="cif",
        e=_causes(d["cause"]),
        cause=2,
        t=[2, 5, 8],
    )
    _check("gee_cif_pbc_cause2_cloglog", res, 3)


def test_rmst_regression_near_survrm2():
    # A different estimator (IPCW): the female-male RMST difference agrees
    # to within a quarter of its standard error.
    x, c, Z = _lung()
    res = pseudo_regression(x, Z, c, quantity="rmst", tau=365)
    ref = values("r_pseudo", "rmst2_lung")
    assert abs(-res.coef[1] - ref["coef"][1]) < 0.25 * ref["se"][1]
    assert_allclose(res.se[1], ref["se"][1], rtol=0.1)
