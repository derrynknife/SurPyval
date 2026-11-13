"""The stratified and the clustered Fine-Gray model,
``FineGray.fit(..., strata=, strata_censoring=, cluster=)``, against R.

R 4.6.2 with ``crrSC`` 1.1.2 and ``cmprsk`` 2.2.12, stored in
``data/r_crrsc.json`` by ``scripts/reference/reference_r_crrsc.R``, on two
seeded synthetic fixtures (``make_fixtures.fine_gray_strata``: three strata
with their own baselines and censoring; ``fine_gray_cluster``: 100 pairs of
dependent units sharing a covariate):

- ``crrs(..., ctype = 1)``, the censoring distribution within each
  stratum, is ``strata=`` (``strata_censoring="within"``, the default):
  coefficients and robust standard errors, on the three strata and on the
  100 pairs taken as small strata;
- ``crrs(..., ctype = 2)``, one censoring distribution for every stratum,
  is ``strata_censoring="common"``. crrSC's variance for it sums the
  score residuals within each stratum (the strata are its clusters), so
  it is ``strata_censoring="common"`` with ``cluster=`` the strata too;
- ``crrc(..., cluster = cluster)`` is ``cluster=``; without the clusters
  it is ``cmprsk::crr``'s fit;
- each stratum's cumulative incidence at a covariate row, from ``crr`` on
  the stratum's rows with the stratified coefficients held, is
  ``cif(x, Z, stratum=...)``.

R's fits are at ``gtol = 1e-12``. Tolerances: the coefficients and
standard errors to ``rtol=1e-7`` (two Newton solutions, each converged
to about 1e-10 in its own arithmetic), the incidences to ``1e-8``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.univariate.competing_risks import FineGray

from ._data import fixture, reference, values


def _rows(name):
    d = fixture(name)
    Z = np.column_stack([d["z1"], d["z2"]])
    e = np.where(d["cause"] == 0, None, d["cause"].astype(int))
    return d, d["x"], Z, e.astype(object)


def _check(model, coef, se):
    assert_allclose(model.beta, coef, rtol=1e-7)
    assert_allclose(model.standard_errors(), se, rtol=1e-7)


def test_strata_within_is_crrs_ctype_1():
    d, x, Z, e = _rows("fine_gray_strata")
    ref = values("r_crrsc", "strata")
    model = FineGray.fit(x, Z, e, event=1, strata=d["stratum"])
    _check(model, ref["coef_within"], ref["se_within"])


def test_strata_common_clustered_by_stratum_is_crrs_ctype_2():
    d, x, Z, e = _rows("fine_gray_strata")
    ref = values("r_crrsc", "strata")
    model = FineGray.fit(
        x,
        Z,
        e,
        event=1,
        strata=d["stratum"],
        strata_censoring="common",
        cluster=d["stratum"],
    )
    _check(model, ref["coef_common"], ref["se_common"])


@pytest.mark.parametrize("censoring", ["within", "common"])
def test_many_small_strata_are_crrs(censoring):
    d, x, Z, e = _rows("fine_gray_cluster")
    ref = values("r_crrsc", "cluster")
    kw = {"strata": d["cluster"], "strata_censoring": censoring}
    if censoring == "common":
        kw["cluster"] = d["cluster"]
    model = FineGray.fit(x, Z[:, 1:], e, event=1, **kw)
    _check(
        model, [ref[f"coef_small_{censoring}"]], [ref[f"se_small_{censoring}"]]
    )


def test_stratum_cif_is_crr_on_the_stratum():
    d, x, Z, e = _rows("fine_gray_strata")
    entry = reference("r_crrsc", "strata")
    model = FineGray.fit(x, Z, e, event=1, strata=d["stratum"])
    z, times = entry["settings"]["z"], entry["settings"]["times"]
    for label in (1, 2, 3):
        assert_allclose(
            model.cif(times, z, stratum=label),
            entry["values"][f"cif_{label}"],
            rtol=1e-8,
        )


def test_one_stratum_is_crr():
    _, x, Z, e = _rows("fine_gray_strata")
    ref = values("r_crrsc", "strata")
    model = FineGray.fit(x, Z, e, event=1, strata=np.zeros(x.size))
    _check(model, ref["coef_crr"], ref["se_crr"])


def test_cluster_is_crrc():
    d, x, Z, e = _rows("fine_gray_cluster")
    ref = values("r_crrsc", "cluster")
    model = FineGray.fit(x, Z, e, event=1, cluster=d["cluster"])
    _check(model, ref["coef_cluster"], ref["se_cluster"])
    alone = FineGray.fit(x, Z, e, event=1)
    _check(alone, ref["coef_crr"], ref["se_crr"])
    # A cluster of one row each is no cluster at all
    singletons = FineGray.fit(x, Z, e, event=1, cluster=np.arange(x.size))
    assert_allclose(singletons.covariance(), alone.covariance(), rtol=1e-12)
