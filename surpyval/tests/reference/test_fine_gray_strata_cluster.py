"""The stratified and the clustered Fine-Gray model,
``FineGray.fit(..., strata=, strata_censoring=, cluster=)``, against R.

R 4.6.2 with ``crrSC`` 1.1.2 and ``cmprsk`` 2.2.12, stored with the data
(crrSC's own example sets ``bce``, ``cdata`` and ``center``) in
``data/crrsc_fine_gray.json`` by ``scripts/reference/reference_r_crrsc.R``:

- ``crrs(..., ctype = 1)``, the censoring distribution within each
  stratum, is ``strata=`` (``strata_censoring="within"``, the default):
  coefficients and robust standard errors, on ``bce`` (three strata of
  the number of positive nodes) and ``center`` (a stratum per centre,
  153 of them);
- ``crrs(..., ctype = 2)``, one censoring distribution for every stratum,
  is ``strata_censoring="common"``. crrSC's variance for it sums the
  score residuals within each stratum (the strata are its clusters), so
  it is ``strata_censoring="common"`` with ``cluster=`` the strata too;
- ``crrc(..., cluster = ID)`` is ``cluster=``, on ``cdata`` (100 pairs);
  without the cluster it is ``cmprsk::crr``'s variance;
- each stratum's cumulative incidence at a covariate row, from ``crr`` on
  the stratum's rows with the stratified coefficients held, is
  ``cif(x, Z, stratum=...)``.

R's fits are at ``gtol = 1e-12``. Tolerances: the coefficients and
standard errors to ``rtol=1e-7`` (two Newton solutions, each converged
to about 1e-10 in its own arithmetic), the incidences to ``1e-8``.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.univariate.competing_risks import FineGray

DATA = Path(__file__).parent / "data" / "crrsc_fine_gray.json"


@pytest.fixture(scope="module")
def ref():
    with open(DATA) as f:
        return json.load(f)


def _causes(status):
    status = np.asarray(status)
    return np.where(status == 0, None, status).astype(object)


def _bce(ref):
    d = ref["data"]["bce"]
    Z = np.column_stack([d["trt"], d["age"], d["tsize"]]).astype(float)
    return np.asarray(d["time"], float), Z, _causes(d["type"]), d["stratum"]


def _center(ref):
    d = ref["data"]["center"]
    Z = np.column_stack([d["fm"], d["cells"]]).astype(float)
    x = np.asarray(d["ftime"], float)
    return x, Z, _causes(d["fstatus"]), np.asarray(d["id"])


def _check(model, coef, se):
    assert_allclose(model.beta, coef, rtol=1e-7)
    assert_allclose(model.standard_errors(), se, rtol=1e-7)


@pytest.mark.parametrize("name", ["bce", "center"])
def test_strata_within_is_crrs_ctype_1(ref, name):
    x, Z, e, strata = (_bce if name == "bce" else _center)(ref)
    values = ref["references"][name]["values"]
    model = FineGray.fit(x, Z, e, event=1, strata=strata)
    _check(model, values["coef_within"], values["se_within"])


@pytest.mark.parametrize("name", ["bce", "center"])
def test_strata_common_clustered_by_stratum_is_crrs_ctype_2(ref, name):
    x, Z, e, strata = (_bce if name == "bce" else _center)(ref)
    values = ref["references"][name]["values"]
    model = FineGray.fit(
        x,
        Z,
        e,
        event=1,
        strata=strata,
        strata_censoring="common",
        cluster=strata,
    )
    _check(model, values["coef_common"], values["se_common"])


def test_stratum_cif_is_crr_on_the_stratum(ref):
    x, Z, e, strata = _bce(ref)
    entry = ref["references"]["bce"]
    model = FineGray.fit(x, Z, e, event=1, strata=strata)
    z, times = entry["settings"]["z"], entry["settings"]["times"]
    for label in ("1-3", "4-9", "10+"):
        assert_allclose(
            model.cif(times, z, stratum=label),
            entry["values"]["cif_" + label],
            rtol=1e-8,
        )


def test_one_stratum_is_crr(ref):
    x, Z, e, _ = _bce(ref)
    values = ref["references"]["bce"]["values"]
    model = FineGray.fit(x, Z, e, event=1, strata=np.zeros(x.size))
    _check(model, values["coef_crr"], values["se_crr"])


def test_cluster_is_crrc(ref):
    d = ref["data"]["cdata"]
    x = np.asarray(d["ftime"], float)
    Z = np.asarray(d["z"], float)[:, None]
    e = _causes(d["fstatus"])
    values = ref["references"]["cdata"]["values"]
    model = FineGray.fit(x, Z, e, event=1, cluster=d["ID"])
    _check(model, [values["coef_cluster"]], [values["se_cluster"]])
    alone = FineGray.fit(x, Z, e, event=1)
    _check(alone, [values["coef_crr"]], [values["se_crr"]])
    assert_allclose(values["se_unclustered"], values["se_crr"], rtol=1e-10)
    # A cluster of one row each is no cluster at all
    singletons = FineGray.fit(x, Z, e, event=1, cluster=np.arange(x.size))
    assert_allclose(singletons.covariance(), alone.covariance(), rtol=1e-12)
