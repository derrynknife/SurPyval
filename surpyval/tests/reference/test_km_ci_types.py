"""The pointwise interval types of the non-parametric survival estimates,
``cb(bound_type=...)``, against R.

R 4.6.2 with ``survival`` 3.8.12 and ``km.ci`` 0.5.6, stored in
``data/r_km_ci.json`` by ``scripts/reference/reference_r_km_ci.R``:
``survfit(conf.type=...)`` for 'exp' ("log-log"), 'normal' ("plain"),
'log', 'logit' and 'arcsine' ("arcsin"), and ``km.ci(method=...)`` for
'peto' and 'rothman', on survival's ``aml`` (95% and 90%) and ``lung``
(95%) data, and for the Nelson-Aalen estimate (``stype = 2, ctype = 1``)
on ``aml``.

R cuts the plain interval to [0, 1] and SurPyval's 'normal' does not, so
it is compared cut. Values R leaves undefined (NA, where the estimate is
0) are not compared: SurPyval fills them as its ``cb`` documents.

Tolerance: closed forms of the same counts, equal to rounding
(``rtol=1e-10``).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

TYPES = ["exp", "normal", "log", "logit", "arcsine", "peto", "rothman"]
CASES = {
    "aml_km_95": ("aml", "KaplanMeier", 0.05),
    "aml_km_90": ("aml", "KaplanMeier", 0.10),
    "lung_km_95": ("lung", "KaplanMeier", 0.05),
    "aml_na_95": ("aml", "NelsonAalen", 0.05),
}


def _model(name, fitter):
    d = fixture(name)
    if name == "aml":
        x, c = d["time"], 1 - d["status"]
    else:
        x, c = d["time"], d["c"]
    return getattr(sp, fitter).fit(x, c=c)


@pytest.mark.parametrize("case", sorted(CASES))
@pytest.mark.parametrize("bound_type", TYPES)
def test_pointwise_interval_matches_r(case, bound_type):
    name, fitter, alpha = CASES[case]
    ref = values("r_km_ci", case)
    if "lower_" + bound_type not in ref:
        pytest.skip("km.ci bounds only the Kaplan-Meier estimate")
    model = _model(name, fitter)
    t = ref["time"]
    assert_allclose(model.sf(t), ref["surv"], rtol=1e-10)
    got = model.cb(t, on="sf", alpha_ci=alpha, bound_type=bound_type)
    if bound_type == "normal":
        got = np.clip(got, 0.0, 1.0)
    for k, side in enumerate(("lower", "upper")):
        want = ref[f"{side}_{bound_type}"]
        ok = np.isfinite(want)
        assert ok.sum() >= 10
        assert_allclose(got[ok, k], want[ok], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("bound_type", ["log", "logit", "arcsine", "rothman"])
def test_one_sided_bound_is_the_end_of_the_two_sided_at_twice_alpha(
    bound_type,
):
    model = _model("aml", "KaplanMeier")
    t = values("r_km_ci", "aml_km_90")["time"][:-2]
    two = model.cb(t, alpha_ci=0.1, bound_type=bound_type)
    assert_allclose(
        model.cb(t, alpha_ci=0.05, bound="lower", bound_type=bound_type),
        two[:, 0],
        rtol=1e-12,
    )
    assert_allclose(
        model.cb(t, alpha_ci=0.05, bound="upper", bound_type=bound_type),
        two[:, 1],
        rtol=1e-12,
    )
