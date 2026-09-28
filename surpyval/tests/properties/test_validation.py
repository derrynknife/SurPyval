"""Invalid input is refused with a ``ValueError`` (#379).

Each generated data set is valid exact / right censored data with one
defect (``strategies.INVALID_KINDS``): a negative, zero or fractional
count, a left truncation time at or above its value, a right truncation
time below it, arrays of different lengths, a NaN time, an infinite
exactly observed time or an unknown censoring flag. Every fitter that
takes such data must raise ``ValueError`` -- not another exception type,
and not a model fitted to garbage.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

import surpyval as sp
from surpyval import recurrent as rc
from surpyval.tests.properties import strategies as gen
from surpyval.tests.properties.common import quietly
from surpyval.univariate import competing_risks as cr

UNIVARIATE: tuple[str, ...] = ("Weibull", "Exponential", "LogNormal", "Normal")
UNIVARIATE += ("KaplanMeier", "NelsonAalen", "Turnbull")


def _refuses(fit, **data):
    with pytest.raises(ValueError):
        quietly(fit, **data)


@pytest.mark.parametrize("name", UNIVARIATE)
@given(case=gen.invalid_xcnt())
def test_univariate_refuses_invalid_data(name, case):
    _, data = case
    _refuses(getattr(sp, name).fit, **data)


# Cox takes no right truncation. An infinite exact time is accepted, a
# known failure pinned in test_known_failures.py.
_COX_KINDS = tuple(
    k for k in gen.INVALID_KINDS if k not in ("tr below x", "inf exact time")
)


@given(case=gen.invalid_xcnt(kinds=_COX_KINDS))
def test_cox_refuses_invalid_data(case):
    _, data = case
    Z = np.linspace(-1.0, 1.0, len(data["x"]))[:, None]
    _refuses(sp.CoxPH.fit, Z=Z, **data)


@given(data=gen.xcnt(censoring=gen.RIGHT_CENSORING, min_rows=2))
def test_cox_refuses_mismatched_covariates(data):
    data = {k: v for k, v in data.items() if k in ("x", "c", "n")}
    Z = np.linspace(-1.0, 1.0, len(data["x"]) - 1)[:, None]
    _refuses(sp.CoxPH.fit, Z=Z, **data)


@given(data=gen.competing_risks(min_rows=2), defect=st.integers(0, 3))
def test_competing_risks_refuses_invalid_data(data, defect):
    x, e, n = data["x"].copy(), data["e"], data["n"].copy()
    if defect == 0:
        n[0] = -1
    elif defect == 1:
        x[0] = np.nan
    elif defect == 2:
        e = e[:-1]
    else:
        n = n[:-1]
    _refuses(cr.CompetingRisks.fit, x=x, e=e, n=n)


@given(data=gen.xicn(), defect=st.integers(0, 2))
def test_recurrent_refuses_invalid_data(data, defect):
    x, i, c = data["x"].copy(), data["i"], data["c"]
    if defect == 0:
        x[0] = np.nan
    elif defect == 1:
        i = i[:-1]
    else:
        c = c[:-1]
    _refuses(rc.NonParametricCounting.fit, x=x, i=i, c=c)
