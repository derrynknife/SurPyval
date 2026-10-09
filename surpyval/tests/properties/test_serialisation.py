"""The serialisation round trip on generated fits (#379).

The conformance suite checks one fitted model per kind; here the fits
are to generated data -- ties, all kinds of censoring, truncation, tiny
samples -- so the stored arrays take shapes and values (a curve that
ends at zero, an infinite cumulative hazard, a one-point ladder) a
fixture does not. Each must survive ``to_dict`` as strict JSON and
``from_dict`` with every prediction unchanged (``checks.check_round_trip``).
"""

import numpy as np
import pytest
from hypothesis import assume, example, given
from hypothesis import strategies as st

import surpyval as sp
from surpyval.tests.conformance.checks import check_round_trip
from surpyval.tests.properties import known
from surpyval.tests.properties import strategies as gen
from surpyval.tests.properties.common import case_for, outcome, query_points

NONPARAMETRIC = ("KaplanMeier", "NelsonAalen", "Turnbull")
PARAMETRIC = (
    ("Weibull", "LogNormal", "Gamma") if gen.THOROUGH else ("Weibull",)
)


@pytest.mark.parametrize("name", NONPARAMETRIC)
@given(data=st.data())
def test_nonparametric(name, data):
    if name == "Turnbull":
        d = data.draw(gen.xcnt(), label="data")
    else:
        d = data.draw(
            gen.xcnt(censoring=gen.RIGHT_CENSORING, right_truncation=False),
            label="data",
        )
    status, model = outcome(getattr(sp, name).fit, **d)
    assert status == "ok", model
    check_round_trip(case_for(name, d), model, x=query_points(d))


@pytest.mark.parametrize("name", PARAMETRIC)
@given(data=gen.xcnt())
def test_parametric(name, data):
    assume(not known.point_mass_supremum(data))
    status, model = outcome(getattr(sp, name).fit, **data)
    assume(status == "ok")
    check_round_trip(case_for(name, data), model, x=query_points(data))


@given(data=gen.regression(constant_column=False))
# Separated data: the coefficients run off, and on a row far beyond the
# data the hazard step and H overflow, so the density was inf * 0 (#714).
@example(
    data=dict(
        x=np.array([2.0, 1.0, 1.5, 0.5, 0.5, 2.0]),
        Z=np.array(
            [[0, 1.5], [0.5, 1], [-1.5, 1], [2, -1], [0, -1.5], [0, 1.5]]
        ),
        c=np.zeros(6, dtype=int),
        n=np.ones(6, dtype=int),
    )
)
def test_cox(data):
    status, model = outcome(
        sp.CoxPH.fit, **{k: data[k] for k in ("x", "Z", "c", "n")}
    )
    # A generated column can be constant within every risk set, which Cox
    # (no intercept) aliases (#476); a refusal is a failure.
    assume(not (status == "ValueError" and "risk set" in model))
    assert status == "ok", model
    x = np.linspace(0.0, np.max(data["x"]) + 1.0, len(data["Z"]))
    check_round_trip(case_for("CoxPH", data), model, x=x, Z=data["Z"])
