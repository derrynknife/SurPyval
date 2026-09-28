"""Properties of the competing-risks and recurrent-event estimators on
generated data (#379).

Competing risks (``x, e, n``; ``e`` a cause or ``None`` for censored):

- the cumulative incidences are valid: in [0, 1], non-decreasing, and
  summing to at most 1; under the Kaplan-Meier method they sum to
  ``1 - sf`` exactly (the Aalen-Johansen identity);
- the fit does not depend on the row order or on counts versus
  repeated rows;
- competing-risks Cox (one covariate) does not depend on the row order
  (unsorted input broke it once).

Recurrent events (``x, i, c``; per item, events and an end of
observation):

- the mean cumulative function is non-negative and non-decreasing;
- it does not depend on the row order or on the items' labels.
"""

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from surpyval import recurrent as rc
from surpyval.tests.conformance.checks import (
    RULES,
    check_valid,
    compare,
    expanded,
    permuted,
)
from surpyval.tests.conformance.registry import predictions
from surpyval.tests.properties import strategies as gen
from surpyval.tests.properties.common import (
    case_for,
    outcome,
    query_points,
    quietly,
)
from surpyval.univariate import competing_risks as cr

METHODS = ("Nelson-Aalen", "Kaplan-Meier")


def _cr_case(method, d):
    causes = tuple(sorted({e for e in d["e"] if e is not None}))
    return case_for(f"CompetingRisks[{method}]", d, events=causes)


def _cr_fit(method, d):
    return quietly(cr.CompetingRisks.fit, **d, method=method)


@pytest.mark.parametrize("method", METHODS)
@given(d=gen.competing_risks())
def test_cumulative_incidence_is_valid(method, d):
    case = _cr_case(method, d)
    model = _cr_fit(method, d)
    x = query_points(d)
    total = np.zeros(x.size)
    for e in case.events:
        cif = np.asarray(model.cif(x, e), float)
        check_valid(f"cif[{e}]", cif, RULES["cif"])
        total += cif
    assert np.all(total <= 1 + 1e-10), total
    check_valid("sf", model.sf(x), RULES["sf"])
    if method == "Kaplan-Meier":
        np.testing.assert_allclose(total, 1 - model.sf(x), atol=1e-10)


@pytest.mark.parametrize("method", METHODS)
@given(data=st.data())
def test_competing_risks_row_order(method, data):
    d = data.draw(gen.competing_risks(), label="data")
    perm = data.draw(gen.permutations(len(d["x"])), label="perm")
    case = _cr_case(method, d)
    x = query_points(d)
    ref = predictions(case, _cr_fit(method, d), x=x)
    got = predictions(case, _cr_fit(method, permuted(case, d, perm)), x=x)
    compare(case, got, ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
@given(d=gen.competing_risks())
def test_competing_risks_counts(method, d):
    case = _cr_case(method, d)
    x = query_points(d)
    ref = predictions(case, _cr_fit(method, d), x=x)
    got = predictions(case, _cr_fit(method, expanded(case, d)), x=x)
    compare(case, got, ref, rtol=1e-9, atol=1e-12)


@given(data=st.data())
def test_competing_risks_cox_row_order(data):
    d = data.draw(gen.competing_risks(min_rows=4), label="data")
    z = data.draw(
        st.lists(
            st.sampled_from((0.0, 0.5, 1.0)),
            min_size=len(d["x"]),
            max_size=len(d["x"]),
        ),
        label="z",
    )
    d["Z"] = np.array(z)[:, None]
    perm = data.draw(gen.permutations(len(d["x"])), label="perm")
    causes = sorted({e for e in d["e"] if e is not None})
    fit = cr.CompetingRisksProportionalHazards.fit
    status, ref = outcome(fit, **d)
    status2, got = outcome(
        fit, **{k: np.asarray(v)[perm] for k, v in d.items()}
    )
    # All censored is refused, whatever the order.
    assert status == status2, (ref, got)
    if status != "ok":
        return
    x = query_points(d)
    Z = np.full((x.size, 1), 0.5)
    for e in causes:
        np.testing.assert_allclose(
            got.cif(x, Z, e), ref.cif(x, Z, e), rtol=1e-6, atol=1e-10
        )


def _mcf(d):
    return quietly(rc.NonParametricCounting.fit, **d)


def _mcf_query(d):
    # The MCF is NaN past the last observed time (documented).
    x = query_points(d)
    return x[x <= np.max(d["x"])]


@given(d=gen.xicn())
def test_mcf_is_valid(d):
    mcf = _mcf(d).mcf(_mcf_query(d))
    check_valid("mcf", mcf, RULES["mcf"])


@given(data=st.data())
def test_mcf_row_order_and_labels(data):
    d = data.draw(gen.xicn(), label="data")
    perm = data.draw(gen.permutations(len(d["x"])), label="perm")
    items = np.unique(d["i"])
    relabel = data.draw(gen.permutations(items.size), label="relabel")
    x = _mcf_query(d)
    ref = _mcf(d).mcf(x)
    shuffled = {k: np.asarray(v)[perm] for k, v in d.items()}
    np.testing.assert_allclose(_mcf(shuffled).mcf(x), ref, rtol=1e-12)
    # Items renamed (10, 20, ... in a permuted order).
    names = dict(zip(items, 10 * (relabel + 1)))
    renamed = dict(d, i=np.array([names[i] for i in d["i"]]))
    np.testing.assert_allclose(_mcf(renamed).mcf(x), ref, rtol=1e-12)
