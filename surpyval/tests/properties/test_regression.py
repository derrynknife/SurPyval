"""Properties of the regression models on generated data (#379).

Data: exact and right censored times with ties and counts, one or two
numeric covariates on a coarse grid, sometimes a constant column (which a
parametric model folds into its scale, and which Cox, having no
intercept, refuses: it is left out of Cox's data, #409), and, for the
formula path, a categorical label.

- **row order**: permuting the rows does not change the predictions;
- **counts**: a count ``n`` predicts what that many repeated rows do;
- **row independence**: covariate rows predicted together give what
  each gives alone, and one vector broadcasts over every time;
- the formula path with a categorical column is invariant to row order
  too (the levels, and so the reference level, must not depend on the
  order the rows arrive in).
"""

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

import surpyval as sp
from surpyval.tests.conformance.checks import compare, expanded, permuted
from surpyval.tests.conformance.registry import predictions
from surpyval.tests.properties import strategies as gen
from surpyval.tests.properties.common import case_for, quietly

MODELS: tuple[str, ...] = ("CoxPH", "WeibullPH")
if gen.THOROUGH:
    MODELS += ("WeibullAFT",)
# Cox is fitted by Newton's method to 1e-10; the parametric models by a
# general optimiser (see test_parametric's module docstring).
RTOL = {"CoxPH": 1e-6, "WeibullPH": 1e-3, "WeibullAFT": 1e-3}


def _fit(name, data):
    keys = ("x", "Z", "c", "n")
    return quietly(
        getattr(sp, name).fit, **{k: data[k] for k in keys if k in data}
    )


def _separated(data):
    """Whether the data (nearly) separate: the Cox coefficients run off
    to infinity (monotone partial likelihood) and a fit is wherever its
    optimiser stopped. A property comparing two fits does not apply."""
    data, _ = _for_cox(data, data["Z"])
    try:
        beta = np.asarray(_fit("CoxPH", data).beta, dtype=float)
    except ValueError:
        # A column that does not vary within the risk sets: Cox has no
        # coefficient for it (#409), and nothing runs off.
        return False
    return bool(np.max(np.abs(data["Z"] @ beta)) > 8)


def _for_cox(data, Z):
    """The data and query rows without the constant column, which Cox
    (no intercept) refuses (#409)."""
    if not _constant(data):
        return data, Z
    return {**data, "Z": data["Z"][:, :-1]}, Z[:, :-1]


def _prepared(name, data, Z):
    """``(data, Z)`` for model ``name``: for Cox without the constant
    column, and assumed to be data Cox can fit -- a generated column can
    still be constant within every risk set, and Cox refuses that."""
    if name != "CoxPH":
        return data, Z
    data, Z = _for_cox(data, Z)
    try:
        _fit(name, data)
    except ValueError:
        assume(False)
    return data, Z


def _query(data, Z):
    """Times spread over the data, one per row of ``Z``."""
    top = np.max(data["x"])
    return np.linspace(0.25 * gen.STEP, top + gen.STEP, len(Z))


@st.composite
def _with_query(draw, **kw):
    d = draw(gen.regression(**kw))
    Z = draw(gen.query_rows(d["Z"].shape[1] - _constant(d), size=6))
    if _constant(d):
        Z = np.column_stack([Z, np.ones(len(Z))])
    return d, Z


def _constant(d):
    return int(d["Z"].shape[1] > 1 and np.all(d["Z"][:, -1] == 1.0))


@pytest.mark.parametrize("name", MODELS)
@given(data=st.data())
def test_row_order(name, data):
    d, Z = _prepared(name, *data.draw(_with_query(), label="data, Z"))
    assume(not _separated(d))
    perm = data.draw(gen.permutations(len(d["x"])), label="perm")
    case = case_for(name, d, rtol=RTOL[name])
    x = _query(d, Z)
    ref = predictions(case, _fit(name, d), x=x, Z=Z)
    got = predictions(case, _fit(name, permuted(case, d, perm)), x=x, Z=Z)
    compare(case, got, ref, atol=1e-8)


@pytest.mark.parametrize("name", MODELS)
@given(data=_with_query())
def test_counts_equal_repeated_rows(name, data):
    d, Z = _prepared(name, *data)
    assume(np.any(d["n"] > 1) and not _separated(d))
    case = case_for(name, d, rtol=RTOL[name])
    x = _query(d, Z)
    ref = predictions(case, _fit(name, d), x=x, Z=Z)
    got = predictions(case, _fit(name, expanded(case, d)), x=x, Z=Z)
    compare(case, got, ref, atol=1e-8)


@pytest.mark.parametrize("name", MODELS)
@given(data=_with_query())
def test_rows_are_independent(name, data):
    d, Z = _prepared(name, *data)
    model = _fit(name, d)
    x = _query(d, Z)
    jumps = ("hf", "df") if name == "CoxPH" else ()
    for fname in ("sf", "ff", "Hf", "hf", "df"):
        if fname in jumps:
            continue
        f = getattr(model, fname)
        together = np.asarray(f(x, Z), float)
        alone = np.array(
            [
                np.asarray(f(x[k : k + 1], Z[k : k + 1]), float).item()
                for k in range(len(x))
            ]
        )
        np.testing.assert_allclose(together, alone, rtol=1e-12, err_msg=fname)
        one = np.asarray(f(x, Z[1]), float)
        tiled = np.asarray(f(x, np.tile(Z[1], (len(x), 1))), float)
        np.testing.assert_allclose(one, tiled, rtol=1e-12, err_msg=fname)


def _frame(d):
    df = pd.DataFrame(d["Z"][:, :1], columns=["z0"])
    df["g"] = d["g"]
    df["x"], df["c"], df["n"] = d["x"], d["c"], d["n"]
    return df


@pytest.mark.parametrize("name", ("CoxPH", "WeibullPH"))
@given(data=st.data())
def test_formula_row_order(name, data):
    d = data.draw(
        gen.regression(columns=(1,), constant_column=False, categorical=True),
        label="data",
    )
    perm = data.draw(gen.permutations(len(d["x"])), label="perm")
    df = _frame(d)
    fitter = getattr(sp, name)
    kw = dict(x_col="x", c_col="c", n_col="n", formula="z0 + C(g)")
    try:
        ref = quietly(fitter.fit_from_df, df, **kw)
    except ValueError:
        # Cox refuses a column constant within every risk set (#409).
        assume(name != "CoxPH")
        raise
    got = quietly(
        fitter.fit_from_df, df.iloc[perm].reset_index(drop=True), **kw
    )
    query = pd.DataFrame({"z0": [0.0, 1.0, -1.0], "g": ["a", "b", "c"]})
    query = query[query["g"].isin(set(d["g"]))]
    x = np.linspace(gen.STEP, np.max(d["x"]), len(query))
    np.testing.assert_allclose(
        got.sf(x, query), ref.sf(x, query), rtol=RTOL[name], atol=1e-8
    )
