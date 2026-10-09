"""Properties of the regression models on generated data (#379).

Data: exact and right censored times with ties and counts, one or two
numeric covariates on a coarse grid, sometimes a constant column (which
Cox, having no intercept, aliases, #476: it is left out of Cox's data),
and, for the
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
from hypothesis import assume, example, given
from hypothesis import strategies as st

import surpyval as sp
from surpyval.tests.conformance.checks import compare, expanded, permuted
from surpyval.tests.conformance.registry import predictions
from surpyval.tests.properties import strategies as gen
from surpyval.tests.properties.common import case_for, outcome, quietly

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
        model = _fit("CoxPH", data)
    except ValueError:
        # Cox has no coefficient for a column that does not vary within
        # the risk sets (#409, #476), and nothing runs off.
        return False
    if model.maximum == "no finite maximum":
        # Said so by the fit, also where an aliased column's coefficient
        # is nan and the risk scores below would be nan (#714).
        return True
    beta = np.nan_to_num(np.asarray(model.beta, dtype=float))
    return bool(np.max(np.abs(data["Z"] @ beta)) > 8)


def _refused(message):
    """Whether ``message`` is the documented refusal (#463) of a fit whose
    baseline at Z = 0 over- or underflows. On the generated data it comes
    with a likelihood that has no finite maximum, where the coefficients
    or the baseline's shape run off and the baseline moved to Z = 0 with
    them (#714); where the search sees the runaway, the refusal says so."""
    return "cannot be represented" in message and "center=True" in message


def _fit_both(name, d, other):
    """The fits of ``d`` and of ``other`` (the same data rearranged), where
    a comparison of the two applies: neither refused (as above; whether
    a refusal is reached depends on where the search stopped on its way
    off, so one order may be refused and the other not) nor without a
    finite maximum (:func:`_runs_off`)."""
    fitter = getattr(sp, name).fit
    fits = [outcome(fitter, **_columns(e)) for e in (d, other)]
    refused = [model for status, model in fits if status != "ok"]
    for message in refused:
        assert _refused(message), message
    assume(not refused)
    ref, got = (model for _, model in fits)
    assume(not _runs_off(ref, got))
    return ref, got


def _columns(d):
    return {k: d[k] for k in ("x", "Z", "c", "n") if k in d}


def _runs_off(ref, got):
    """Whether either fit ``ref`` or ``got`` of one data set found that
    its likelihood has no finite maximum (and warned so). Their parameters
    are then wherever the optimiser stopped on the way to a limit, and
    need not agree: on exact 13 and 9 (two each) and censored 4.5 and 8.0
    (n 2), whose Cox fit separates, WeibullPH's shape ran to 567, and its
    hazard far out differed by 0.8% between counts and repeated rows
    (#714). Nor need the verdicts: the runaway check reads the search's
    last step, and on a level with only censored rows (exact 1.0, n 2, and
    0.5 at level a; censored 0.5 at level c) the level's coefficient
    stopped at -8.62 ("no finite maximum") or -8.60 ("unverified", which
    warns too) by the row order. (A parametric fit now runs on its rows
    in one order, so another row order gives the same fit, #728; counts
    and repeated rows still differ.)"""
    return any(
        getattr(m, "maximum", None) == "no finite maximum" for m in (ref, got)
    )


def _for_cox(data, Z):
    """The data and query rows without the constant column, which Cox
    (no intercept) aliases (#476)."""
    if not _constant(data):
        return data, Z
    return {**data, "Z": data["Z"][:, :-1]}, Z[:, :-1]


def _prepared(name, data, Z):
    """``(data, Z)`` for model ``name``: for Cox without the constant
    column, and assumed to be data Cox can fit."""
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
    ref_model, got_model = _fit_both(name, d, permuted(case, d, perm))
    ref = predictions(case, ref_model, x=x, Z=Z)
    got = predictions(case, got_model, x=x, Z=Z)
    compare(case, got, ref, atol=1e-8)


@pytest.mark.parametrize("name", MODELS)
@given(data=_with_query())
# No finite maximum (#714): the fits stop at different points.
@example(
    data=(
        dict(
            x=np.array([13.0, 9.0, 4.5, 8.0]),
            Z=np.array(
                [
                    [-1.5, 2.0, 1.0],
                    [2.0, -1.5, 1.0],
                    [0, 1.0, 1],
                    [1.5, -1.5, 1],
                ]
            ),
            c=np.array([0, 0, 1, 1]),
            n=np.array([2, 2, 1, 2]),
        ),
        np.array([[0.5, -1.5, 1.0], [-1.5, 0, 1]] + [[0.0, 0.0, 1.0]] * 4),
    )
)
def test_counts_equal_repeated_rows(name, data):
    d, Z = _prepared(name, *data)
    assume(np.any(d["n"] > 1) and not _separated(d))
    case = case_for(name, d, rtol=RTOL[name])
    x = _query(d, Z)
    ref_model, got_model = _fit_both(name, d, expanded(case, d))
    ref = predictions(case, ref_model, x=x, Z=Z)
    got = predictions(case, got_model, x=x, Z=Z)
    compare(case, got, ref, atol=1e-8)


@pytest.mark.parametrize("name", MODELS)
@given(data=_with_query())
# Refused by WeibullPH (#714): exact 0.5 at z = 1.5 and 1.0 twice at 0.
@example(
    data=(
        dict(
            x=np.array([0.5, 1.0, 0.5, 1.0]),
            Z=np.array([[1.5], [0.0], [0.0], [0.0]]),
            c=np.array([0, 0, 1, 0]),
            n=np.array([1, 1, 1, 1]),
        ),
        np.zeros((6, 1)),
    )
)
def test_rows_are_independent(name, data):
    d, Z = _prepared(name, *data)
    status, model = outcome(getattr(sp, name).fit, **_columns(d))
    if status != "ok":
        # This property is about predictions, and a refused fit has none.
        # On the shrunk examples the refusal was warranted: the likelihood
        # had no finite maximum (Cox separated the data, and the Weibull
        # shape ran to 1365 for WeibullPH, 1e15 for WeibullAFT, #714).
        assert _refused(model), model
        assume(False)
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


def _cox_ran_off(model):
    """Whether the Cox fit ``model`` (``None`` for a refusal) ran off, as
    it says. It used to miss a run along a combination of coefficients
    (on exact 0.5 at level b (z 0.5), exact 1.0 and 0.5 (z -1, -1, -1) and
    censored 0.5 (z 0.5, 0.5) at level a it stopped at (-24.7, 37.6),
    "unverified", or (-23.1, 35.2), "verified", by the row order, #714),
    and a coefficient beyond 4 was taken as a run-off too; the data now
    decide (#728, #746)."""
    return model is not None and model.maximum == "no finite maximum"


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

    def fit(frame):
        try:
            return quietly(fitter.fit_from_df, frame, **kw)
        except ValueError as e:
            # A parametric fit refuses a baseline at Z = 0 it cannot
            # represent, as when separated data send the coefficients off
            # (#463); whether it is reached depends on where the search
            # stopped on its way off, so in one row order only (#714).
            assume(name != "CoxPH" and not _refused(str(e)))
            raise

    # Data Cox separates have, as a rule, no finite maximum for the
    # parametric fit either (below), and such a fit takes seconds (the
    # search runs off, and the runaway check follows it): set them aside
    # first, with a Cox fit of a hundredth of the time.
    try:
        cox = quietly(sp.CoxPH.fit_from_df, df, **kw)
    except ValueError:
        cox = None
    assume(not _cox_ran_off(cox))
    ref = fit(df)
    got = fit(df.iloc[perm].reset_index(drop=True))
    # On data with no finite maximum the fits stop wherever the search
    # gave up: 1.0 (n 2) and 0.5 at z -0.5, censored 0.5 at z -0.5 and -2
    # (n 3), sf at z = 0 of 3.1e-7 or 1.3e-3 by the row order (#714).
    assume(not _runs_off(ref, got))
    query = pd.DataFrame({"z0": [0.0, 1.0, -1.0], "g": ["a", "b", "c"]})
    query = query[query["g"].isin(set(d["g"]))]
    x = np.linspace(gen.STEP, np.max(d["x"]), len(query))
    np.testing.assert_allclose(
        got.sf(x, query), ref.sf(x, query), rtol=RTOL[name], atol=1e-8
    )
