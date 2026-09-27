"""
Formula prediction (#370, #371).

#371: a categorical level a formula model was not fitted with used to be
coded silently as the reference level (only formulaic's
``DataMismatchWarning`` said so); it now raises a ``ValueError`` naming the
column and the level, in every family that takes a ``formula``, before and
after ``to_dict`` -> JSON -> ``from_dict``. Levels declared with
``C(g, levels=[...])`` count as seen. A missing categorical value is not a
level: its row predicts ``nan``, in place, as a missing numeric does.

#370: the competing-risks Cox model predicts from a DataFrame of raw
covariates (the formula applied, the columns read by name), as ``CoxPH``
does; it used to read a DataFrame by column position and could not expand
a formula's categoricals.
"""

import json

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import AcceleratedLife, Linear, Weibull
from surpyval.univariate.competing_risks.regression import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.regression.regression_data import prepare_Z

CR = CompetingRisksProportionalHazards


def _df(seed=3, n=300):
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b", "c"], n)
    k = rng.integers(1, 4, n)
    z = rng.uniform(1, 3, n)
    eff = 0.4 * z + np.select([g == "b", g == "c"], [0.5, -0.4], 0.0)
    t = 10 * rng.weibull(1.5, n) * np.exp(-eff / 1.5)
    c = (rng.uniform(size=n) < 0.2).astype(int)
    return pd.DataFrame(
        {
            "t": t,
            "c": c,
            "z": z,
            "w": rng.uniform(0, 1, n),
            "g": g,
            "k": k,
            "cause": np.where(c == 1, None, rng.choice(["u", "v"], n)),
            "unit": rng.integers(0, 40, n),
        }
    )


FAMILIES = [
    "WeibullPH",
    "WeibullAFT",
    "WeibullPO",
    "WeibullAH",
    "AL",
    "CoxPH",
    "AdditiveHazards",
    "BuckleyJames",
    "WeibullFrailty",
    "CR-Cox",
    "CR-Fine-Gray",
]


def _fit(family, formula, df):
    kw = dict(x_col="t", c_col="c", formula=formula)
    if family == "WeibullFrailty":
        return surpyval.WeibullFrailty.fit_from_df(df, group_col="unit", **kw)
    if family == "AL":
        # a single-stress life model: 'g' with two levels, 'a' and 'b'
        kw["formula"] = formula.replace("z + ", "")
        return AcceleratedLife(Weibull, Linear).fit_from_df(
            df[df["g"] != "c"], **kw
        )
    if family.startswith("CR-"):
        how = family[3:]
        return CR.fit_from_df(
            df, "t", "cause", c_col="c", formula=formula, how=how
        )
    return getattr(surpyval, family).fit_from_df(df, **kw)


def _rt(model):
    text = json.dumps(model.to_dict(), allow_nan=False)
    return surpyval.from_dict(json.loads(text))


def _sf(model, family, frame, t=5.0):
    kw = {"event": "u"} if family.startswith("CR-") else {}
    if family == "BuckleyJames":
        # Buckley-James predicts one covariate vector at a time
        return np.concatenate(
            [model.sf([t], frame.iloc[[i]]) for i in range(len(frame))]
        )
    return np.asarray(
        model.sf(np.full(len(frame), t), frame, **kw), dtype=float
    ).ravel()


@pytest.fixture(scope="module")
def fitted():
    df = _df()
    out = {}
    for family in FAMILIES:
        model = _fit(family, "z + g", df)
        out[family] = (model, _rt(model))
    return out


# -- #371: unseen levels --------------------------------------------------


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("family", FAMILIES)
def test_unseen_level_raises(fitted, family, restored):
    model = fitted[family][restored]
    seen = pd.DataFrame({"z": [2.0, 2.0], "g": ["a", "b"]})
    assert np.isfinite(_sf(model, family, seen)).all()
    # 'd' used to be predicted exactly as the reference level 'a'
    unseen = pd.DataFrame({"z": [2.0, 2.0], "g": ["a", "d"]})
    with pytest.raises(ValueError, match=r"column 'g'.*\['d'\]"):
        _sf(model, family, unseen)


@pytest.mark.parametrize("family", ["WeibullPH", "CoxPH", "CR-Cox"])
def test_unseen_level_error_names_every_column(family):
    df = _df()
    model = _fit(family, "z + C(g, contr.sum) + C(k)", df)
    new = pd.DataFrame({"z": [2.0, 2.0], "g": ["e", "x"], "k": [1, 7]})
    for m in (model, _rt(model)):
        with pytest.raises(ValueError) as err:
            _sf(m, family, new)
        text = str(err.value)
        assert "column 'g'" in text and "['e', 'x']" in text
        # integer levels stay integers, in the message too
        assert "column 'k'" in text and "[7]" in text


def test_unseen_level_old_behaviour_was_the_reference_level():
    # The numbers of the issue: 'd' gave the reference level's survival.
    df = _df()
    model = surpyval.WeibullPH.fit_from_df(
        df, x_col="t", c_col="c", formula="z + g"
    )
    ref = model.sf([5.0], pd.DataFrame({"z": [2.0], "g": ["a"]}))
    # the reference level's row of the design matrix is all zeros...
    Z_ref = prepare_Z(
        pd.DataFrame({"z": [2.0], "g": ["a"]}),
        model.feature_names,
        model._model_spec,
    )
    np.testing.assert_array_equal(Z_ref, [[2.0, 0.0, 0.0]])
    np.testing.assert_allclose(model.sf([5.0], Z_ref), ref)
    # ...and an unknown level no longer gets that row
    with pytest.raises(ValueError, match="Unknown categorical level"):
        prepare_Z(
            pd.DataFrame({"z": [2.0], "g": ["d"]}),
            model.feature_names,
            model._model_spec,
        )


@pytest.mark.parametrize("family", ["WeibullPH", "CoxPH", "CR-Cox"])
def test_declared_levels_count_as_seen(family):
    df = _df()
    formula = "z + C(g, levels=['a', 'b', 'c', 'd'])"
    model = _fit(family, formula, df)
    assert sum("[T.d]" in name for name in model.feature_names) == 1
    new = pd.DataFrame({"z": [2.0], "g": ["d"]})
    for m in (model, _rt(model)):
        assert np.isfinite(_sf(m, family, new)).all()
        with pytest.raises(ValueError, match=r"column 'g'.*\['e'\]"):
            _sf(m, family, pd.DataFrame({"z": [2.0], "g": ["e"]}))


@pytest.mark.parametrize("family", ["WeibullPH", "CoxPH", "CR-Cox"])
def test_fit_rejects_data_outside_declared_levels(family):
    # The same coding at fit time: rows of level 'c' were fitted as 'a'.
    df = _df()
    with pytest.raises(ValueError, match=r"column 'g'.*\['c'\]"):
        _fit(family, "z + C(g, levels=['a', 'b'])", df)


def test_computed_categorical_term_is_named():
    # A term that does not code a column as is cannot be checked by
    # column; formulaic's detail is quoted with the term.
    df = _df()
    model = surpyval.WeibullPH.fit_from_df(
        df, x_col="t", c_col="c", formula="z + C(k + 1)"
    )
    new = pd.DataFrame({"z": [2.0], "k": [7]})
    for m in (model, _rt(model)):
        with pytest.raises(ValueError, match=r"C\(k \+ 1\).*8"):
            m.sf([5.0], new)


@pytest.mark.parametrize("family", FAMILIES)
def test_numeric_columns_are_not_checked(fitted, family):
    # A numeric value far outside the fitted range is still predicted.
    model = fitted[family][0]
    new = pd.DataFrame({"z": [2.0, 50.0], "g": ["b", "b"]})
    assert _sf(model, family, new).shape == (2,)


# -- #371: missing categorical values -------------------------------------


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("missing", [None, np.nan])
def test_missing_categorical_predicts_nan_in_place(
    fitted, family, restored, missing
):
    model = fitted[family][restored]
    full = pd.DataFrame({"z": [2.0, 2.5, 1.5], "g": ["b", "a", "a"]})
    gap = full.assign(g=pd.Series(["b", missing, "a"], dtype=object))
    expected = _sf(model, family, full)
    out = _sf(model, family, gap)
    # Buckley-James used to give survival 0 for the missing row
    assert np.isnan(out[1])
    np.testing.assert_allclose(out[[0, 2]], expected[[0, 2]], rtol=1e-12)


def test_missing_value_in_a_categorical_dtype_column():
    df = _df()
    model = surpyval.CoxPH.fit_from_df(
        df, x_col="t", c_col="c", formula="z + g"
    )
    g = pd.Categorical(["b", None, "c"], categories=["a", "b", "c"])
    out = model.sf(np.full(3, 5.0), pd.DataFrame({"z": [2.0] * 3, "g": g}))
    assert np.isnan(out[1]) and np.isfinite(out[[0, 2]]).all()


def test_buckley_james_missing_numeric_covariate_is_nan():
    df = _df()
    model = surpyval.BuckleyJames.fit_from_df(
        df, x_col="t", c_col="c", Z_cols=["z", "w"]
    )
    out = model.sf([1.0, 5.0], [2.0, np.nan])
    assert np.isnan(out).all()
    assert np.isnan(model.ff([5.0], [np.nan, 0.5])).all()
    assert np.isfinite(model.sf([1.0, 5.0], [2.0, 0.5])).all()


# -- #370: competing-risks Cox predicts from a DataFrame ------------------

NEW = pd.DataFrame(
    {
        "z": [1.2, 2.5, 2.9, 1.7],
        "w": [0.1, 0.9, 0.4, 0.6],
        "g": ["b", "a", "c", "b"],
        "k": [2, 1, 3, 3],
    }
)
T = np.array([2.0, 5.0, 8.0, 12.0])


def _methods(how):
    if how == "Cox":
        return ["sf", "ff", "Hf", "hf", "df"]
    return ["sf", "ff", "Hf"]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
@pytest.mark.parametrize(
    "formula", ["z + g", "0 + C(g) + poly(z, 2)", "z * g + C(k)"]
)
def test_competing_risks_formula_predicts_from_dataframe(
    how, formula, restored
):
    model = CR.fit_from_df(
        _df(), "t", "cause", c_col="c", formula=formula, how=how
    )
    if restored:
        model = _rt(model)
    Z = prepare_Z(NEW, model.feature_names, model._model_spec)
    assert Z.shape == (4, len(model.feature_names))
    for method in _methods(how):
        for event in ["u", "v"] + ([None] if how == "Cox" else []):
            f = getattr(model, method)
            np.testing.assert_allclose(
                f(T, NEW, event=event), f(T, Z, event=event), rtol=1e-12
            )
    for event in ["u", "v"]:
        np.testing.assert_allclose(
            model.cif(T, NEW, event), model.cif(T, Z, event), rtol=1e-12
        )
        # one covariate row for every time
        np.testing.assert_allclose(
            model.cif(T, NEW.iloc[[1]], event),
            model.cif(T, Z[1], event),
            rtol=1e-12,
        )
    np.testing.assert_allclose(model.phi(NEW), model.phi(Z), rtol=1e-12)
    np.testing.assert_allclose(
        model.phi_e(NEW, 1), model.phi_e(Z, 1), rtol=1e-12
    )


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_competing_risks_dataframe_is_read_by_column_name(how):
    model = CR.fit_from_df(
        _df(), "t", "cause", c_col="c", Z_cols=["z", "w"], how=how
    )
    arr = NEW[["z", "w"]].to_numpy()
    expected = model.cif(T, arr, "u")
    # reordered, with an extra column: it used to be read by position
    shuffled = NEW[["k", "w", "z"]]
    for m in (model, _rt(model)):
        np.testing.assert_allclose(
            m.cif(T, shuffled, "u"), expected, rtol=1e-12
        )
        np.testing.assert_allclose(
            m.sf(T, shuffled, event="u"),
            m.sf(T, arr, event="u"),
            rtol=1e-12,
        )


def test_competing_risks_matches_cox_ph_on_a_formula():
    # A single cause: the cause-specific model is CoxPH itself.
    df = _df()
    df["one"] = np.where(df["c"] == 1, None, "u")
    cr = CR.fit_from_df(df, "t", "one", c_col="c", formula="z + g")
    cox = surpyval.CoxPH.fit_from_df(df, x_col="t", c_col="c", formula="z + g")
    np.testing.assert_allclose(cr.betas[0], cox.beta, rtol=1e-6)
    np.testing.assert_allclose(
        cr.sf(T, NEW, event="u"), cox.sf(T, NEW), rtol=1e-6
    )


def test_competing_risks_array_fit_refuses_a_dataframe():
    df = _df()
    model = CR.fit(
        df["t"].to_numpy(),
        df[["z", "w"]].to_numpy(),
        df["cause"].to_numpy(),
        c=df["c"].to_numpy(),
    )
    with pytest.raises(ValueError, match="named covariates"):
        model.sf(T, NEW[["z", "w"]], event="u")
    # the array interface is unchanged
    assert model.sf(T, NEW[["z", "w"]].to_numpy(), event="u").shape == (4,)
