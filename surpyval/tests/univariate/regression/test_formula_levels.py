"""
Empty declared levels (#377) and ``None`` covariates (#375 item 8b).

#377: a level declared with ``C(g, levels=[...])`` (or an unused category of
a ``pd.Categorical`` column) with no rows in the fitted data got a
coefficient with nothing to estimate it from, and a prediction for it was a
made-up number (the reference level's for WeibullPH and CoxPH, whose
coefficient stayed at 0). The fit now warns, naming the column and the
level, and predicting for the level raises the "not fitted with"
``ValueError`` of an unseen level (#371), in every family that takes a
``formula``, before and after ``to_dict`` -> JSON -> ``from_dict``.

#375 8b: covariates given as a list (or object array) holding ``None``
failed with a ``TypeError`` in the parametric PH and AH families (at fit,
after the rows were dropped, and at prediction), in the accelerated-life
fit and in the ``AdditiveHazards`` prediction; ``None`` is now a missing
value, dropped with a warning at fit and predicted as ``nan``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import AcceleratedLife, Weibull
from surpyval.life_models import Linear, Power
from surpyval.serialisation import required_schema
from surpyval.tests._helpers import strict_json_model_round_trip
from surpyval.univariate.competing_risks.regression import (
    CompetingRisksProportionalHazards,
)

CR = CompetingRisksProportionalHazards
LEVELS = "C(g, levels=['a', 'b', 'c', 'd'])"


def _df(seed=3, n=300):
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b", "c"], n)
    z = rng.uniform(1, 3, n)
    eff = 0.4 * z + np.select([g == "b", g == "c"], [0.5, -0.4], 0.0)
    t = 10 * rng.weibull(1.5, n) * np.exp(-eff / 1.5)
    c = (rng.uniform(size=n) < 0.2).astype(int)
    return pd.DataFrame(
        {
            "t": t,
            "c": c,
            "z": z,
            "g": g,
            "cause": np.where(c == 1, None, rng.choice(["u", "v"], n)),
            "unit": rng.integers(0, 40, n),
        }
    )


# Every family that fits a formula to these data (the accelerated-life,
# additive-hazards and Buckley-James fits are below).
FAMILIES = [
    "WeibullPH",
    "WeibullAFT",
    "WeibullPO",
    "WeibullAH",
    "CoxPH",
    "WeibullFrailty",
    "CR-Cox",
    "CR-Fine-Gray",
]


def _fit(family, formula, df, **extra):
    kw = dict(x_col="t", c_col="c", formula=formula, **extra)
    if family == "WeibullFrailty":
        return surpyval.WeibullFrailty.fit_from_df(df, group_col="unit", **kw)
    if family.startswith("CR-"):
        return CR.fit_from_df(
            df, "t", "cause", c_col="c", formula=formula, model=family[3:]
        )
    return getattr(surpyval, family).fit_from_df(df, **kw)


def _fit_quietly(family, formula, df, **extra):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _fit(family, formula, df, **extra)


def _sf(model, family, frame, t=5.0):
    kw = {"event": "u"} if family.startswith("CR-") else {}
    return np.asarray(
        model.sf(np.full(len(frame), t), frame, **kw), dtype=float
    ).ravel()


# -- #377: a declared level with no rows ----------------------------------


@pytest.mark.parametrize("family", FAMILIES)
def test_empty_declared_level_warns_at_fit(family):
    with pytest.warns(UserWarning) as record:
        _fit(family, "z + " + LEVELS, _df())
    messages = [str(w.message) for w in record if "no rows" in str(w.message)]
    # one warning, naming the column, the level and the term
    assert len(messages) == 1
    assert "column 'g'" in messages[0] and "['d']" in messages[0]
    assert LEVELS in messages[0]


@pytest.mark.parametrize("restored", [False, True])
@pytest.mark.parametrize("family", FAMILIES)
def test_empty_declared_level_raises_at_prediction(family, restored):
    model = _fit_quietly(family, "z + " + LEVELS, _df())
    if restored:
        model = strict_json_model_round_trip(model)
    seen = pd.DataFrame({"z": [2.0, 2.0, 2.0], "g": ["a", "b", "c"]})
    assert np.isfinite(_sf(model, family, seen)).all()
    # 'd' used to be predicted (as 'a' for WeibullPH and CoxPH)
    empty = pd.DataFrame({"z": [2.0, 2.0], "g": ["a", "d"]})
    with pytest.raises(ValueError, match="not fitted with") as err:
        _sf(model, family, empty)
    text = str(err.value)
    assert "column 'g' has the level(s) ['d']" in text
    assert "['a', 'b', 'c']" in text and "no rows in the fitted data" in text


def test_empty_declared_level_old_answer_was_the_reference_level():
    # The numbers of the issue: WeibullPH left the coefficient of 'd' at
    # its start value 0, so 'd' predicted exactly as the reference 'a'.
    # Nothing estimates it: it is aliased now (#476), reported as nan and
    # predicted with as 0.
    model = _fit_quietly("WeibullPH", "z + " + LEVELS, _df())
    assert np.isnan(model.params[-1])
    assert model.feature_names[-1] == LEVELS + "[T.d]"
    Z_a = np.array([[2.0, 0.0, 0.0, 0.0]])
    Z_d = np.array([[2.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose(model.sf([5.0], Z_d), model.sf([5.0], Z_a))
    with pytest.raises(ValueError, match="not fitted with"):
        model.sf([5.0], pd.DataFrame({"z": [2.0], "g": ["d"]}))


def test_empty_level_and_unseen_level_in_one_message():
    model = _fit_quietly("WeibullPH", "z + " + LEVELS, _df())
    new = pd.DataFrame({"z": [2.0, 2.0], "g": ["d", "x"]})
    for m in (model, strict_json_model_round_trip(model)):
        with pytest.raises(ValueError) as err:
            m.sf(np.full(2, 5.0), new)
        text = str(err.value)
        assert "['d', 'x']" in text and "(['d'] declared" in text


def test_missing_value_on_an_empty_level_model_is_nan():
    model = _fit_quietly("CoxPH", "z + " + LEVELS, _df())
    new = pd.DataFrame(
        {"z": [2.0, 2.0], "g": pd.Series(["b", None], dtype=object)}
    )
    for m in (model, strict_json_model_round_trip(model)):
        out = m.sf(np.full(2, 5.0), new)
        assert np.isfinite(out[0]) and np.isnan(out[1])


def test_level_present_only_in_dropped_rows_is_empty():
    # The only 'c' rows have a missing 'z': they are dropped from the fit,
    # so the declared 'c' has no rows in the fitted data.
    df = _df()
    df.loc[df["g"] == "c", "z"] = np.nan
    with pytest.warns(UserWarning) as record:
        model = _fit("WeibullPH", "z + " + LEVELS, df)
    messages = " ".join(str(w.message) for w in record)
    assert "Dropped" in messages and "['c', 'd']" in messages
    with pytest.raises(ValueError, match="not fitted with"):
        model.sf([5.0], pd.DataFrame({"z": [2.0], "g": ["c"]}))


@pytest.mark.parametrize("family", ["WeibullPH", "CoxPH", "CR-Cox"])
def test_unused_category_of_a_categorical_column(family):
    # The same for a bare pd.Categorical column with an unused category.
    df = _df()
    df["g"] = pd.Categorical(df["g"], categories=["a", "b", "c", "d"])
    with pytest.warns(UserWarning, match=r"no rows at the level\(s\) \['d'\]"):
        model = _fit(family, "z + g", df)
    for m in (model, strict_json_model_round_trip(model)):
        assert np.isfinite(
            _sf(m, family, pd.DataFrame({"z": [2.0], "g": ["b"]}))
        ).all()
        with pytest.raises(ValueError, match="not fitted with"):
            _sf(m, family, pd.DataFrame({"z": [2.0], "g": ["d"]}))


def test_empty_level_needs_the_schema_2_reader():
    # A bare categorical of string levels is stored with the pair of
    # v0.17 - v0.20 layout too, which a v0.20 reader rebuilds -- without
    # the empty level, which it would predict for. With an empty level the
    # pair is left out, so the document needs schema 2, which v0.20
    # refuses with a request to upgrade.
    df = _df()
    plain = _fit_quietly("WeibullPH", "z + g", df).to_dict()
    assert "factor_levels" in plain["formula_meta"]
    assert plain["schema"] == 1
    df["g"] = pd.Categorical(df["g"], categories=["a", "b", "c", "d"])
    model = _fit_quietly("WeibullPH", "z + g", df)
    out = model.to_dict()
    assert "factor_levels" not in out["formula_meta"]
    assert out["schema"] == 2 == required_schema(out)
    state = out["formula_meta"]["encoder_state"]["g"]["state"]
    assert state["empty_levels"] == ["d"]


def test_no_warning_when_every_level_has_rows():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = surpyval.CoxPH.fit_from_df(
            _df(),
            x_col="t",
            c_col="c",
            formula="z + C(g, levels=['c', 'b', 'a'])",
        )
    assert np.isfinite(model.sf([5.0], pd.DataFrame({"z": [2.0], "g": ["a"]})))


def test_accelerated_life_empty_level():
    # A single-stress life model: 'g' is 'a' in every row, 'd' is empty.
    df = _df()
    df = df[df["g"] == "a"]
    with pytest.warns(UserWarning, match=r"no rows at the level\(s\) \['d'\]"):
        model = AcceleratedLife(Weibull, Linear).fit_from_df(
            df,
            x_col="t",
            c_col="c",
            formula="C(g, levels=['a', 'd'])",
            init=[1.0, 1.5, 10.0, 1.0],
        )
    for m in (model, strict_json_model_round_trip(model)):
        assert np.isfinite(m.sf([5.0], pd.DataFrame({"g": ["a"]}))).all()
        with pytest.raises(ValueError, match="not fitted with"):
            m.sf([5.0], pd.DataFrame({"g": ["d"]}))


@pytest.mark.parametrize("name", ["AdditiveHazards", "BuckleyJames"])
def test_semi_parametric_fits_alias_the_empty_column(name):
    # Lin-Ying and Buckley-James refused the empty level's column, which
    # does not vary; it is aliased now (#476), as in the other fits, and
    # the one warning is the empty level's.
    with pytest.warns(UserWarning) as record:
        model = getattr(surpyval, name).fit_from_df(
            _df(), x_col="t", c_col="c", formula="z + " + LEVELS
        )
    assert len(record) == 1
    assert "no rows at the level(s) ['d']" in str(record[0].message)
    np.testing.assert_array_equal(model.aliased, [3])
    with pytest.raises(ValueError, match="not fitted with"):
        model.sf([5.0], pd.DataFrame({"z": [2.0], "g": ["d"]}))


# -- #375 8b: None in list covariates -------------------------------------


def _arrays(seed=1, n=150):
    rng = np.random.default_rng(seed)
    Z = np.column_stack([rng.normal(size=n), rng.uniform(0, 1, n)])
    T = 10 * rng.weibull(1.5, n) * np.exp(-(0.5 * Z[:, 0] - 0.8 * Z[:, 1]))
    C = rng.uniform(5, 30, n)
    return np.minimum(T, C), (T > C).astype(int), Z


PARAMETRIC = [
    "WeibullPH",
    "ExponentialPH",
    "WeibullAFT",
    "WeibullPO",
    "WeibullAH",
    "LogNormalAH",
]


@pytest.mark.parametrize("name", PARAMETRIC)
def test_none_covariate_is_dropped_at_parametric_fit(name):
    x, c, Z = _arrays()
    Z_list = Z.tolist()
    Z_list[0][0] = None
    Z_list[7][1] = None
    fitter = getattr(surpyval, name)
    # WeibullPH, ExponentialPH and the AH families raised a TypeError
    # after dropping the rows
    with pytest.warns(UserWarning, match="Dropped 2 of 150 rows"):
        model = fitter.fit(x, Z_list, c)
    keep = np.ones(150, dtype=bool)
    keep[[0, 7]] = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        expected = fitter.fit(x[keep], Z[keep], c[keep])
    np.testing.assert_allclose(model.params, expected.params, rtol=1e-6)


@pytest.mark.parametrize("name", PARAMETRIC)
def test_none_covariate_predicts_nan(name):
    x, c, Z = _arrays()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = getattr(surpyval, name).fit(x, Z, c)
    good = model.sf(np.full(2, 5.0), np.array([[0.2, 0.1], [0.3, 0.4]]))
    for rows in (
        [[None, 0.5], [0.2, 0.1]],
        np.array([[None, 0.5], [0.2, 0.1]], dtype=object),
    ):
        out = np.asarray(model.sf(np.full(2, 5.0), rows), dtype=float)
        assert np.isnan(out[0])
        np.testing.assert_allclose(out[1], good[0], rtol=1e-12)
    for method in ("ff", "hf", "Hf", "df"):
        out = getattr(model, method)(
            np.full(2, 5.0), [[0.2, None], [0.2, 0.1]]
        )
        assert np.isnan(out[0]) and np.isfinite(out[1])


def test_none_stress_accelerated_life():
    x, c, Z = _arrays()
    stress = np.exp(Z[:, 0]).tolist()
    stress[3] = None
    model_al = AcceleratedLife(Weibull, Power)
    # 'The axis argument to unique is not supported for dtype object'
    with pytest.warns(UserWarning, match="Dropped 1 of 150 rows"):
        model = model_al.fit(x, stress, c)
    out = model.sf(np.full(2, 5.0), [None, 1.0])
    assert np.isnan(out[0]) and np.isfinite(out[1])


def test_none_covariate_additive_hazards_predicts_nan():
    x, c, Z = _arrays()
    model = surpyval.AdditiveHazards.fit(x, Z, c)
    # a TypeError from ``Z @ beta``
    out = model.sf(np.full(2, 5.0), [[None, 0.5], [0.2, 0.1]])
    expected = model.sf(np.full(2, 5.0), [[0.2, 0.1], [0.2, 0.1]])
    assert np.isnan(out[0])
    np.testing.assert_allclose(out[1], expected[1], rtol=1e-12)
