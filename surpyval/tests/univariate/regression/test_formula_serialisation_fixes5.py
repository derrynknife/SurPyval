"""
Formula-fit regression models round-trip through ``to_dict`` / JSON /
``from_dict`` for every formula ``fit_from_df`` accepts (#244): wrapped
categoricals (``C(g)``, with ``levels=`` or contrasts), integer levels,
data-dependent transforms (``scale``, ``center``, ``poly``, splines) and
``0 +`` formulas, across the families that take ``formula=``.
"""

import json

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import AcceleratedLife, Power, Weibull
from surpyval.univariate.competing_risks.regression import (
    CompetingRisksProportionalHazards,
)
from surpyval.univariate.regression import regression_data
from surpyval.univariate.regression.regression_data import prepare_Z


def _df(seed=3, n=300):
    rng = np.random.default_rng(seed)
    g = rng.choice(["a", "b", "c"], n)
    h = rng.choice(["x", "y"], n)
    k = rng.integers(1, 4, n)
    z = rng.uniform(1, 3, n)
    eff = 0.4 * z + np.select([g == "b", g == "c"], [0.5, -0.4], 0.0)
    eff = eff + 0.3 * (h == "y") + 0.2 * (k == 3)
    t = 10 * rng.weibull(1.5, n) * np.exp(-eff / 1.5)
    c = (rng.uniform(size=n) < 0.2).astype(int)
    return pd.DataFrame(
        {
            "t": t,
            "c": c,
            "z": z,
            "g": g,
            "h": h,
            "k": k,
            # integer levels, in a non-sorted order (reference level 3)
            "kc": pd.Categorical(k, categories=[3, 1, 2]),
            "cause": np.where(c == 1, None, rng.choice(["u", "v"], n)),
            "unit": rng.integers(0, 40, n),
        }
    )


NEW = pd.DataFrame(
    {
        "z": [1.2, 2.5, 2.9, 1.7],
        "g": ["b", "a", "c", "b"],
        "h": ["y", "x", "y", "x"],
        "k": [2, 1, 3, 3],
        "kc": pd.Categorical([2, 1, 3, 3], categories=[3, 1, 2]),
    }
)
T = np.array([2.0, 5.0, 8.0, 12.0])


def _rt(model):
    # strict JSON both ways, as a database or JavaScript client would see it
    text = json.dumps(model.to_dict(), allow_nan=False)
    return surpyval.from_dict(json.loads(text))


FORMULAS = [
    "z + g",
    "z + C(g)",
    "z + C(g, levels=['c', 'b', 'a'])",
    "z + C(g, contr.treatment('b'))",
    "z + C(g, contr.sum)",
    "z * g",
    "C(h) * z",
    "np.log(z) + I(z ** 2) + g",
    "poly(z, 2) + g",
    "bs(z, df=3)",
    "cs(z, df=3)",
    "scale(z) + center(z ** 2)",
    "z + C(k)",
    "z + kc",
    "0 + z + g",
]


def _fit(family, formula, df):
    kw = dict(x_col="t", c_col="c", formula=formula)
    fitters = {
        "WeibullPH": surpyval.WeibullPH,
        "WeibullAFT": surpyval.WeibullAFT,
        "WeibullPO": surpyval.WeibullPO,
        "WeibullAH": surpyval.WeibullAH,
        "CoxPH": surpyval.CoxPH,
        "AdditiveHazards": surpyval.AdditiveHazards,
        "BuckleyJames": surpyval.BuckleyJames,
    }
    if family == "WeibullFrailty":
        return surpyval.WeibullFrailty.fit_from_df(df, group_col="unit", **kw)
    if family == "AL":
        return AcceleratedLife(Weibull, Power).fit_from_df(df, **kw)
    return fitters[family].fit_from_df(df, **kw)


def _sf(model, family, frame):
    if family == "BuckleyJames":
        # Buckley-James predicts one covariate vector at a time
        return np.concatenate(
            [model.sf(T, frame.iloc[[i]]) for i in range(len(frame))]
        )
    return np.asarray(model.sf(T, frame), dtype=float)


@pytest.mark.parametrize("formula", FORMULAS)
@pytest.mark.parametrize("family", ["WeibullPH", "CoxPH"])
def test_every_formula_round_trips(family, formula):
    model = _fit(family, formula, _df())
    restored = _rt(model)
    assert restored.feature_names == model.feature_names
    np.testing.assert_allclose(
        _sf(restored, family, NEW), _sf(model, family, NEW), rtol=1e-12
    )


@pytest.mark.parametrize(
    "family",
    [
        "WeibullAFT",
        "WeibullPO",
        "WeibullAH",
        "AdditiveHazards",
        "BuckleyJames",
        "WeibullFrailty",
    ],
)
def test_other_families_round_trip(family):
    formula = "C(g, levels=['c', 'b', 'a']) + scale(z) + C(k) + h"
    model = _fit(family, formula, _df())
    restored = _rt(model)
    np.testing.assert_allclose(
        _sf(restored, family, NEW), _sf(model, family, NEW), rtol=1e-12
    )


def test_accelerated_life_transform_round_trips():
    # a single-stress life model with a data-dependent transform
    model = _fit("AL", "I(scale(z) + 5)", _df())
    restored = _rt(model)
    np.testing.assert_allclose(
        restored.sf(T, NEW), model.sf(T, NEW), rtol=1e-12
    )


def test_competing_risks_formula_round_trips():
    df = _df()
    model = CompetingRisksProportionalHazards.fit_from_df(
        df, "t", "cause", c_col="c", formula="0 + C(g) + poly(z, 2)"
    )
    restored = _rt(model)
    before = prepare_Z(NEW, model.feature_names, model._model_spec)
    after = prepare_Z(NEW, restored.feature_names, restored._model_spec)
    np.testing.assert_array_equal(after, before)
    np.testing.assert_allclose(
        restored.sf(T[:1], after[:1], "u"),
        model.sf(T[:1], before[:1], "u"),
        rtol=1e-12,
    )


def test_level_order_and_reference_level_are_kept():
    model = _fit("WeibullPH", "z + C(g, levels=['c', 'b', 'a']) + kc", _df())
    restored = _rt(model)
    # reference levels 'c' (from levels=) and 3 (the column's first category)
    assert restored.feature_names == [
        "z",
        "C(g, levels=['c', 'b', 'a'])[T.b]",
        "C(g, levels=['c', 'b', 'a'])[T.a]",
        "kc[T.1]",
        "kc[T.2]",
    ]
    state = restored.to_dict()["formula_meta"]["encoder_state"]
    assert state["kc"]["state"]["categories"] == [3, 1, 2]
    assert state["C(g, levels=['c', 'b', 'a'])"]["state"]["categories"] == [
        "c",
        "b",
        "a",
    ]
    # the reference rows expand to zeros exactly as the original does
    ref = pd.DataFrame({"z": [2.0], "g": ["c"], "kc": [3]})
    np.testing.assert_array_equal(
        prepare_Z(ref, restored.feature_names, restored._model_spec),
        [[2.0, 0.0, 0.0, 0.0, 0.0]],
    )


def test_integer_levels_round_trip_to_the_same_prediction():
    # v0.20 stored every level as a string, so integer levels (3, 1, 2)
    # came back as '3', '1', '2' and matched nothing: every row was coded
    # as the reference level, off by up to ~0.05 in sf.
    model = _fit("WeibullPH", "z + kc", _df())
    restored = _rt(model)
    np.testing.assert_allclose(
        restored.sf(T, NEW), model.sf(T, NEW), rtol=1e-12
    )


def test_numeric_column_for_categorical_factor_is_coded_by_level():
    # A column fitted as a Categorical of integers but passed as plain
    # integers was read as numerical and its raw value copied into every
    # level's column (kc[T.1] = kc[T.2] = 2 for kc = 2).
    model = _fit("WeibullPH", "z + kc", _df())
    plain = NEW.assign(kc=[2, 1, 3, 3])
    np.testing.assert_array_equal(
        prepare_Z(plain, model.feature_names, model._model_spec),
        prepare_Z(NEW, model.feature_names, model._model_spec),
    )
    np.testing.assert_allclose(model.sf(T, plain), model.sf(T, NEW))


@pytest.mark.parametrize("family", ["CoxPH", "WeibullPH"])
def test_no_intercept_formula_text_is_kept(family):
    # Cox stored ``str`` of its parsed formula, which drops "0 +"; the
    # restored model then expected 3 columns for its 4 coefficients.
    model = _fit(family, "0 + z + g", _df())
    d = model.to_dict()
    assert d["formula"].replace(" ", "").startswith("0+")
    restored = surpyval.from_dict(json.loads(json.dumps(d)))
    np.testing.assert_allclose(
        restored.sf(T, NEW), model.sf(T, NEW), rtol=1e-12
    )


def _legacy(d, formula=None):
    # the formula_meta layout v0.17 - v0.20 wrote
    d = json.loads(json.dumps(d))
    meta = d["formula_meta"]
    for key in ("encoder_state", "transform_state", "data_variables"):
        meta.pop(key, None)
    if formula is not None:
        d["formula"] = formula
    return d


def test_legacy_pair_written_only_when_an_older_reader_gets_it_right():
    df = _df()
    bare = _fit("WeibullPH", "z * g", df).to_dict()["formula_meta"]
    assert bare["factor_levels"] == {"g": ["a", "b", "c"]}
    assert bare["numeric_features"] == ["z"]
    for formula in ["z + C(g)", "z + kc", "scale(z) + g"]:
        meta = _fit("WeibullPH", formula, df).to_dict()["formula_meta"]
        assert "factor_levels" not in meta
        assert "numeric_features" not in meta


def test_legacy_layout_still_reads():
    model = _fit("WeibullPH", "z * g", _df())
    restored = surpyval.from_dict(_legacy(model.to_dict()))
    np.testing.assert_allclose(
        restored.sf(T, NEW), model.sf(T, NEW), rtol=1e-12
    )


def test_legacy_cox_dict_with_dropped_zero_is_repaired():
    # what v0.20 wrote for a Cox "0 + z + g" fit: the formula text "z + g"
    # with the four full-rank feature names
    model = _fit("CoxPH", "0 + z + g", _df())
    d = _legacy(model.to_dict(), formula="z + g")
    d["formula_meta"]["factor_levels"] = {"g": ["a", "b", "c"]}
    d["formula_meta"]["numeric_features"] = ["z"]
    restored = surpyval.from_dict(d)
    assert restored.formula == "0 + z + g"
    np.testing.assert_allclose(
        restored.sf(T, NEW), model.sf(T, NEW), rtol=1e-12
    )


def test_inconsistent_formula_raises_on_load():
    model = _fit("WeibullPH", "z + g", _df())
    d = json.loads(json.dumps(model.to_dict()))
    d["formula"] = "0 + z + g"
    with pytest.raises(ValueError, match="fit with"):
        surpyval.from_dict(d)


def test_state_codec_round_trips_types():
    state = {
        "alpha": {0: np.float64(2.5), 1: 3.0},
        "knots": [1.0, 2.0],
        "constraints": np.array([[0.25, 0.75]]),
        "pair": (1, "a"),
        "flag": True,
        "none": None,
        "__tuple__": 1,
    }
    encoded = regression_data._state_to_json(state, "t(z)")
    text = json.dumps(encoded, allow_nan=False)
    back = regression_data._state_from_json(json.loads(text))
    assert back["alpha"] == {0: 2.5, 1: 3.0}
    assert back["pair"] == (1, "a")
    assert back["__tuple__"] == 1
    assert back["constraints"].dtype == np.float64
    np.testing.assert_array_equal(back["constraints"], [[0.25, 0.75]])
    assert back["knots"] == [1.0, 2.0] and back["flag"] is True
    assert back["none"] is None


def test_unstorable_state_raises_at_to_dict_time():
    with pytest.raises(NotImplementedError, match="t\\(z\\)"):
        regression_data._state_to_json(
            {"when": pd.Timestamp("2020-01-01")}, "t(z)"
        )


@pytest.mark.parametrize(
    "formula, schema",
    [("z + g", 1), ("z", 1), ("z + C(g)", 2), ("scale(z) + g", 2)],
)
def test_schema_stamp_is_the_oldest_that_rebuilds_the_formula(formula, schema):
    # v0.20 reads schema 1 and rebuilds plain columns and string
    # categoricals from factor_levels; anything it would fail on is
    # stamped 2, so it asks for an upgrade instead of a formula error.
    import surpyval as surv
    from surpyval.serialisation import required_schema

    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "x": rng.weibull(2, 60) * 10,
            "z": rng.normal(size=60),
            "g": rng.choice(["a", "b", "c"], 60),
        }
    )
    for family in (surv.WeibullPH, surv.CoxPH):
        d = family.fit_from_df(df, x_col="x", formula=formula).to_dict()
        assert d["schema"] == required_schema(d)
        if family is surv.CoxPH:
            # A Cox model's nonzero covariate centre alone makes it schema
            # 2 (#459); the formula is judged without it.
            assert d["schema"] == 2
            d = {k: v for k, v in d.items() if k != "center"}
        assert required_schema(d) == schema
