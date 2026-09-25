"""Regression tests for the second docs-review bug-fix round (competing risks).

- ``gray_test`` reads a NaN cause as censored when ``c`` is omitted, the same
  missing-cause rule as every other competing-risks class (it used to count
  such a row as a failure from a competing cause).
- ``CompetingRisksProportionalHazards`` serialises (``to_dict``/``from_dict``,
  JSON files, ``surpyval.from_dict``) for both ``how="Cox"`` and
  ``how="Fine-Gray"``, and the reloaded model predicts identically.
"""

import json

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import gray_test
from surpyval.serialisation import SCHEMA_VERSION
from surpyval.univariate.competing_risks import (
    CompetingRisksProportionalHazards,
)

# -- gray_test: missing causes ------------------------------------------------

X8 = [1, 2, 3, 4, 5, 6, 7, 8]
G8 = [0, 0, 0, 0, 1, 1, 1, 1]


def test_gray_test_nan_cause_is_censored():
    with_none = gray_test(X8, [1, 2, None, 1, 2, 1, None, 2], G8, cause=1)
    with_nan = gray_test(X8, [1, 2, np.nan, 1, 2, 1, np.nan, 2], G8, cause=1)
    assert with_nan.statistic == pytest.approx(with_none.statistic)
    assert with_nan.p_value == pytest.approx(with_none.p_value)
    # and both match the explicit censoring flags
    c = [0, 0, 1, 0, 0, 0, 1, 0]
    explicit = gray_test(X8, [1, 2, None, 1, 2, 1, None, 2], G8, 1, c=c)
    assert with_nan.statistic == pytest.approx(explicit.statistic)


def test_gray_test_pandas_missing_cause_is_censored():
    frame = pd.DataFrame(
        {"x": X8, "e": [1, 2, None, 1, 2, 1, None, 2], "g": G8}
    )
    assert frame["e"].isna().sum() == 2  # stored as NaN by pandas
    res = gray_test(frame["x"], frame["e"], frame["g"], cause=1)
    ref = gray_test(X8, [1, 2, None, 1, 2, 1, None, 2], G8, cause=1)
    assert res.statistic == pytest.approx(ref.statistic)


def test_gray_test_c_must_agree_with_missing_causes():
    # a NaN cause on a row flagged as a failure is as ambiguous here as it is
    # for the model classes
    with pytest.raises(ValueError, match="missing event"):
        gray_test(
            X8,
            [1, 2, np.nan, 1, 2, 1, np.nan, 2],
            G8,
            cause=1,
            c=np.zeros(8),
        )


# -- CompetingRisksProportionalHazards serialisation ------------------------


def _cr_data(seed=0, n=150):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    t_a = rng.exponential(1 / (0.1 * np.exp(Z @ [0.7, 0.2])))
    t_b = rng.exponential(1 / (0.05 * np.exp(Z @ [-0.3, 0.1])))
    t_c = rng.uniform(0, 20, n)
    x = np.minimum.reduce([t_a, t_b, t_c]).round(2)
    e = np.where(
        t_c < np.minimum(t_a, t_b), None, np.where(t_a < t_b, "a", "b")
    )
    return x, Z, e


T = np.array([0.0, 0.5, 2.0, 5.0, 10.0, 30.0])
ZS = [[0.0, 0.0], [1.0, -2.0], [3.0, 3.0]]


def _assert_same_predictions(model, restored):
    for z in ZS:
        for ev in ["a", "b"]:
            for f in ["cif", "sf", "ff", "Hf"]:
                np.testing.assert_array_equal(
                    getattr(model, f)(T, z, ev), getattr(restored, f)(T, z, ev)
                )
            if model.how == "Cox":
                for f in ["hf", "df"]:
                    np.testing.assert_array_equal(
                        getattr(model, f)(T, z, ev),
                        getattr(restored, f)(T, z, ev),
                    )
        if model.how == "Cox":
            for f in ["sf", "Hf", "hf"]:
                np.testing.assert_array_equal(
                    getattr(model, f)(T, z), getattr(restored, f)(T, z)
                )
    np.testing.assert_array_equal(model.betas, restored.betas)
    np.testing.assert_array_equal(model.beta, restored.beta)
    assert restored.event_idx_map == model.event_idx_map


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_round_trips_through_json(how):
    x, Z, e = _cr_data()
    model = CompetingRisksProportionalHazards.fit(x, Z, e, how=how)
    d = model.to_dict()
    assert d["model"] == "CompetingRisksProportionalHazards"
    assert d["schema"] == SCHEMA_VERSION
    restored = surpyval.from_dict(json.loads(json.dumps(d)))
    assert type(restored) is CompetingRisksProportionalHazards
    assert restored.how == how
    assert restored.results is None
    _assert_same_predictions(model, restored)
    # class-level reader too
    again = CompetingRisksProportionalHazards.from_dict(d)
    _assert_same_predictions(model, again)


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_to_json_file(tmp_path, how):
    x, Z, e = _cr_data(seed=1)
    model = CompetingRisksProportionalHazards.fit(x, Z, e, how=how)
    path = tmp_path / "crph.json"
    model.to_json(path)
    _assert_same_predictions(model, surpyval.from_json(path))
    _assert_same_predictions(
        model, CompetingRisksProportionalHazards.from_json(path)
    )


def test_crph_numpy_integer_causes_round_trip():
    # numpy-integer cause labels (as np.where produces) must serialise to
    # JSON, including inside the per-cause Fine-Gray models
    x, Z, e = _cr_data(seed=2)
    codes = np.array(
        [None if v is None else np.int64(1 if v == "a" else 2) for v in e],
        dtype=object,
    )
    for how in ["Cox", "Fine-Gray"]:
        model = CompetingRisksProportionalHazards.fit(x, Z, codes, how=how)
        restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
        for ev in [1, 2]:
            np.testing.assert_array_equal(
                model.cif(T, [0.5, -1.0], ev), restored.cif(T, [0.5, -1.0], ev)
            )


def test_crph_formula_metadata_round_trips():
    x, Z, e = _cr_data(seed=3)
    frame = pd.DataFrame({"t": x, "cause": e, "z1": Z[:, 0], "z2": Z[:, 1]})
    model = CompetingRisksProportionalHazards.fit_from_df(
        frame, x_col="t", e_col="cause", formula="z1 + z2"
    )
    restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.feature_names == model.feature_names == ["z1", "z2"]
    assert restored.formula == str(model.formula)
    _assert_same_predictions(model, restored)


def test_crph_from_dict_rejects_other_models():
    x, Z, e = _cr_data()
    fg = surpyval.univariate.competing_risks.FineGray.fit(x, Z, e, cause="a")
    with pytest.raises(ValueError, match="CompetingRisksProportionalHazards"):
        CompetingRisksProportionalHazards.from_dict(fg.to_dict())


@pytest.mark.parametrize("how", ["Cox", "Fine-Gray"])
def test_crph_dict_is_bson_native(how):
    # MongoDB's encoder rejects numpy scalars that json.dumps tolerates
    bson = pytest.importorskip("bson")
    x, Z, e = _cr_data(seed=4)
    model = CompetingRisksProportionalHazards.fit(x, Z, e, how=how)
    doc = bson.decode(bson.encode(model.to_dict()))
    _assert_same_predictions(model, surpyval.from_dict(doc))
