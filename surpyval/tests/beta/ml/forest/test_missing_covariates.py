"""
Missing covariates in survival trees and forests (#375 item 6).

A NaN compares false with every split value, so a tree sent it down the
right-hand branch of every split on its feature: at fit such rows were kept
without a word, and at prediction a row with a missing covariate got a
number. The package rule: at fit a row with a missing covariate is dropped
with one warning giving the count; at prediction it gives NaN, and the
other rows are unaffected.
"""

import contextlib
import io
import warnings

import numpy as np
import pytest

from surpyval.beta.ml import RandomSurvivalForest, SurvivalTree

FUNCTIONS = ["sf", "ff", "df", "hf", "Hf"]
XS = np.array([2.0, 5.0, 8.0])


def _data():
    # Life halves when z0 > 0.5
    rng = np.random.default_rng(0)
    Z = rng.uniform(0, 1, (120, 2))
    x = rng.weibull(2.0, 120) * np.where(Z[:, 0] > 0.5, 5.0, 10.0)
    c = (x > 12).astype(int)
    return np.minimum(x, 12), c, Z


def _fit_tree(x, Z, c):
    np.random.seed(0)
    return SurvivalTree.fit(
        x, Z, c, max_depth=2, n_features_split="all", kind="exponential"
    )


def _fit_forest(x, Z, c):
    np.random.seed(0)
    with contextlib.redirect_stderr(io.StringIO()):  # joblib progress log
        return RandomSurvivalForest.fit(
            x, Z, c, n_trees=4, max_depth=2, kind="exponential"
        )


FITTERS = {"tree": _fit_tree, "forest": _fit_forest}


@pytest.fixture(scope="module", params=list(FITTERS))
def fitted(request):
    x, c, Z = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = FITTERS[request.param](x, Z, c)
    return request.param, model


@pytest.mark.parametrize("fn", FUNCTIONS)
def test_missing_covariate_row_predicts_nan(fitted, fn):
    _, model = fitted
    Zq = np.array([[0.2, 0.5], [np.nan, 0.5], [0.8, np.nan], [0.8, 0.5]])
    out = getattr(model, fn)(XS, Zq, grid=True)
    # Old code: rows 1 and 2 were routed right and got numbers
    assert out.shape == (4, XS.size)
    assert np.isnan(out[1:3]).all()
    # The other rows are exactly what they are without the missing rows
    np.testing.assert_array_equal(
        out[[0, 3]], getattr(model, fn)(XS, Zq[[0, 3]], grid=True)
    )
    assert np.isfinite(out[[0, 3]]).all()


def test_missing_covariate_vector_predicts_nan(fitted):
    _, model = fitted
    out = model.sf(XS, [np.nan, 0.5])
    assert out.shape == XS.shape
    assert np.isnan(out).all()
    # None in a list is a missing value too
    assert np.isnan(model.sf(XS, [[None, 0.5]])).all()
    assert np.isfinite(model.sf(XS, [0.2, 0.5])).all()


def test_all_rows_missing_predicts_all_nan(fitted):
    _, model = fitted
    out = model.sf(XS, np.full((3, 2), np.nan), grid=True)
    assert out.shape == (3, XS.size)
    assert np.isnan(out).all()


def test_restored_model_predicts_nan(fitted):
    kind, model = fitted
    cls = SurvivalTree if kind == "tree" else RandomSurvivalForest
    restored = cls.from_dict(model.to_dict())
    out = restored.sf(XS, [[np.nan, 0.5], [0.2, 0.5]], grid=True)
    assert np.isnan(out[0]).all()
    np.testing.assert_array_equal(out[1], model.sf(XS, [0.2, 0.5]))


@pytest.mark.parametrize("kind", list(FITTERS))
def test_fit_drops_missing_rows_with_one_warning(kind):
    x, c, Z = _data()
    Zn = Z.copy()
    Zn[3, 0] = np.nan
    Zn[10, 1] = np.nan
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = FITTERS[kind](x, Zn, c)
    dropped = [w for w in caught if "Dropped" in str(w.message)]
    # Old code: no warning, and the NaN rows were kept
    assert len(dropped) == 1
    assert issubclass(dropped[0].category, UserWarning)
    assert "Dropped 2 of 120 rows" in str(dropped[0].message)
    assert len(model.data) == 118
    assert not np.isnan(model.Z).any()

    # Identical to a fit on the complete rows alone
    keep = np.ones(120, dtype=bool)
    keep[[3, 10]] = False
    clean = FITTERS[kind](x[keep], Z[keep], c[keep])
    Zq = np.array([[0.2, 0.5], [0.8, 0.5], [0.6, 0.1]])
    np.testing.assert_allclose(
        model.sf(XS, Zq, grid=True), clean.sf(XS, Zq, grid=True)
    )


def test_fit_with_every_row_missing_raises():
    x, c, Z = _data()
    with pytest.raises(ValueError, match="Every row has a missing"):
        _fit_tree(x, np.full_like(Z, np.nan), c)


def test_fit_with_wrong_number_of_rows_raises():
    x, c, Z = _data()
    with pytest.raises(ValueError, match="Z has 119 row"):
        _fit_tree(x, Z[:-1], c)


def test_forest_score_with_missing_covariate_is_nan():
    x, c, Z = _data()
    forest = _fit_forest(x, Z, c)
    Zq = Z[:20].copy()
    assert np.isfinite(forest.score(x[:20], Zq, c[:20]))
    Zq[4, 1] = np.nan
    # Old code: the NaN row was routed right and the index was a number
    assert np.isnan(forest.score(x[:20], Zq, c[:20]))
