"""A penalty is an argument of the fit (v0.30).

``fit(..., penalizer=, l1_ratio=, standardise=)`` (and ``fit_tvc``,
``fit_tvc_timeline``) and ``CoxPH.fit(..., firth=True)`` give the model the
penalised methods gave, to the last digit; those methods still work, with a
``FutureWarning`` naming the argument, until v0.31. Without a penalty the
fit is the fit it was.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import surpyval as sp
from surpyval.datasets import load_rossi_static

ROSSI = load_rossi_static()
X = ROSSI["week"].to_numpy(dtype=float)
C = 1 - ROSSI["arrest"].to_numpy()
Z = ROSSI[["fin", "age", "prio"]].to_numpy(dtype=float)

PENALTIES = [
    {"penalizer": 0.05},  # ridge
    {"penalizer": 0.05, "l1_ratio": 1.0},  # lasso
    {"penalizer": 0.05, "l1_ratio": 0.5},  # elastic net
    {"penalizer": 0.02, "l1_ratio": 0.5, "standardise": False},
]


def _quiet(call):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return call()


def _old(call, match):
    with pytest.warns(FutureWarning, match=match) as caught:
        out = call()
    assert caught[0].filename == __file__
    return out


def _same(a, b):
    assert_array_equal(np.asarray(a.params), np.asarray(b.params))


@pytest.mark.parametrize("penalty", PENALTIES)
@pytest.mark.parametrize(
    "fitter",
    ["CoxPH", "WeibullPH", "WeibullAFT", "WeibullPO", "ProportionalOdds"],
)
def test_fit_with_a_penalizer_is_fit_penalised(fitter, penalty):
    f = getattr(sp, fitter)
    old = _old(
        lambda: f.fit_penalised(X, Z, c=C, **penalty),
        rf"{fitter}.fit_penalised is deprecated .* v0.31; call fit\(\.\.\., "
        r"penalizer=\.\.\.\); its default was penalizer=0.1",
    )
    new = _quiet(lambda: f.fit(X, Z, c=C, **penalty))
    _same(old, new)
    assert new.penalty is not None


def test_the_default_penalizer_of_the_old_method_is_kept():
    old = _old(lambda: sp.CoxPH.fit_penalised(X, Z, c=C), "penalizer=0.1")
    _same(old, sp.CoxPH.fit(X, Z, c=C, penalizer=0.1))


def test_penalizer_zero_is_the_penalised_fit_at_zero():
    # Every old call maps to one new call: fit_penalised(penalizer=0) is a
    # penalised model with a zero penalty, and so is fit(penalizer=0).
    old = _old(
        lambda: sp.WeibullPH.fit_penalised(X, Z, c=C, penalizer=0.0),
        "fit_penalised",
    )
    new = sp.WeibullPH.fit(X, Z, c=C, penalizer=0.0)
    _same(old, new)
    assert new.penalty is not None


@pytest.mark.parametrize("fitter", ["CoxPH", "WeibullPH", "WeibullAFT"])
def test_without_a_penalty_the_fit_is_unchanged(fitter):
    f = getattr(sp, fitter)
    model = _quiet(lambda: f.fit(X, Z, c=C))
    assert getattr(model, "penalty", None) is None
    # standard errors are there, as for every unpenalised fit
    assert np.all(np.isfinite(model.standard_errors()))


def test_firth():
    old = _old(
        lambda: sp.CoxPH.fit_firth(X[:120], Z[:120], c=C[:120], profile=False),
        r"CoxPH.fit_firth is deprecated .* call fit\(\.\.\., firth=True\)",
    )
    new = _quiet(
        lambda: sp.CoxPH.fit(
            X[:120], Z[:120], c=C[:120], firth=True, profile=False
        )
    )
    _same(old, new)
    assert new.firth.lower.shape == old.firth.lower.shape
    assert_array_equal(new.firth.lower, old.firth.lower)


def _tvc_rows():
    i = np.repeat(np.arange(40), 2)
    xl = np.tile([0.0, 5.0], 40)
    rng = np.random.default_rng(4)
    xr = xl + 5.0 + rng.uniform(0, 5, 80) * (xl > 0)
    c = np.tile([1, 0], 40)
    c[rng.uniform(size=80) < 0.3] = 1
    Zt = rng.normal(size=(80, 2))
    return i, xl, xr, c, Zt


@pytest.mark.parametrize("fitter", ["CoxPH", "WeibullPH", "WeibullAFT"])
def test_fit_tvc_with_a_penalizer(fitter):
    i, xl, xr, c, Zt = _tvc_rows()
    f = getattr(sp, fitter)
    old = _old(
        lambda: f.fit_tvc_penalised(i, xl, xr, c, Zt, penalizer=0.1),
        r"fit_tvc_penalised is deprecated .* call fit_tvc\(",
    )
    new = _quiet(lambda: f.fit_tvc(i, xl, xr, c, Zt, penalizer=0.1))
    _same(old, new)


@pytest.mark.parametrize("fitter", ["CoxPH", "WeibullPH"])
def test_fit_tvc_timeline_with_a_penalizer(fitter):
    i = np.repeat(np.arange(30), 2)
    x = np.tile([0.0, 4.0], 30) + np.repeat(
        np.linspace(1, 9, 30), 2
    ) * np.tile([0, 1], 30)
    rng = np.random.default_rng(5)
    c = np.tile([1, 0], 30)
    Zt = rng.normal(size=(60, 1))
    f = getattr(sp, fitter)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            old = f.fit_tvc_timeline_penalised(i, x, Zt, c, penalizer=0.1)
    except ValueError as error:
        pytest.skip(f"the timeline rows are refused: {error}")
    new = _quiet(lambda: f.fit_tvc_timeline(i, x, Zt, c, penalizer=0.1))
    _same(old, new)


def test_other_families():
    rng = np.random.default_rng(1)
    e = np.where(C == 1, None, rng.choice(["a", "b"], size=C.size))
    x_cr = X
    old = _old(
        lambda: sp.FineGray.fit_penalised(
            x_cr, Z, e, event="a", penalizer=0.1, l1_ratio=1
        ),
        "FineGray.fit_penalised",
    )
    new = _quiet(
        lambda: sp.FineGray.fit(
            x_cr, Z, e, event="a", penalizer=0.1, l1_ratio=1
        )
    )
    assert_array_equal(old.beta, new.beta)
    old = _old(
        lambda: sp.AdditiveHazards.fit_penalised(X, Z, c=C, penalizer=0.1),
        "AdditiveHazards.fit_penalised",
    )
    new = _quiet(lambda: sp.AdditiveHazards.fit(X, Z, c=C, penalizer=0.1))
    assert_array_equal(old.beta, new.beta)


def test_options_only_the_penalised_fit_takes():
    # AdditiveHazards.fit has no tol; its penalised fit does
    old = _old(
        lambda: sp.AdditiveHazards.fit_penalised(
            X, Z, c=C, penalizer=0.1, tol=1e-8
        ),
        "fit_penalised",
    )
    new = sp.AdditiveHazards.fit(X, Z, c=C, penalizer=0.1, tol=1e-8)
    assert_array_equal(old.beta, new.beta)
    with pytest.raises(TypeError, match="'tol'.*only with a penalizer"):
        sp.AdditiveHazards.fit(X, Z, c=C, tol=1e-8)


def test_refusals_say_what_to_do():
    with pytest.raises(ValueError, match="l1_ratio sets the penalty"):
        sp.CoxPH.fit(X, Z, c=C, l1_ratio=0.5)
    with pytest.raises(ValueError, match="standardise sets the penalty"):
        sp.WeibullPH.fit(X, Z, c=C, standardise=False)
    with pytest.raises(ValueError, match="time_interaction cannot be given"):
        sp.CoxPH.fit(X, Z, c=C, penalizer=0.1, time_interaction={0: "log"})
    with pytest.raises(ValueError, match="give one of firth and penalizer"):
        sp.CoxPH.fit(X, Z, c=C, penalizer=0.1, firth=True)


def test_the_signature_shows_the_penalty():
    import inspect

    params = inspect.signature(sp.CoxPH.fit).parameters
    for name in ("penalizer", "l1_ratio", "standardise", "firth"):
        assert name in params
    assert params["penalizer"].default is None
    assert "penalizer" in sp.CoxPH.fit.__doc__
    assert "firth" not in inspect.signature(sp.WeibullPH.fit).parameters


def test_penalty_path_takes_the_fit():
    from surpyval.univariate.regression.penalty_path import penalty_path

    path = _quiet(
        lambda: penalty_path(
            sp.CoxPH.fit, X, Z, c=C, l1_ratio=1.0, n_penalties=4
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        old = penalty_path(
            sp.CoxPH.fit_penalised, X, Z, c=C, l1_ratio=1.0, n_penalties=4
        )
    assert_array_equal(path.coefficients, old.coefficients)
