"""
Arguments that a call by position silently reads as something else.

A value passed by position lands on whatever argument stands there, and a
few methods hold a different thing at a place where the package's other
methods of that name (or the ``x, c, n, t`` order of the data) hold
another, so the call returns a number with no hint that the value was
read as something else:

- the recurrent fits take the item ids ``i`` where every other fit takes
  the censoring flags ``c``: ``CrowAMSAA.fit(x, c)`` reads the flags as
  two items with no censoring (alpha 19.99 for 7.63 with ``c=c``);
- ``Parametric.mean(defective, tau)`` and
  ``ParametricCompetingRisks.mean(defective)`` take ``defective`` where a
  non-parametric ``mean(tau)`` takes ``tau``, so ``mean(5)`` ignores the
  5;
- ``Parametric.mean_cb(alpha_ci, ...)`` takes ``alpha_ci`` where a
  non-parametric ``mean_cb(tau, alpha_ci)`` takes ``tau``;
- ``Parametric.random(size, a, b)`` and ``random_data`` take the lower
  truncation bound where every other model's ``random(size,
  random_state)`` takes the seed;
- ``CompetingRisks.plot(stacked, ax)`` takes ``stacked`` where every other
  ``plot(ax)`` takes the axes.

Each now warns with a FutureWarning saying what the value was taken as
and the keyword to use, and stays silent when the argument is named. The
result is exactly what it was (the call is unchanged).
"""

import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.utils.deprecation import REMOVED_IN_NEXT

matplotlib.use("Agg")

X = np.array([5.0, 12, 20, 31, 45, 60, 80])
C = np.array([0, 0, 0, 0, 0, 0, 1])
LIFE = np.arange(3.0, 11.0)


def _warned(call):
    """``call()``'s result and the messages of its FutureWarnings."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, [
        str(w.message) for w in caught if w.category is FutureWarning
    ]


def _quiet(call):
    """``call()``, which must give no FutureWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        warnings.simplefilter("error", FutureWarning)
        return call()


def _unwrapped(method):
    """The method as it was before the warning was added."""
    return method.__wrapped__


# -- recurrent fits: item ids where the other fits take censoring ----------

RECURRENT = [
    sp.CrowAMSAA,
    sp.Duane,
    sp.CoxLewis,
    sp.HPP,
    sp.NonParametricCounting,
    sp.ARA,
    sp.ARI,
    sp.GeneralizedRenewal,
    sp.GeneralizedOneRenewal,
]


def _same(model, other):
    """The two fits are bit for bit the same."""
    if hasattr(model, "params"):
        return np.array_equal(
            np.asarray(model.params), np.asarray(other.params)
        )
    return np.array_equal(model.mcf(X), other.mcf(X))


def _name(fitter):
    return type(fitter).__name__


def test_crow_amsaa_example():
    # The case that started this: the flags were read as two items.
    model, messages = _warned(lambda: sp.CrowAMSAA.fit(X, C))
    np.testing.assert_allclose(model.params, [19.9857, 0.99943], rtol=1e-4)
    assert len(messages) == 1
    message = messages[0]
    assert message.startswith("CrowAMSAA.fit: the second positional")
    assert "'i'" in message and "c=..." in message and "i=..." in message
    assert f"v{REMOVED_IN_NEXT}" in message
    named = _quiet(lambda: sp.CrowAMSAA.fit(X, c=C))
    np.testing.assert_allclose(named.params, [7.6347, 0.76267], rtol=1e-4)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter", RECURRENT, ids=_name)
def test_recurrent_fit_warns_on_flags_read_as_items(fitter):
    model, messages = _warned(lambda: fitter.fit(X, C))
    assert len(messages) == 1, messages
    assert f"{_name(fitter)}.fit:" in messages[0]
    before = _unwrapped(type(fitter).fit)(fitter, X, C)
    assert _same(model, before)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter", RECURRENT, ids=_name)
def test_recurrent_fit_is_quiet_when_unambiguous(fitter):
    i = np.ones(len(X), int)
    _quiet(lambda: fitter.fit(X, c=C))
    _quiet(lambda: fitter.fit(X, i, C))  # c given: the order is known
    _quiet(lambda: fitter.fit(X, i))  # one item
    _quiet(lambda: fitter.fit(X, i=np.r_[0, 0, 0, 1, 1, 1, 1]))  # named
    _quiet(lambda: fitter.fit(X, np.r_[1, 1, 1, 2, 2, 2, 2]))  # ids


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter", RECURRENT, ids=_name)
def test_recurrent_fit_from_df(fitter):
    df = pd.DataFrame({"x": X, "c": C, "i": np.ones(len(X), int)})
    model, messages = _warned(lambda: fitter.fit_from_df(df, "x", "c"))
    assert len(messages) == 1, messages
    assert "'i_col'" in messages[0] and "c_col=..." in messages[0]
    same = _quiet(lambda: fitter.fit(X, i=C))
    assert _same(model, same)
    _quiet(lambda: fitter.fit_from_df(df, "x", c_col="c"))
    _quiet(lambda: fitter.fit_from_df(df, "x", "i", "c"))
    _quiet(lambda: fitter.fit_from_df(df, "x", "i"))


@pytest.mark.parametrize(
    "fitter", [sp.CauseSpecificNHPP, sp.CauseSpecificMCF], ids=str
)
def test_cause_specific_recurrent(fitter):
    e = np.array(["a", "b"] * 3 + ["a"])
    _, messages = _warned(lambda: fitter.fit(X, C, e=e))
    assert len(messages) == 1
    _quiet(lambda: fitter.fit(X, np.ones(len(X)), C, e=e))
    df = pd.DataFrame({"x": X, "c": C, "i": 1, "e": e})
    _, messages = _warned(lambda: fitter.fit_from_df(df, "x", "e", "c"))
    assert len(messages) == 1
    _quiet(lambda: fitter.fit_from_df(df, "x", "e", "i", "c"))


# The regressions take ``i`` after ``Z``, where a regression's
# ``fit(x, Z, c, n)`` takes ``c``. Read as ids these flags make items
# whose covariates change between rows, so the fit is refused after the
# warning.
XS = np.array([10, 30, 45, 70, 100, 20, 35, 60, 100, 5, 50, 100], dtype=float)
IS = np.repeat([1, 2, 3], [5, 4, 3])
CS = np.array([0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1])
ZS = np.repeat([0.0, 1.0, 0.5], [5, 4, 3])[:, None]

REGRESSIONS = [
    sp.ProportionalIntensityHPP,
    sp.ProportionalIntensityNHPP,
    sp.AndersenGill,
    sp.PrenticeWilliamsPeterson,
    sp.WeiLinWeissfeld,
]


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("fitter", REGRESSIONS, ids=_name)
def test_recurrent_regression(fitter):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            fitter.fit(XS, ZS, CS)
    assert [w for w in caught if w.category is FutureWarning]
    _quiet(lambda: fitter.fit(XS, ZS, IS, CS))
    _quiet(lambda: fitter.fit(XS, ZS, i=IS, c=CS))
    df = pd.DataFrame({"x": XS, "z": ZS[:, 0], "i": IS, "c": CS})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError):
            fitter.fit_from_df(df, "x", "z", "c")
    assert [w for w in caught if w.category is FutureWarning]
    _quiet(lambda: fitter.fit_from_df(df, "x", "z", "i", "c"))


# -- model methods ----------------------------------------------------------


@pytest.fixture(scope="module")
def weibull():
    return sp.Weibull.fit(LIFE)


def test_parametric_mean(weibull):
    value, messages = _warned(lambda: weibull.mean(5))
    assert len(messages) == 1
    assert messages[0].startswith("Parametric.mean: the first positional")
    assert "'defective'" in messages[0] and "tau=..." in messages[0]
    assert value == _unwrapped(type(weibull).mean)(weibull, 5)
    np.testing.assert_allclose(value, 6.5238, rtol=1e-4)
    assert _quiet(lambda: weibull.mean(defective=True)) == value
    _quiet(lambda: weibull.mean(tau=5))
    _quiet(weibull.mean)


def test_parametric_mean_cb(weibull):
    bounds, messages = _warned(lambda: weibull.mean_cb(0.1))
    assert len(messages) == 1 and "'alpha_ci'" in messages[0]
    before = _unwrapped(type(weibull).mean_cb)(weibull, 0.1)
    assert np.array_equal(bounds, before)
    assert np.array_equal(
        _quiet(lambda: weibull.mean_cb(alpha_ci=0.1)), before
    )


@pytest.mark.parametrize("method", ["random", "random_data"])
def test_parametric_random_truncation(weibull, method):
    draw = getattr(weibull, method)
    raw = _unwrapped(getattr(type(weibull), method))
    out, messages = _warned(lambda: draw(5, 4.0, random_state=1))
    assert len(messages) == 1
    assert "'a'" in messages[0] and "random_state=..." in messages[0]
    before = raw(weibull, 5, 4.0, random_state=1)
    if method == "random":
        out, before = (out,), (before,)
    for got, want in zip(out, before):
        assert np.array_equal(got, want)
    _quiet(lambda: draw(5, a=4.0, random_state=1))
    _quiet(lambda: draw(5, random_state=1))


def test_parametric_competing_risks_mean():
    x = np.arange(1.0, 13.0)
    e = np.array(["a", "b"] * 6)
    model = sp.ParametricCompetingRisks.fit(x, e)
    value, messages = _warned(lambda: model.mean(True))
    assert len(messages) == 1 and "'defective'" in messages[0]
    assert value == _unwrapped(type(model).mean)(model, True)
    assert _quiet(lambda: model.mean(defective=True)) == value


def test_competing_risks_plot():
    import matplotlib.pyplot as plt

    model = sp.CompetingRisks.fit(np.arange(1.0, 9.0), ["a", "b"] * 4)
    fig, ax = plt.subplots()
    try:
        _, messages = _warned(lambda: model.plot(ax))
        assert len(messages) == 1
        assert "'stacked'" in messages[0] and "ax=..." in messages[0]
        _quiet(lambda: model.plot(ax=ax))
        _quiet(lambda: model.plot(stacked=True, ax=ax))
    finally:
        plt.close("all")
