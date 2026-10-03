"""Fits whose likelihood has no finite maximum say so (#392).

Each model here used to return, in silence, wherever its search gave up
on data whose likelihood keeps increasing towards a limit. Each now warns
once, at the caller's line, and still returns the model it reached; the
univariate fits refuse such data (a point mass at the edge of a
truncation window too). Every case is paired with an ordinary fit of the
same model that stays silent.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.degradation import (
    DestructiveDegradation,
    GammaProcess,
    WienerProcess,
)
from surpyval.multivariate import (
    Clayton,
    Frank,
    Gaussian,
    Gumbel,
    StudentT,
)
from surpyval.multivariate.parametric.copula.copula import (
    _perfect_dependence,
)
from surpyval.multivariate.parametric.data import MultivariateSurpyvalData


def _caught(fit, *args, **kwargs):
    """``fit(*args, **kwargs)`` and every warning it gave."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fit(*args, **kwargs)
    return out, caught


def _one_no_maximum(caught, *phrases):
    """Exactly one warning, the no-maximum one, at this file's line."""
    assert len(caught) == 1, [str(w.message)[:80] for w in caught]
    (w,) = caught
    assert w.category is UserWarning
    message = str(w.message)
    assert message.startswith("No finite maximum: "), message
    for phrase in phrases:
        assert phrase in message, message
    assert w.filename == __file__, w.filename


def _silent(caught):
    assert not caught, [str(w.message)[:80] for w in caught]


# ---------------------------------------------------------------------------
# Copulas: data at a Frechet bound
# ---------------------------------------------------------------------------
_U = (np.arange(1, 21) - 0.3) / 20.4
_X1 = 10.0 * (-np.log1p(-_U)) ** 0.5
_MARGINS = [sp.Weibull, sp.Weibull]


def _dependent():
    # Neighbours swapped: positively dependent, not perfectly
    x2 = 5.0 * (-np.log1p(-_U[np.arange(20) ^ 1])) ** (1 / 1.4)
    return np.column_stack([_X1, x2])


@pytest.mark.parametrize("copula", [Clayton, Gumbel, Frank, Gaussian])
def test_comonotone_data_warn(copula):
    # x2 = x1 / 2: Clayton returned theta 3.16e6, Frank 1.24e7, Gumbel
    # 105.5 (with a log-likelihood of inf, and 230 raw numpy overflow
    # warnings), Gaussian rho at its cap of 0.9999, all without a word.
    X = np.column_stack([_X1, _X1 / 2])
    model, caught = _caught(copula.fit, X, margins=_MARGINS)
    _one_no_maximum(caught, "perfectly concordant", "comonotone")
    assert np.isfinite(model.params).all()


def test_the_joint_fit_warns_once_too():
    X = np.column_stack([_X1, _X1 / 2])
    _, caught = _caught(Clayton.fit, X, margins=_MARGINS, how="MLE")
    _one_no_maximum(caught, "Clayton", "theta grows without bound")


@pytest.mark.parametrize("copula", [Frank, Gaussian])
def test_countermonotone_data_warn_where_the_family_reaches_them(copula):
    X = np.column_stack([_X1, 100.0 - _X1])
    _, caught = _caught(copula.fit, X, margins=_MARGINS)
    _one_no_maximum(caught, "perfectly discordant", "countermonotone")


def test_a_family_without_negative_dependence_is_not_told_about_it():
    # Clayton cannot reach the countermonotone copula; it goes to its
    # independence end, a valid copula, and says nothing.
    X = np.column_stack([_X1, 100.0 - _X1])
    _, caught = _caught(Clayton.fit, X, margins=_MARGINS)
    _silent(caught)


@pytest.mark.parametrize("copula", [Clayton, Gumbel, Frank, Gaussian])
def test_ordinary_copula_fits_are_silent(copula):
    _, caught = _caught(copula.fit, _dependent(), margins=_MARGINS)
    _silent(caught)


@pytest.mark.parametrize("copula", [Gaussian, StudentT])
@pytest.mark.parametrize("sign", [1, -1])
def test_541_perfectly_dependent_elliptical_fit_reaches_a_valid_rho(
    copula, sign
):
    # rho was clipped to +-0.9999 inside the formulas; the search now runs
    # it as far as a double goes, keeps it inside (-1, 1), and warns once.
    X = np.column_stack([_X1, _X1 / 2 if sign > 0 else 100.0 - _X1])
    model, caught = _caught(copula.fit, X, margins=_MARGINS)
    kind = "comonotone" if sign > 0 else "countermonotone"
    _one_no_maximum(caught, kind)
    rho = model.params[0]
    assert 0.998 < sign * rho < 1.0
    assert np.isfinite(model.copula.pdf(0.3, 0.4, *model.params))


@pytest.mark.parametrize("copula", [Gaussian, StudentT])
@pytest.mark.parametrize("how", ["IFM", "MLE"])
def test_541_rho_running_to_one_on_censored_data_warns(copula, how):
    # The second series is right censored below the comonotone value x1/2
    # in every row: the likelihood rises (to a plateau in floating point)
    # as rho tends to 1. No row is observed in both series, so the data
    # are not "perfectly dependent"; the fit stopped silently at 0.99999.
    X = np.column_stack([_X1, 0.4 * _X1])
    c = np.column_stack([np.zeros(20, int), np.ones(20, int)])
    margins = [sp.Weibull.from_params([10, 2]), sp.Weibull.from_params([5, 2])]
    model, caught = _caught(copula.fit, X, c=c, margins=margins, how=how)
    _one_no_maximum(caught, "rho runs to 1", "comonotone")
    assert 0.999 < model.params[0] < 1.0


def test_perfect_dependence_is_decided_exactly():
    def sign(x, c=None):
        c = None if c is None else np.asarray(c)
        data = MultivariateSurpyvalData(np.asarray(x, float), c=c)
        return _perfect_dependence(data)[0]

    rising = [[1, 2], [2, 3], [3, 5]]
    assert sign(rising) == 1
    assert sign(rising[::-1]) == 1  # order of the rows is irrelevant
    assert sign([[1, 5], [2, 3], [3, 2]]) == -1
    # A repeated row is tied in both coordinates: still perfect
    assert sign(rising + [[2, 3]]) == 1
    # One discordant pair, or a tie in one coordinate only, is not
    assert sign([[1, 2], [2, 3], [3, 2.5]]) == 0
    assert sign([[1, 2], [2, 3], [2, 5]]) == 0
    # A censored row does not count; one distinct pair decides nothing
    assert sign(rising, c=[[0, 0], [0, 0], [0, 1]]) == 1
    assert sign([[1, 2], [1, 2]]) == 0


# ---------------------------------------------------------------------------
# Mixture: a component collapsed onto a point mass
# ---------------------------------------------------------------------------
_MIX = np.r_[np.linspace(2.0, 6.0, 10), np.linspace(20.0, 40.0, 10)]


def test_a_mixture_component_on_a_point_mass_warns():
    # 10 of 20 rows at 3.0: the first Weibull came back with beta 9100.
    model = sp.MixtureModel(dist=sp.Weibull, m=2)
    _, caught = _caught(model.fit, np.r_[np.full(10, 3.0), _MIX[10:]])
    _one_no_maximum(caught, "component 0", "point mass", "(3)")
    assert np.isfinite(model.params).all()


def test_an_ordinary_mixture_fit_is_silent():
    model = sp.MixtureModel(dist=sp.Weibull, m=2)
    _, caught = _caught(model.fit, _MIX)
    _silent(caught)


# ---------------------------------------------------------------------------
# Degradation: noise-free readings
# ---------------------------------------------------------------------------
_T = np.tile(np.arange(0.0, 110.0, 10.0), 5)
_UNIT = np.repeat(np.arange(5), 11)
_Z = np.repeat([1.0, 1.0, 2.0, 2.0, 3.0], 11)
_STEPS = 5.0 + 3.0 * np.sin(np.arange(50.0)).reshape(5, 10)
_NOISY = np.hstack([np.r_[0.0, np.cumsum(s)] for s in _STEPS])


def test_a_noise_free_gamma_process_warns():
    # y = t / 2 exactly: alpha, beta = 1e6, 2e6, the end of the search
    model, caught = _caught(GammaProcess.fit, _T, 0.5 * _T, _UNIT, 100.0)
    _one_no_maximum(caught, "alpha", "noise-free", "0.5 per unit time")
    assert np.isfinite([model.alpha, model.beta]).all()


def test_a_noise_free_gamma_process_with_stress_warns():
    # On its stress clock: alpha 3.1e12, in silence
    y = 0.5 * np.exp(0.3 * _Z) * _T
    _, caught = _caught(GammaProcess.fit, _T, y, _UNIT, 100.0, Z=_Z)
    _one_no_maximum(caught, "on the fitted stress clock")


def test_a_gamma_process_from_a_data_frame_warns_at_the_caller():
    df = pd.DataFrame({"x": _T, "y": 0.5 * _T, "i": _UNIT})
    _, caught = _caught(GammaProcess.fit_from_df, df, threshold=100.0)
    _one_no_maximum(caught, "alpha")


@pytest.mark.parametrize("Z", [None, _Z])
def test_ordinary_gamma_process_fits_are_silent(Z):
    _, caught = _caught(GammaProcess.fit, _T, _NOISY, _UNIT, 100.0, Z=Z)
    _silent(caught)


def test_the_wiener_process_still_refuses_noise_free_readings():
    with pytest.raises(ValueError, match="sigma is 0"):
        WienerProcess.fit(_T, 0.5 * _T, _UNIT, threshold=100.0)


_AGES = np.repeat([10.0, 20.0, 30.0, 40.0], 5)


@pytest.mark.parametrize(
    "y, kwargs",
    [
        (np.exp(4.0 - 0.02 * _AGES), {}),  # sigma was 9.9e-16
        (np.exp(4.0 - 0.02 * _AGES), {"transform": "best"}),
        (50.0 - 0.5 * _AGES, {"distribution": "Normal"}),  # 1.3e-14
    ],
)
def test_noise_free_destructive_readings_warn(y, kwargs):
    model, caught = _caught(
        DestructiveDegradation.fit, _AGES, y, threshold=20.0, **kwargs
    )
    _one_no_maximum(caught, "sigma", "noise-free")
    assert np.isfinite(model.sigma)


def test_an_ordinary_destructive_fit_is_silent():
    y = np.exp(4.0 - 0.02 * _AGES + 0.1 * np.sin(np.arange(20.0)))
    _, caught = _caught(DestructiveDegradation.fit, _AGES, y, threshold=20.0)
    _silent(caught)


# ---------------------------------------------------------------------------
# BetaGeometric: the Geometric limit
# ---------------------------------------------------------------------------
_CYCLES = dict(
    x=np.array([1, 2, 2, 3, 3, 3, 4, 4, 5, 6, 7, 9]),
    c=np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]),
    n=np.array([1, 1, 2, 1, 1, 1, 1, 3, 1, 1, 1, 1]),
)


def test_a_beta_geometric_fit_in_its_geometric_limit_warns():
    # a, b = 9.8e4, 3.5e5 in silence, and every bound nan
    model, caught = _caught(sp.BetaGeometric.fit, **_CYCLES)
    _one_no_maximum(caught, "Geometric limit", "use Geometric")
    assert np.isfinite(model.params).all()


def test_a_beta_geometric_fit_from_a_data_frame_warns_at_the_caller():
    df = pd.DataFrame(_CYCLES)
    _, caught = _caught(
        sp.BetaGeometric.fit_from_df, df, x_col="x", c_col="c", n_col="n"
    )
    _one_no_maximum(caught, "use Geometric")


def test_an_ordinary_beta_geometric_fit_is_silent():
    x = sp.BetaGeometric.random(200, 3.0, 5.0, random_state=0)
    model, caught = _caught(sp.BetaGeometric.fit, x)
    _silent(caught)
    np.testing.assert_allclose(model.params, [6.47, 11.79], rtol=1e-2)


# ---------------------------------------------------------------------------
# Univariate: a point mass at the edge of a truncation window is refused,
# as one inside every row's set already was (#462)
# ---------------------------------------------------------------------------
_EDGE = [
    # A failure at 1, observable only up to 1, one at 2 and one before 3:
    # a spike at 2 explains all three (the first through f(1) / F(1)).
    # Weibull returned beta 455.6 and Normal sigma 0.037, in silence.
    dict(x=[1.0, 2.0, 3.0], c=[0, 0, -1], tr=[1.0, np.inf, np.inf]),
    # Two failures at 2 and one in (3, 5] observed only after 3: a spike
    # at 2 leaves the third's conditional mass piled just above 3.
    # Weibull returned beta 86.5, and Normal sigma 7.7e-9 after 28 s.
    dict(
        x=[[2.0, 2.0], [2.0, 2.0], [3.0, 5.0]],
        c=[0, 0, 2],
        tl=[0.0, 0.0, 3.0],
    ),
]


@pytest.mark.parametrize("data", _EDGE)
@pytest.mark.parametrize("dist", [sp.Weibull, sp.Normal, sp.LogNormal])
def test_a_point_mass_at_a_truncation_edge_is_refused(dist, data):
    with pytest.raises(ValueError, match="no maximum"):
        dist.fit(**data)


@pytest.mark.parametrize("dist", [sp.Weibull, sp.Normal])
def test_a_set_short_of_its_window_edge_still_fits(dist):
    # As the second case above, but observed from 2.5: a spike at 2
    # leaves the third row's mass just above 2.5, outside (3, 5]
    data = dict(_EDGE[1], tl=[0.0, 0.0, 2.5])
    model, _ = _caught(dist.fit, **data)
    assert np.isfinite(model.params).all()
