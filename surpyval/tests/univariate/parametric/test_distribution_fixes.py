"""
Regression tests for the distribution-level defect batch (#257).
"""

import inspect

import numpy as np
import pytest

from surpyval import (
    Bernoulli,
    Beta,
    Binomial,
    ExactEventTime,
    Exponential,
    ExpoWeibull,
    Gamma,
    Gumbel,
    GumbelLEV,
    Logistic,
    LogNormal,
    Normal,
    Rayleigh,
    Weibull,
)
from surpyval.univariate.parametric.parametric_fitter import (
    ParametricFitter,
)
from surpyval.utils.surpyval_data import SurpyvalData


def test_lognormal_fits_negative_mu():
    # mu (mean of log-data) is any real; a (0, None) bound crashed every
    # fit with geometric mean < 1.
    np.random.seed(0)
    m = LogNormal.fit(LogNormal.random(500, -0.5, 0.5))
    assert m.params[0] == pytest.approx(-0.5, abs=0.1)


def test_bernoulli_fit_works_for_general_inputs():
    assert Bernoulli.fit([0, 1, 1, 0, 1]).params[0] == pytest.approx(0.6)
    assert Bernoulli.fit([1, 1]).params[0] == 1.0
    assert Bernoulli.fit([0, 1], n=[3, 1]).params[0] == pytest.approx(0.25)
    with pytest.raises(ValueError):
        Bernoulli.fit([0, 2, 1])
    assert Bernoulli.from_params(0.3).params[0] == pytest.approx(0.3)


def test_exponential_offset_mpp_rr_x_inversion():
    np.random.seed(0)
    x = Exponential.random(2000, 0.5) + 10
    m = Exponential.fit(x, how="MPP", rr="x", offset=True)
    assert m.params[0] == pytest.approx(0.5, rel=0.3)
    assert m.gamma == pytest.approx(10.0, abs=0.5)


def test_gamma_censored_mpp_rr_x_does_not_crash():
    # Was a LinAlgError: rr="x" regressed the filtered y against the raw
    # x whenever censoring filtered any point (#257). Gamma declines MPP
    # entirely now (#158), so the crashing path is unreachable -- the
    # refusal arrives before any regression is attempted.
    np.random.seed(1)
    x = Gamma.random(300, 3, 2)
    c = (x > 2.5).astype(int)
    with pytest.raises(ValueError, match="does not work with probability"):
        Gamma.fit(np.minimum(x, 2.5), c=c, how="MPP", rr="x")


@pytest.mark.parametrize("rr", ["x", "y"])
def test_gamma_offset_mpp_recovers_offset(rr):
    # Gamma no longer offers MPP at all (#158): the probability plot's
    # own y-axis is the inverse incomplete gamma, which needs the shape
    # being estimated, so the fit regressed against an axis built from a
    # guess and returned a confident wrong answer. It now refuses.
    np.random.seed(2)
    x = Gamma.random(300, 3, 2) + 10
    with pytest.raises(ValueError, match="does not work with probability"):
        Gamma.fit(x, how="MPP", rr=rr, offset=True)

    # The offset recovery this test existed to protect still holds under
    # the estimators Gamma does support.
    m = Gamma.fit(x, offset=True)
    assert m.gamma == pytest.approx(10.0, abs=1.5)
    assert m.params[0] == pytest.approx(3.0, rel=0.5)


def test_expo_weibull_tail_is_stable():
    # 1 - (1 - e^-t)^mu underflowed to exactly 0 once e^-t < 1e-16.
    s = float(ExpoWeibull.sf(8, 3, 4, 1.2))
    assert 0.0 < s < 1e-18
    assert np.isfinite(float(ExpoWeibull.Hf(8, 3, 4, 1.2)))
    assert np.isfinite(float(ExpoWeibull.log_sf(8, 3, 4, 1.2)))
    # Moderate-x values agree with the naive form.
    naive = 1 - (1 - np.exp(-((2 / 3) ** 4))) ** 1.2
    assert float(ExpoWeibull.sf(2, 3, 4, 1.2)) == pytest.approx(
        naive, abs=1e-12
    )


def test_probability_plot_transform_roundtrips():
    F = np.array([0.05, 0.3, 0.7, 0.95])
    assert np.allclose(
        Exponential.mpp_inv_y_transform(Exponential.mpp_y_transform(F)), F
    )
    assert np.allclose(
        GumbelLEV.mpp_inv_y_transform(GumbelLEV.mpp_y_transform(F)), F
    )
    assert np.allclose(
        Beta.mpp_inv_y_transform(Beta.mpp_y_transform(F, 2.0, 3.0), 2.0, 3.0),
        F,
    )


def test_logistic_log_functions_deep_tail():
    assert np.isfinite(Logistic.log_sf(np.array([-800000.0]), 0.0, 1.0)).all()
    assert np.isfinite(Logistic.log_ff(np.array([800000.0]), 0.0, 1.0)).all()


def test_exact_event_time_informative_error():
    with pytest.raises(ValueError, match="right-censored"):
        ExactEventTime.fit(np.array([1.0, 2.0, 3.0]), c=np.array([1, 1, 1]))


# --- from_params argument names and structural rejection -----------------
#
# Bernoulli and ExactEventTime used to name the base's ``params`` argument
# ``p`` and ``T``, so positional calls worked and keyword calls raised.
# Bernoulli's was worse than a rename: the base's ``p`` is the proportion
# that never fails, so the same keyword meant two unrelated things on
# sibling classes. All three now match ParametricFitter.from_params, and
# reject the structural arguments they cannot honour.


@pytest.mark.parametrize(
    "dist, params, expected",
    [
        (Bernoulli, 0.3, [0.3]),
        (ExactEventTime, 10, [10]),
        (Binomial, [5, 0.3], [5.0, 0.3]),
    ],
)
def test_from_params_accepts_the_params_keyword(dist, params, expected):
    by_keyword = dist.from_params(params=params)
    by_position = dist.from_params(params)
    assert by_keyword.params == pytest.approx(expected)
    assert by_position.params == pytest.approx(expected)


@pytest.mark.parametrize(
    "dist, params",
    [
        (Bernoulli, 0.3),
        (ExactEventTime, 10),
        (Binomial, [5, 0.3]),
    ],
)
@pytest.mark.parametrize("structural", ["gamma", "p", "f0"])
def test_from_params_rejects_unsupported_structural_args(
    dist, params, structural
):
    with pytest.raises(ValueError, match="does not support"):
        dist.from_params(params, **{structural: 0.5})


def test_from_params_signature_matches_the_base():
    # The narrower signatures could not be called through a
    # ParametricFitter reference, which is what made this a real bug and
    # not a naming preference.
    base = set(inspect.signature(ParametricFitter.from_params).parameters)
    for dist in (Bernoulli, Binomial, ExactEventTime):
        own = set(inspect.signature(type(dist).from_params).parameters)
        assert base <= own, dist.name


# --- Rayleigh's initial guess had to be a sequence ------------------------
#
# Rayleigh is the only single-parameter distribution here, and its
# _parameter_initialiser returned a bare scalar for the non-offset case.
# `np.array(init)` in _initial_guess then produced a 0-dimensional array
# rather than a length-1 one, and the lfp and zi paths concatenate the p
# and f0 seeds onto it -- which a 0-d array cannot do.


def test_rayleigh_initial_guess_is_a_sequence():
    seed = Rayleigh._parameter_initialiser(
        SurpyvalData(np.array([1.0, 2.0, 3.0, 4.0]))
    )
    assert np.array(seed).ndim == 1


@pytest.mark.parametrize("structural", ["lfp", "zi"])
def test_rayleigh_fits_with_lfp_and_zi(structural):
    np.random.seed(0)
    x = Rayleigh.random(200, 10.0)
    if structural == "zi":
        x = np.concatenate([x, np.zeros(10)])
    model = Rayleigh.fit(x, **{structural: True})
    # The sigma estimate is unaffected; the point is that it runs at all.
    assert model.params[0] == pytest.approx(9.92, abs=0.5)


# --- _parameter_initialiser takes a SurpyvalData -------------------------
#
# It used to take (x, c=None, n=None, t=None, offset=False), and every
# implementation re-established the conventions that SurpyvalData had
# already guaranteed -- inconsistently. Normal indexed with c and Gumbel
# tested membership on it before either was defaulted, so both raised
# TypeError for the signature the base class documented; every caller
# inside the package passed c and n, which is why it went unnoticed.
# There is now nothing to default: the argument is the normalised
# object, so c, n and t are always arrays.


@pytest.mark.parametrize(
    "dist",
    [Normal, Gumbel, GumbelLEV, Weibull, LogNormal, Logistic, Rayleigh],
)
def test_parameter_initialiser_takes_surpyval_data(dist):
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.5])
    seed = dist._parameter_initialiser(SurpyvalData(x))
    assert np.asarray(seed).ndim == 1
    assert np.isfinite(np.asarray(seed, dtype=float)).all()


@pytest.mark.parametrize(
    "dist",
    [Normal, Gumbel, GumbelLEV, Weibull, LogNormal, Logistic, Rayleigh],
)
def test_parameter_initialiser_rejects_loose_arrays(dist):
    # The old signature is gone rather than deprecated. Passing a bare
    # array reaches the ``.x`` attribute access and fails loudly, which
    # is the point: a silent partial acceptance is what let the c=None
    # divergence above survive.
    with pytest.raises(AttributeError):
        dist._parameter_initialiser(np.array([1.0, 2.0, 3.0]))
