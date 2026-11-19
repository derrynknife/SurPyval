"""The hypoexponential distribution: a sum of exponential stages."""

import warnings
from types import SimpleNamespace

import autograd.numpy as anp
import numpy as np
import pandas as pd
import pytest
from autograd import grad, hessian
from scipy import integrate

import surpyval
from surpyval import (
    Exponential,
    Gamma,
    Hypoexponential,
    Parametric,
    SurpyvalData,
    Weibull,
)
from surpyval.tests._helpers import json_round_trip
from surpyval.univariate.parametric.distributions.hypoexponential import (
    Hypoexponential_,
    _slowest_first,
)

RATES = (0.5, 1.5, 3.0)


# -- closed forms -----------------------------------------------------------


def test_sf_starts_at_one_and_decreases_to_zero():
    x = np.linspace(0.0, 60.0, 601)
    sf = Hypoexponential.sf(x, *RATES)
    assert sf[0] == 1.0
    assert np.all(np.diff(sf) <= 0)
    assert sf[-1] == pytest.approx(0.0, abs=1e-12)
    assert np.all((0.0 <= sf) & (sf <= 1.0))
    assert np.allclose(sf + Hypoexponential.ff(x, *RATES), 1.0)


def test_two_stage_matches_direct_convolution():
    a, b = 0.7, 2.3
    x = np.linspace(0.0, 10.0, 101)
    direct = (b * np.exp(-a * x) - a * np.exp(-b * x)) / (b - a)
    assert np.allclose(Hypoexponential.sf(x, a, b), direct, atol=1e-14)
    density = a * b / (b - a) * (np.exp(-a * x) - np.exp(-b * x))
    assert np.allclose(Hypoexponential.df(x, a, b), density, atol=1e-14)


def test_mean_and_variance_are_sums_of_stage_moments():
    rates = np.array(RATES)
    assert Hypoexponential.mean(*RATES) == pytest.approx(np.sum(1 / rates))
    model = Hypoexponential.from_params(RATES)
    assert model.mean() == pytest.approx(np.sum(1 / rates))
    assert model.var() == pytest.approx(np.sum(1 / rates**2))
    # moments against the integrated density, up to order 4
    for m in range(1, 5):
        expected = integrate.quad(
            lambda t: t**m * Hypoexponential.df(t, *RATES), 0, np.inf
        )[0]
        assert Hypoexponential.moment(m, *RATES) == pytest.approx(
            expected, rel=1e-8
        )


def test_density_integrates_to_one_and_matches_sf_slope():
    total = integrate.quad(lambda t: Hypoexponential.df(t, *RATES), 0, np.inf)
    assert total[0] == pytest.approx(1.0, abs=1e-10)
    x = np.linspace(0.1, 8.0, 50)
    h = 1e-6
    slope = (
        Hypoexponential.sf(x - h, *RATES) - Hypoexponential.sf(x + h, *RATES)
    ) / (2 * h)
    assert np.allclose(Hypoexponential.df(x, *RATES), slope, rtol=1e-6)


def test_hazard_identities():
    x = np.linspace(0.1, 8.0, 50)
    sf = Hypoexponential.sf(x, *RATES)
    assert np.allclose(Hypoexponential.Hf(x, *RATES), -np.log(sf))
    assert np.allclose(
        Hypoexponential.hf(x, *RATES), Hypoexponential.df(x, *RATES) / sf
    )
    # the hazard starts at zero (every stage must complete) and tends
    # to the slowest stage rate
    assert Hypoexponential.hf(0.0, *RATES) == pytest.approx(0.0)
    assert Hypoexponential.hf(200.0, *RATES) == pytest.approx(
        min(RATES), rel=1e-6
    )


def test_qf_inverts_ff():
    u = np.array([1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-6])
    q = Hypoexponential.qf(u, *RATES)
    assert np.all(np.diff(q) > 0)
    assert np.allclose(Hypoexponential.ff(q, *RATES), u, atol=1e-12)
    assert Hypoexponential.qf(0.0, *RATES) == 0.0
    assert Hypoexponential.qf(1.0, *RATES) == np.inf
    scalar = Hypoexponential.qf(0.5, *RATES)
    assert np.ndim(scalar) == 0
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        assert np.isnan(Hypoexponential.qf(1.5, *RATES))


def test_single_rate_is_exponential():
    x = np.linspace(0.0, 10.0, 51)
    for fn in ("sf", "ff", "df", "hf", "Hf"):
        assert np.allclose(
            getattr(Hypoexponential, fn)(x, 2.0),
            getattr(Exponential, fn)(x, 2.0),
        )
    assert Hypoexponential.mean(2.0) == pytest.approx(0.5)
    u = np.array([0.1, 0.5, 0.9])
    assert np.allclose(Hypoexponential.qf(u, 2.0), Exponential.qf(u, 2.0))


def test_random_matches_sf(seed=3):
    np.random.seed(seed)
    draws = Hypoexponential.random(100_000, *RATES)
    assert draws.shape == (100_000,)
    assert np.all(draws > 0)
    for u in (0.1, 0.5, 0.9):
        q = Hypoexponential.qf(u, *RATES)
        assert np.mean(draws > q) == pytest.approx(1 - u, abs=0.006)
    assert draws.mean() == pytest.approx(
        Hypoexponential.mean(*RATES), rel=0.02
    )
    # tuple sizes broadcast per stage
    assert Hypoexponential.random((4, 5), *RATES).shape == (4, 5)


def test_model_random_goes_through_qf():
    model = Hypoexponential.from_params(RATES)
    np.random.seed(5)
    draws = model.random(50_000)
    assert draws.shape == (50_000,)
    q = Hypoexponential.qf(0.5, *RATES)
    assert np.mean(draws > q) == pytest.approx(0.5, abs=0.01)


# -- the model object --------------------------------------------------------


def test_from_params_builds_a_variable_arity_model():
    two = Hypoexponential.from_params([1.0, 2.0])
    five = Hypoexponential.from_params([1.0, 2.0, 3.0, 4.0, 5.0])
    assert isinstance(two, Parametric)
    assert (two.k, five.k) == (2, 5)
    assert five.dist.parameter_names == [
        "lambda_1",
        "lambda_2",
        "lambda_3",
        "lambda_4",
        "lambda_5",
    ]
    assert five.dist.name == "Hypoexponential"
    # a fresh instance per arity; the exported singleton is untouched
    assert Hypoexponential.k == 0
    assert isinstance(five.dist, Hypoexponential_)
    x = np.array([0.5, 1.0, 2.0])
    assert np.allclose(five.sf(x), Hypoexponential.sf(x, 1, 2, 3, 4, 5))
    assert "lambda_5" in repr(five)


def test_from_params_supports_offset_lfp_and_zi():
    shifted = Hypoexponential.from_params(RATES, gamma=2.0)
    assert shifted.sf(1.0) == 1.0
    assert shifted.sf(3.0) == pytest.approx(Hypoexponential.sf(1.0, *RATES))
    assert shifted.mean() == pytest.approx(Hypoexponential.mean(*RATES) + 2.0)
    cured = Hypoexponential.from_params(RATES, lfp_p=0.8)
    assert cured.sf(1e6) == pytest.approx(0.2)


def test_serialisation_round_trip():
    model = Hypoexponential.from_params(RATES)
    d = model.to_dict()
    assert d["distribution"] == "Hypoexponential"
    assert d["param_names"] == ["lambda_1", "lambda_2", "lambda_3"]
    restored = surpyval.from_dict(json_round_trip(d))
    assert isinstance(restored, Parametric)
    assert restored.k == 3
    assert restored.dist.parameter_names == model.dist.parameter_names
    x = np.linspace(0.0, 10.0, 21)
    assert np.array_equal(restored.sf(x), model.sf(x))
    assert np.array_equal(restored.ff(x), model.ff(x))
    assert restored.mean() == model.mean()
    # the class-level reader and JSON file paths agree too
    assert np.array_equal(
        Parametric.from_dict(json_round_trip(d)).sf(x), model.sf(x)
    )
    # a different arity round-trips to a different arity
    six = Hypoexponential.from_params([1, 2, 3, 4, 5, 6])
    assert surpyval.from_dict(json_round_trip(six.to_dict())).k == 6


def test_serialisation_round_trip_json_file(tmp_path):
    model = Hypoexponential.from_params(RATES, gamma=1.0)
    fp = tmp_path / "hypo.json"
    model.to_json(fp)
    restored = surpyval.from_json(fp)
    x = np.array([1.5, 3.0, 6.0])
    assert np.array_equal(restored.sf(x), model.sf(x))
    assert restored.gamma == 1.0


# -- validation ---------------------------------------------------------------


@pytest.mark.parametrize(
    "rates, match",
    [
        ([], "non-empty"),
        ([[1.0, 2.0]], "one-dimensional"),
        ([1.0, -2.0], "strictly positive"),
        ([1.0, 0.0], "strictly positive"),
        ([1.0, np.inf], "finite"),
        ([1.0, np.nan], "finite"),
    ],
)
def test_invalid_rates_are_refused(rates, match):
    with pytest.raises(ValueError, match=match):
        Hypoexponential.from_params(rates)
    with pytest.raises(ValueError, match=match):
        Hypoexponential.sf(1.0, *rates)


# -- equal and close rates ----------------------------------------------------

_FUNCTIONS = ("sf", "ff", "df", "hf", "Hf", "log_sf", "log_ff", "log_df")


@pytest.mark.parametrize("m, rate", [(2, 1.0), (3, 2.0), (5, 0.3)])
def test_equal_rates_are_the_erlang(m, rate):
    # from the origin, through the series and the matrix forms, to a tail
    # where sf and df underflow but their logs do not
    x = np.array([1e-12, 1e-3, 0.1, 0.5, 1.0, 3.0, 10.0, 50.0, 2000.0])
    x = x / rate
    rates = [rate] * m
    for fn in _FUNCTIONS:
        got = getattr(Hypoexponential, fn)(x, *rates)
        want = getattr(Gamma, fn)(x, m, rate)
        assert np.allclose(got, want, rtol=1e-12, atol=0), fn
    model = Hypoexponential.from_params(rates)
    assert model.k == m
    assert np.allclose(model.sf(x), Gamma.sf(x, m, rate), rtol=1e-12, atol=0)
    u = np.array([1e-9, 0.1, 0.5, 0.9, 1 - 1e-9])
    assert np.allclose(
        Hypoexponential.qf(u, *rates), Gamma.qf(u, m, rate), rtol=1e-10
    )


def _exact(x, rates):
    """sf, ff and df from the generator's matrix exponential in 150-digit
    arithmetic."""
    mp = pytest.importorskip("mpmath")
    m = len(rates)
    with mp.workdps(150):
        Q = mp.zeros(m + 1, m + 1)
        for i, r in enumerate(rates):
            Q[i, i] = -mp.mpf(float(r))
            Q[i, i + 1] = mp.mpf(float(r))
        row = mp.expm(Q * mp.mpf(float(x)), method="taylor")
        sf = mp.fsum(row[0, j] for j in range(m))
        return float(sf), float(row[0, m]), float(rates[-1] * row[0, m - 1])


@pytest.mark.parametrize(
    "rates",
    [
        # four rates 1e-5 apart (relative): the partial fractions gave
        # twice the density and a CDF 18% off at x = 2
        0.5 * (1 + 1e-5 * np.array([-1.5, -0.5, 0.5, 1.5])),
        np.array([1.0, 1.0 + 1e-9, 3.0]),
        np.array([0.2, 0.2, 0.2 * (1 + 1e-3), 5.0]),
    ],
)
def test_close_rates_keep_their_precision(rates):
    x = np.array([0.05, 0.5, 2.0, 8.0, 20.0, 60.0])
    for fn, values in zip(
        ("sf", "ff", "df"), zip(*[_exact(t, rates) for t in x])
    ):
        got = getattr(Hypoexponential, fn)(x, *rates)
        assert np.allclose(got, values, rtol=1e-12, atol=0), fn


def test_two_close_rates_match_the_two_stage_closed_form():
    # (b e^(-a x) - a e^(-b x)) / (b - a) in a form that does not cancel:
    # e^(-a x) (1 + a (1 - e^(-(b - a) x)) / (b - a))
    a, b = 1.0, 1.0 + 1e-4
    x = np.array([1.5, 3.0, 10.0, 30.0])
    d = b - a
    sf = np.exp(-a * x) * (1 + a * -np.expm1(-d * x) / d)
    assert np.allclose(Hypoexponential.sf(x, a, b), sf, rtol=1e-12)


def test_moments_with_equal_rates_are_the_erlang():
    for k in range(1, 5):
        erlang = np.prod(np.arange(3, 3 + k)) / 2.0**k
        assert Hypoexponential.moment(k, 2.0, 2.0, 2.0) == pytest.approx(
            erlang, rel=1e-14
        )
    model = Hypoexponential.from_params([2.0, 2.0, 2.0])
    assert model.var() == pytest.approx(3 / 4)


# -- derivatives --------------------------------------------------------------


def _central(f, at, eps=1e-6):
    at = np.asarray(at, dtype=float)
    steps = eps * np.eye(len(at))
    return np.array([(f(at + s) - f(at - s)) / (2 * eps) for s in steps])


@pytest.mark.parametrize("rates", [(0.5, 1.5, 3.0), (1.0, 1.0), (0.7,)])
@pytest.mark.parametrize("fn", _FUNCTIONS)
def test_derivatives_match_finite_differences(fn, rates):
    # The likelihood is differentiated in the rates (to second order, for
    # the Hessian) and, with an offset, in x
    x = np.array([0.05, 0.4, 1.3, 2.5, 7.0, 30.0])

    def total(r):
        return anp.sum(getattr(Hypoexponential, fn)(x, *r))

    rates = np.array(rates)
    gradient = grad(total)
    assert np.allclose(
        gradient(rates), _central(total, rates), rtol=1e-6, atol=1e-8
    )
    assert np.allclose(
        hessian(total)(rates),
        _central(gradient, rates),
        rtol=1e-5,
        atol=1e-7,
    )
    in_x = grad(lambda t: anp.sum(getattr(Hypoexponential, fn)(t, *rates)))
    eps = 1e-6
    by_x = (
        getattr(Hypoexponential, fn)(x + eps, *rates)
        - getattr(Hypoexponential, fn)(x - eps, *rates)
    ) / (2 * eps)
    assert np.allclose(in_x(x), by_x, rtol=1e-6, atol=1e-8)


# -- fitting ------------------------------------------------------------------


def _neg_ll(model, data):
    return float(
        model.dist._neg_ll_func(
            data, *model.params, model.gamma, model.f0, model.lfp_p
        )
    )


def test_fit_recovers_the_rates():
    x = Hypoexponential.random(3000, 0.5, 2.0, random_state=11)
    model = Hypoexponential.fit(x, stages=2)
    assert isinstance(model, Parametric)
    assert model.k == 2
    assert model.maximum == "verified"
    se = model.standard_errors()
    assert np.all(np.abs(model.params - [0.5, 2.0]) < 3 * se)
    assert model.params[0] < model.params[1]


def test_fit_with_censoring_and_truncation_is_a_maximum():
    x = Hypoexponential.random(400, 0.3, 1.2, random_state=4)
    keep = x > 0.5
    x = x[keep]
    c = (x > 8).astype(int)
    x = np.minimum(x, 8)
    model = Hypoexponential.fit(x, c, tl=0.5, stages=2)
    assert model.maximum == "verified"
    data = SurpyvalData(x, c, tl=0.5)
    best = _neg_ll(model, data)
    for step in ([1, 0], [0, 1], [1, 1], [1, -1]):
        for sign in (1, -1):
            moved = Hypoexponential.from_params(
                model.params * (1 + sign * 1e-3 * np.array(step))
            )
            assert _neg_ll(moved, data) > best


def test_erlang_data_give_equal_rates():
    x = Gamma.random(200, 3, 1.0, random_state=2)
    model = Hypoexponential.fit(x, stages=3)
    assert model.maximum == "verified"
    rates = model.params
    assert np.all(np.diff(rates) >= 0)
    assert np.ptp(rates) < 1e-2 * rates.mean()
    # an Erlang of three stages is a Hypoexponential of three: the fit is
    # at least as likely as the best of them
    erlang = Gamma.fit(x, fixed={"alpha": 3})
    assert model.log_likelihood >= erlang.log_likelihood - 1e-6


def test_one_stage_is_the_exponential():
    x = Weibull.random(50, 10, 1.3, random_state=5)
    c = (x > 15).astype(int)
    one = Hypoexponential.fit(np.minimum(x, 15), c, stages=1)
    exponential = Exponential.fit(np.minimum(x, 15), c)
    assert one.params == pytest.approx(exponential.params, rel=1e-6)


def test_stages_is_required_and_checked():
    x = [1.0, 2.0, 3.0, 4.5]
    with pytest.raises(ValueError, match="stages"):
        Hypoexponential.fit(x)
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="stages"):
            Hypoexponential.fit(x, stages=bad)


def test_a_fitted_model_refits_with_its_own_stages():
    x = Hypoexponential.random(300, 0.4, 1.0, 3.0, random_state=8)
    model = Hypoexponential.fit(x, stages=3)
    again = model.dist.fit(x)
    assert again.k == 3
    assert np.allclose(again.params, model.params)
    assert model.dist.fit(x, stages=2).k == 2


def test_fit_paths_agree():
    x = Hypoexponential.random(200, 0.5, 2.0, random_state=3)
    c = (x > 4).astype(int)
    x = np.minimum(x, 4)
    model = Hypoexponential.fit(x, c, stages=2)
    df = pd.DataFrame({"t": x, "status": c})
    by_df = Hypoexponential.fit_from_df(
        df, x_col="t", c_col="status", stages=2
    )
    by_data = Hypoexponential.fit(SurpyvalData(x, c), stages=2)
    for other in (by_df, by_data):
        assert np.allclose(other.params, model.params, rtol=1e-8)


def test_too_many_stages_warn_and_point_to_fewer():
    # more spread than one exponential stage (a coefficient of variation
    # above 1): no sum of stages fits better than the fewest
    x = Weibull.random(100, 10, 0.8, random_state=3)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = Hypoexponential.fit(x, stages=2)
    assert model.maximum == "no finite maximum"
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1
    assert "stages=1" in messages[0]


def test_offset_and_limited_failure_fits():
    x = 5 + Hypoexponential.random(400, 0.5, 2.0, random_state=9)
    shifted = Hypoexponential.fit(x, stages=2, offset=True)
    assert shifted.gamma < x.min()
    assert shifted.gamma == pytest.approx(5, abs=0.3)
    lives = Hypoexponential.random(400, 0.5, 2.0, random_state=10)
    lives[:100] = np.inf
    c = (lives > 15).astype(int)
    cured = Hypoexponential.fit(np.minimum(lives, 15), c, stages=2, lfp=True)
    assert cured.lfp_p == pytest.approx(0.75, abs=0.06)


def test_a_fitted_model_has_bounds_and_round_trips():
    x = Hypoexponential.random(300, 0.5, 2.0, random_state=12)
    model = Hypoexponential.fit(x, stages=2)
    t = np.array([0.5, 2.0, 6.0])
    bounds = model.cb(t, alpha_ci=0.1)
    assert np.all(bounds[:, 0] <= model.sf(t))
    assert np.all(model.sf(t) <= bounds[:, 1])
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert restored.k == 2
    assert np.array_equal(restored.sf(t), model.sf(t))
    assert np.allclose(restored.cb(t, alpha_ci=0.1), bounds)


def test_rates_are_put_slowest_first_with_their_covariance():
    cov = np.array([[1.0, 0.1, 0.2], [0.1, 2.0, 0.3], [0.2, 0.3, 3.0]])
    model = SimpleNamespace(
        params=np.array([2.0, 1.0]),
        _covariance=cov,
        hess_inv=cov[:2, :2],
        dist=Hypoexponential_("Hypoexponential", 2),
    )
    _slowest_first(model)
    assert np.array_equal(model.params, [1.0, 2.0])
    assert np.array_equal(
        model._covariance,
        [[2.0, 0.1, 0.3], [0.1, 1.0, 0.2], [0.3, 0.2, 3.0]],
    )
    assert np.array_equal(model.hess_inv, [[2.0, 0.1], [0.1, 1.0]])
    assert (model.lambda_1, model.lambda_2) == (1.0, 2.0)


def test_fit_bayes_refuses_and_says_what_to_use():
    with pytest.raises(ValueError, match="stages"):
        Hypoexponential.fit_bayes([1.0, 2.0, 3.0])


def test_from_params_rejects_offset_on_top_of_invalid_rates_first():
    with pytest.raises(ValueError, match="strictly positive"):
        Hypoexponential.from_params([-1.0], gamma=1.0)
