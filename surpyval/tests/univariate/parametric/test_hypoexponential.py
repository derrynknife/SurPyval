"""The hypoexponential distribution: a sum of exponential stages."""

import json

import numpy as np
import pytest
from scipy import integrate

import surpyval
from surpyval import Exponential, Gamma, Hypoexponential, Parametric
from surpyval.univariate.parametric.distributions.hypoexponential import (
    DISTINCT_RATES_TOL,
    Hypoexponential_,
)

RATES = (0.5, 1.5, 3.0)


def _rt(d):
    return json.loads(json.dumps(d))


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
    assert five.dist.param_names == [
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
    cured = Hypoexponential.from_params(RATES, p=0.8)
    assert cured.sf(1e6) == pytest.approx(0.2)


def test_serialisation_round_trip():
    model = Hypoexponential.from_params(RATES)
    d = model.to_dict()
    assert d["distribution"] == "Hypoexponential"
    assert d["param_names"] == ["lambda_1", "lambda_2", "lambda_3"]
    restored = surpyval.from_dict(_rt(d))
    assert isinstance(restored, Parametric)
    assert restored.k == 3
    assert restored.dist.param_names == model.dist.param_names
    x = np.linspace(0.0, 10.0, 21)
    assert np.array_equal(restored.sf(x), model.sf(x))
    assert np.array_equal(restored.ff(x), model.ff(x))
    assert restored.mean() == model.mean()
    # the class-level reader and JSON file paths agree too
    assert np.array_equal(Parametric.from_dict(_rt(d)).sf(x), model.sf(x))
    # a different arity round-trips to a different arity
    six = Hypoexponential.from_params([1, 2, 3, 4, 5, 6])
    assert surpyval.from_dict(_rt(six.to_dict())).k == 6


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
        ([2.0, 2.0], "distinct"),
        ([2.0, 2.0 * (1 + 0.5 * DISTINCT_RATES_TOL)], "distinct"),
    ],
)
def test_invalid_rates_are_refused(rates, match):
    with pytest.raises(ValueError, match=match):
        Hypoexponential.from_params(rates)
    with pytest.raises(ValueError, match=match):
        Hypoexponential.sf(1.0, *rates)


def test_repeated_rates_error_points_at_erlang():
    with pytest.raises(ValueError, match="Erlang"):
        Hypoexponential.from_params([1.0, 1.0, 1.0])
    # ... which is where equal rates belong: a Gamma with integer shape
    x = np.linspace(0.0, 10.0, 21)
    erlang = Gamma.sf(x, 3, 1.0)
    nearly = Hypoexponential.sf(x, 1.0, 1.0 + 1e-3, 1.0 - 1e-3)
    assert np.allclose(nearly, erlang, atol=1e-5)


def test_rates_just_above_tolerance_are_still_accurate():
    a = 1.0
    b = a * (1 + 10 * DISTINCT_RATES_TOL)
    x = np.linspace(0.0, 10.0, 21)
    # against the Erlang(2) limit, which the two-stage sf approaches
    assert np.allclose(
        Hypoexponential.sf(x, a, b), Gamma.sf(x, 2, a), atol=1e-5
    )
    assert np.all(np.diff(Hypoexponential.sf(x, a, b)) <= 0)


def test_fit_is_refused_with_guidance():
    with pytest.raises(NotImplementedError, match="from_params"):
        Hypoexponential.fit([1.0, 2.0, 3.0])


def test_from_params_rejects_offset_on_top_of_invalid_rates_first():
    with pytest.raises(ValueError, match="strictly positive"):
        Hypoexponential.from_params([-1.0], gamma=1.0)
