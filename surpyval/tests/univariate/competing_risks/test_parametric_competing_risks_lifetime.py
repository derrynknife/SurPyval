"""All-cause lifetime of a parametric competing-risks model: ``df``,
``mean`` and ``qf``, and the net ``sf`` / ``ff`` / ``df`` / ``qf`` of one
cause (``event=k``).

The references are closed forms, not R: the all-cause survival is the
product of the causes' survivals, so

- exponential causes give an exponential with rate :math:`\\sum_k
  \\lambda_k` (mean :math:`1 / \\sum_k \\lambda_k`);
- Weibull causes with a common shape :math:`\\beta` give a Weibull with
  that shape and scale :math:`(\\sum_k \\alpha_k^{-\\beta})^{-1/\\beta}`,
  and cause ``k``'s share of every failure probability is
  :math:`\\alpha_k^{-\\beta} / \\sum_j \\alpha_j^{-\\beta}`;
- two LogLogistic causes with scale :math:`\\alpha` and shape
  :math:`\\beta` give :math:`S = (1 + (t/\\alpha)^\\beta)^{-2}`, whose
  integral is :math:`\\alpha\\,\\Gamma(1 + 1/\\beta)\\Gamma(2 -
  1/\\beta)`.
"""

import warnings
from math import gamma

import numpy as np
import pytest
from scipy.integrate import quad

from surpyval import Exponential, LogLogistic, LogNormal, Normal, Weibull
from surpyval.univariate.competing_risks import ParametricCompetingRisks

P = np.array([1e-12, 1e-6, 0.01, 0.1, 0.5, 0.9, 0.99, 1 - 1e-9])


def _exponential(rates):
    return ParametricCompetingRisks.from_fitted(
        {i: Exponential.from_params([r]) for i, r in enumerate(rates)}
    )


def _weibull(alphas, beta):
    return ParametricCompetingRisks.from_fitted(
        {i: Weibull.from_params([a, beta]) for i, a in enumerate(alphas)}
    )


# -- exponential causes: closed forms --------------------------------------


def test_exponential_df_mean_qf_closed_forms():
    rates = [0.1, 0.3, 0.05]
    total = sum(rates)
    model = _exponential(rates)
    x = np.array([0.0, 0.5, 2.0, 10.0, 50.0])
    np.testing.assert_allclose(
        model.df(x), total * np.exp(-total * x), rtol=1e-12
    )
    assert model.mean() == pytest.approx(1 / total, rel=1e-10)
    np.testing.assert_allclose(model.qf(P), -np.log1p(-P) / total, rtol=1e-12)


# -- Weibull causes with a common shape ------------------------------------


def test_weibull_common_shape_is_a_weibull():
    alphas, beta = [100.0, 60.0, 250.0], 2.5
    model = _weibull(alphas, beta)
    w = np.array(alphas) ** -beta
    alpha = w.sum() ** (-1 / beta)
    ref = Weibull.from_params([alpha, beta])
    x = np.array([1.0, 10.0, 40.0, 80.0, 150.0])
    np.testing.assert_allclose(model.df(x), ref.df(x), rtol=1e-12)
    assert model.mean() == pytest.approx(alpha * gamma(1 + 1 / beta), 1e-10)
    np.testing.assert_allclose(
        model.qf(P), alpha * (-np.log1p(-P)) ** (1 / beta), rtol=1e-12
    )
    # Each cause's share of the failures is its weight, at every time.
    for k in model.causes:
        np.testing.assert_allclose(
            model.cif(x, k), w[k] / w.sum() * model.ff(x), rtol=1e-9
        )


# -- one cause is that cause's Parametric model ----------------------------


@pytest.mark.parametrize(
    "cause",
    [
        Weibull.from_params([10.0, 3.0]),
        LogNormal.from_params([2.0, 0.8]),
        Weibull.from_params([50.0, 0.7]),
    ],
    ids=["Weibull", "LogNormal", "Weibull-infant"],
)
def test_single_cause_is_the_parametric_model(cause):
    model = ParametricCompetingRisks.from_fitted({"only": cause})
    x = np.array([0.5, 3.0, 10.0, 25.0])
    np.testing.assert_allclose(model.df(x), cause.df(x), rtol=1e-12)
    assert model.mean() == pytest.approx(cause.mean(), rel=1e-10)
    p = P[1:]
    np.testing.assert_allclose(model.qf(p), cause.qf(p), rtol=1e-12)


# -- cure (limited-failure) fractions --------------------------------------


def test_cure_ceiling():
    a = Weibull.from_params([100.0, 2.0], lfp_p=0.9)
    b = Weibull.from_params([50.0, 1.5], lfp_p=0.5)
    model = ParametricCompetingRisks.from_fitted({"a": a, "b": b})
    ceiling = 1 - 0.1 * 0.5
    assert model.mean() == np.inf
    # The defective mean: the integral of t dF = of S(t) - S(inf).
    ref, _ = quad(
        lambda t: float(model.sf(t)) - (1 - ceiling), 0, np.inf, limit=500
    )
    assert model.mean(defective=True) == pytest.approx(ref, rel=1e-8)
    q = model.qf([0.5, 0.94, 0.96, 0.99, 1.0])
    assert np.all(np.isfinite(q[:2])) and np.all(np.isinf(q[2:]))
    np.testing.assert_allclose(model.ff(q[:2]), [0.5, 0.94], rtol=1e-12)


def test_cure_ceiling_single_cause_matches_parametric():
    a = Weibull.from_params([100.0, 2.0], lfp_p=0.9)
    model = ParametricCompetingRisks.from_fitted({"a": a})
    assert model.mean() == a.mean() == np.inf
    assert model.mean(defective=True) == pytest.approx(
        a.mean(defective=True), rel=1e-10
    )
    np.testing.assert_array_equal(np.isinf(model.qf([0.5, 0.95])), [0, 1])


# -- tails -----------------------------------------------------------------


def test_heavy_tails():
    # Two LogLogistic(10, 0.8): each alone has an infinite mean; the first
    # of the two does not (S falls as t^-1.6).
    one = LogLogistic.from_params([10.0, 0.8])
    assert ParametricCompetingRisks.from_fitted({0: one}).mean() == np.inf
    two = ParametricCompetingRisks.from_fitted({0: one, 1: one})
    # alpha / beta B(1/beta, 2 - 1/beta) = alpha Gamma(1 + 1/beta)
    # Gamma(2 - 1/beta), Gamma(2) being 1.
    ref = 10.0 * gamma(1 + 1 / 0.8) * gamma(2 - 1 / 0.8)
    assert two.mean() == pytest.approx(ref, rel=1e-10)


def test_upper_tail_quantile_is_precise():
    model = ParametricCompetingRisks.from_fitted(
        {
            "wear": Weibull.from_params([100.0, 2.0]),
            "drift": LogNormal.from_params([4.0, 1.0]),
        }
    )
    p = 1 - 1e-12
    q = model.qf(p)
    assert np.isfinite(q)
    # Round trip on the cumulative-hazard scale, where 1 - p keeps its
    # digits (on the ff scale every q within 1e-4 would round to p).
    assert model.Hf(q) == pytest.approx(-np.log1p(-p), rel=1e-12)
    x = np.array([0.1, 1.0, 10.0, 100.0, 300.0])
    np.testing.assert_allclose(model.qf(model.ff(x)), x, rtol=1e-10)


def test_support_reaching_below_zero():
    # A Normal cause: the mean counts the negative part of the support.
    model = ParametricCompetingRisks.from_fitted(
        {0: Normal.from_params([5.0, 2.0])}
    )
    assert model.mean() == pytest.approx(5.0, rel=1e-10)
    assert model.qf(0.0) == -np.inf and model.qf(1.0) == np.inf


# -- density ---------------------------------------------------------------


def test_density_is_the_sum_of_the_incidence_densities():
    model = ParametricCompetingRisks.from_fitted(
        {
            "a": Weibull.from_params([30.0, 2.0]),
            "b": LogNormal.from_params([3.5, 0.6]),
        }
    )
    x = np.array([1.0, 10.0, 30.0, 60.0])
    total = sum(model.iif(x, k) for k in model.causes)
    np.testing.assert_allclose(model.df(x), total, rtol=1e-12)
    area, _ = quad(lambda t: float(model.df(t)), 0, np.inf, limit=200)
    assert area == pytest.approx(1.0, abs=1e-8)


# -- net functions of one cause (event=k) ----------------------------------


def test_event_gives_the_net_functions():
    a, b = Weibull.from_params([30.0, 2.0]), Exponential.from_params([0.02])
    model = ParametricCompetingRisks.from_fitted({"a": a, "b": b})
    x = np.array([1.0, 10.0, 40.0])
    for name in ("sf", "ff", "df"):
        np.testing.assert_array_equal(
            getattr(model, name)(x, event="a"), getattr(a, name)(x)
        )
    np.testing.assert_array_equal(model.qf([0.1, 0.5], "b"), b.qf([0.1, 0.5]))
    # The defaults are unchanged: all-cause.
    np.testing.assert_allclose(model.sf(x), a.sf(x) * b.sf(x), rtol=1e-12)
    with pytest.raises(ValueError):
        model.sf(x, event="c")


def test_shapes():
    model = _exponential([0.1, 0.3])
    assert np.ndim(model.df(2.0)) == 0 and np.ndim(model.qf(0.3)) == 0
    assert model.df([[1.0, 2.0], [3.0, 4.0]]).shape == (2, 2)
    assert model.qf([[0.1], [0.2]]).shape == (2, 1)
    assert model.qf([]).shape == (0,)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = model.qf([np.nan, 0.5, 10.0])
    assert np.isnan(got[[0, 2]]).all() and np.isfinite(got[1])
    assert len(caught) == 1 and "outside [0, 1]" in str(caught[0].message)


# -- simulation ------------------------------------------------------------


def test_simulation_agrees_with_mean_and_quantiles():
    # Fixed seed chosen before the result was seen; 4 standard errors.
    model = ParametricCompetingRisks.from_fitted(
        {
            "wear": Weibull.from_params([100.0, 3.0]),
            "shock": Exponential.from_params([1 / 150.0]),
        }
    )
    n = 20_000
    x = model.random(n, random_state=20261105)["x"]
    se = x.std(ddof=1) / np.sqrt(n)
    assert abs(x.mean() - model.mean()) < 4 * se
    p = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    ecdf = (x[:, None] <= model.qf(p)[None, :]).mean(axis=0)
    assert np.all(np.abs(ecdf - p) < 4 * np.sqrt(p * (1 - p) / n))
