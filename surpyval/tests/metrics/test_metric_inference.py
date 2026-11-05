"""Properties of the standard errors, intervals and comparisons of the
prediction metrics (``score_predictions``); the R references are in
surpyval/tests/reference/test_metric_inference.py.

* The censoring part of the influence functions is a derivative: giving
  one subject a little more weight in the censoring estimate moves each
  IPCW weight by that subject's influence over ``n``, checked by central
  differences, with and without ties. (The whole influence functions are
  checked against R.)
* Without censoring the Brier score's influence function is its
  residuals, centred.
* A model compared with itself differs by exactly 0; swapping the order
  flips the sign.
* The 95% intervals cover the population Brier score and AUC at about the
  nominal rate (a simulation, seed and size fixed before the first run).
"""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.metrics import (
    PredictionScores,
    brier_score,
    score_predictions,
)
from surpyval.utils.ipcw import (
    ipcw_censoring_influence,
    step_at,
    step_left_limit,
)


def _data(n=40, seed=22901):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 1, n)
    t = 10 * np.exp(-0.5 * z) * rng.weibull(1.3, n)
    u = rng.uniform(0, 20, n)
    x = np.minimum(t, u)
    c = (u < t).astype(int)
    times = np.array([4.0, 8.0])
    S = np.exp(-((np.outer(np.exp(0.4 * z), times) / 10) ** 1.3))
    return x, c, z, S, times


def _log_inverse_g(x, censored, w, at, strict):
    """``log(1 / G(u))`` at each ``at`` (its left limit where ``strict``),
    ``G = exp(-Lambda_C)``, ``Lambda_C`` the Nelson-Aalen estimate of the
    censoring hazard with count weights ``w`` (everyone with ``x >= s`` at
    risk at ``s``)."""
    uniq = np.unique(x)
    at_risk = np.array([w[x >= s].sum() for s in uniq])
    d = np.array([w[(x == s) & censored].sum() for s in uniq])
    lam = np.cumsum(d / at_risk)
    return np.where(
        strict,
        step_left_limit(uniq, lam, at, before=0.0),
        step_at(uniq, lam, at, before=0.0),
    )


@pytest.mark.parametrize("tied", [False, True])
def test_censoring_influence_is_the_derivative(tied):
    """``ipcw_censoring_influence`` is ``n`` times the derivative of
    ``sum_j coef_j log(1 / G(u_j))`` with respect to one subject's weight,
    exactly (the Nelson-Aalen form of ``G``, whose first-order expansion
    the Kaplan-Meier shares; riskRegression uses the same), with and
    without ties, at event times (``strict``) and at a horizon."""
    rng = np.random.default_rng(22903)
    n = 30
    x = rng.exponential(5, n)
    if tied:
        x = np.ceil(x)
    censored = rng.uniform(size=n) < 0.4
    strict = ~censored & (x <= 6)
    at = np.where(strict, x, 6.0)
    coef = rng.normal(size=n)
    got = ipcw_censoring_influence(x, censored, coef, at, strict)
    h = 1e-6
    numeric = np.empty(n)
    for k in range(n):
        up = np.ones(n)
        down = np.ones(n)
        up[k] += h
        down[k] -= h
        numeric[k] = (
            n
            * coef
            @ (
                _log_inverse_g(x, censored, up, at, strict)
                - _log_inverse_g(x, censored, down, at, strict)
            )
            / (2 * h)
        )
    assert_allclose(got, numeric, rtol=1e-6, atol=1e-8)


def test_without_censoring_the_brier_influence_is_the_residuals():
    x, _, _, S, times = _data()
    c = np.zeros(x.size, dtype=int)
    result = score_predictions(x, c, S, times, metrics=("brier",))
    for j, t in enumerate(times):
        r = np.where(x > t, (1 - S[:, j]) ** 2, S[:, j] ** 2)
        assert_allclose(
            result.influence["brier", "model"][:, j], r - r.mean(), atol=1e-15
        )
        assert_allclose(
            result.se("brier")[j],
            np.std(r, ddof=1) / np.sqrt(x.size),
            rtol=1e-12,
        )


def test_a_model_compared_with_itself_and_swapped():
    x, c, z, S, times = _data()
    S2 = np.exp(-((np.outer(np.exp(0.6 * z), times) / 10) ** 1.3))
    same = score_predictions(x, c, {"a": S, "b": S}, times)
    assert np.all(same.comparisons.difference == 0)
    assert np.all(same.comparisons.se == 0)
    assert same.comparisons.p_value.isna().all()
    ab = score_predictions(x, c, {"a": S, "b": S2}, times).comparisons
    ba = score_predictions(x, c, {"b": S2, "a": S}, times).comparisons
    assert_allclose(ab.difference, -ba.difference, atol=1e-15)
    assert_allclose(ab.se, ba.se, rtol=1e-12)
    assert_allclose(ab.p_value, ba.p_value, rtol=1e-10)
    assert len(score_predictions(x, c, S, times).comparisons) == 0


def test_three_models_compare_each_with_those_before():
    x, c, z, S, times = _data()
    preds = {
        name: np.exp(-((np.outer(np.exp(b * z), times) / 10) ** 1.3))
        for name, b in (("a", 0.2), ("b", 0.4), ("c", 0.6))
    }
    table = score_predictions(x, c, preds, times, metrics=("ibs",))
    pairs = list(zip(table.comparisons.model, table.comparisons.reference))
    assert pairs == [("b", "a"), ("c", "a"), ("c", "b")]


def test_horizon_without_cases_is_nan_throughout():
    x, c, _, S, _ = _data()
    t = np.array([x.min() / 2, 8.0])
    S = S.copy()
    result = score_predictions(x, c, {"a": S, "b": S * 0.9}, t)
    auc = result.scores.query("metric == 'auc'")
    assert (
        auc.iloc[[0, 2]][["estimate", "se", "lower", "upper"]]
        .isna()
        .all(axis=None)
    )
    assert not np.isnan(auc.iloc[1].se)


def test_to_dict_is_strict_json_and_round_trips():
    x, c, z, S, times = _data()
    t = np.array([x.min() / 2, 8.0])
    S0 = np.exp(-((np.outer(np.exp(0.4 * z), t) / 10) ** 1.3))
    result = score_predictions(x, c, {"a": S0, "b": S0 * 0.9}, t)
    d = result.to_dict()
    json.dumps(d, allow_nan=False)
    again = PredictionScores.from_dict(json.loads(json.dumps(d)))
    assert again.models == result.models
    assert_allclose(again.scores.estimate, result.scores.estimate)
    assert_allclose(again.comparisons.p_value, result.comparisons.p_value)
    assert "Comparisons" in repr(again)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(metrics=("brier", "c-index")), "metrics"),
        (dict(metrics=()), "metrics"),
        (dict(alpha_ci=1.5), "alpha_ci"),
        (dict(alpha_ci="0.05"), "alpha_ci"),
    ],
)
def test_refusals_name_the_argument(kwargs, match):
    x, c, _, S, times = _data()
    with pytest.raises(ValueError, match=match):
        score_predictions(x, c, S, times, **kwargs)


def test_refuses_a_bad_matrix_naming_the_model():
    x, c, _, S, times = _data()
    with pytest.raises(ValueError, match=r"predictions\['b'\]"):
        score_predictions(x, c, {"a": S, "b": S[:, :1]}, times)
    with pytest.raises(ValueError, match="probabilities"):
        score_predictions(x, c, {"a": S, "b": S * 3}, times)
    with pytest.raises(ValueError, match="empty"):
        score_predictions(x, c, {}, times)


def test_brier_score_unchanged():
    # The plain function returns what it did: (times, scores).
    x, c, _, S, times = _data()
    out = brier_score(x, c, S, times)
    assert isinstance(out, tuple) and len(out) == 2


def test_intervals_cover_at_the_nominal_rate():
    """200 data sets of 150 subjects (seed 22902, fixed before the first
    run): Weibull times, uniform censoring (about 38%), the attenuated
    model of the reference fixtures scored at t = 6. The truths are the
    population Brier score and AUC of that model, from one uncensored
    sample of 200,000. The 95% intervals must cover in [0.904, 0.996],
    0.95 +/- 3 binomial standard errors."""
    t = 6.0

    def draw(rng, n):
        z = rng.normal(0, 1, n)
        T = 10 * np.exp(-0.5 * z) * rng.weibull(1.3, n)
        S = np.exp(-((np.exp(0.4 * z) * t / 10) ** 1.3))
        return z, T, S

    rng = np.random.default_rng(22902)
    _, T, S = draw(rng, 200_000)
    true_brier = np.mean((np.where(T > t, 1.0, 0.0) - S) ** 2)
    risk = 1 - S
    cases, controls = risk[T <= t], np.sort(risk[T > t])
    below = np.searchsorted(controls, cases, side="left")
    true_auc = below.sum() / (cases.size * controls.size)

    covered = {"brier": 0, "auc": 0}
    reps = 200
    for _ in range(reps):
        _, T, S = draw(rng, 150)
        u = rng.uniform(0, 25, 150)
        x, c = np.minimum(T, u), (u < T).astype(int)
        res = score_predictions(x, c, S, [t], metrics=("brier", "auc"))
        for metric, truth in (("brier", true_brier), ("auc", true_auc)):
            row = res.scores.query("metric == @metric").iloc[0]
            covered[metric] += row.lower <= truth <= row.upper
    for metric, k in covered.items():
        assert 0.904 <= k / reps <= 0.996, (metric, k / reps)
