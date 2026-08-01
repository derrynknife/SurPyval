"""Closed-form maximum likelihood estimation.

Where an exact analytic MLE exists it is used instead of the optimiser
ladder. These tests pin three things: that it is taken exactly when it
applies, that it agrees with the optimiser it replaces (and never fits
worse, being exact), and that the resulting model is complete -- a fit
without ``neg_ll`` or a covariance would silently lose ``aic``, ``bic``
and ``cb``.
"""

import numpy as np
import pytest

from surpyval import Exponential, LogNormal, Normal, Uniform, Weibull

SEED = 20260801


def _optimizer(model):
    return getattr(model, "optimizer", None)


def _is_closed_form(model):
    return _optimizer(model) == "closed-form"


# -- the closed form is taken when it applies --------------------------


def test_exponential_uses_closed_form_and_matches_the_formula():
    rng = np.random.default_rng(SEED)
    x = rng.exponential(1 / 0.7, 300)
    model = Exponential.fit(x)
    assert _is_closed_form(model)
    np.testing.assert_allclose(
        model.params[0], len(x) / x.sum(), rtol=1e-12, atol=0
    )


def test_exponential_closed_form_with_right_censoring_and_weights():
    rng = np.random.default_rng(SEED)
    x = rng.exponential(1 / 0.7, 300)
    c = (rng.random(300) < 0.3).astype(int)
    n = rng.integers(1, 4, 300)
    model = Exponential.fit(x, c=c, n=n)
    assert _is_closed_form(model)
    expected = (n * (c == 0)).sum() / (n * x).sum()
    np.testing.assert_allclose(model.params[0], expected, rtol=1e-12, atol=0)


def test_exponential_closed_form_with_left_truncation():
    # Left truncation only moves each unit's exposure from x to x - tl,
    # so the estimator stays events-over-exposure.
    rng = np.random.default_rng(SEED)
    x = rng.exponential(1 / 0.7, 400)
    tl = rng.uniform(0, 0.4, 400)
    keep = x > tl
    x, tl = x[keep], tl[keep]
    model = Exponential.fit(x, tl=tl)
    assert _is_closed_form(model)
    np.testing.assert_allclose(
        model.params[0], len(x) / (x - tl).sum(), rtol=1e-12, atol=0
    )


@pytest.mark.parametrize(
    "dist,transform", [(Normal, lambda v: v), (LogNormal, np.log)]
)
def test_normal_family_closed_form_matches_moments(dist, transform):
    rng = np.random.default_rng(SEED)
    x = (
        rng.normal(10, 2, 300)
        if dist is Normal
        else rng.lognormal(2, 0.5, 300)
    )
    model = dist.fit(x)
    assert _is_closed_form(model)
    v = transform(x)
    # The MLE divides by the total weight, not by total - 1.
    np.testing.assert_allclose(model.params[0], v.mean(), rtol=1e-12, atol=0)
    np.testing.assert_allclose(model.params[1], v.std(), rtol=1e-12, atol=0)


def test_normal_closed_form_honours_weights():
    rng = np.random.default_rng(SEED)
    x = rng.normal(10, 2, 200)
    n = rng.integers(1, 5, 200)
    model = Normal.fit(x, n=n)
    assert _is_closed_form(model)
    mu = (n * x).sum() / n.sum()
    sd = np.sqrt((n * (x - mu) ** 2).sum() / n.sum())
    np.testing.assert_allclose(model.params, [mu, sd], rtol=1e-12, atol=0)


# -- and skipped when it does not ---------------------------------------


def _exp_data(rng, n=200):
    return rng.exponential(1 / 0.7, n)


@pytest.mark.parametrize(
    "kind",
    ["left_censored", "interval_censored", "right_truncated"],
)
def test_exponential_falls_back_on_unsupported_shapes(kind):
    # Each of these introduces a log(1 - exp(-lambda t)) term, making the
    # score transcendental.
    rng = np.random.default_rng(SEED)
    x = _exp_data(rng)
    if kind == "left_censored":
        c = np.where(rng.random(x.size) < 0.3, -1, 0)
        model = Exponential.fit(x, c=c)
    elif kind == "interval_censored":
        xr = x + rng.uniform(0.1, 1.0, x.size)
        model = Exponential.fit(
            x=np.column_stack([x, xr]), c=np.full(x.size, 2)
        )
    else:
        model = Exponential.fit(x, tr=x + rng.uniform(0.5, 3.0, x.size))
    assert not _is_closed_form(model)


@pytest.mark.parametrize("dist", [Normal, LogNormal])
@pytest.mark.parametrize("kind", ["right_censored", "left_truncated"])
def test_normal_family_falls_back_on_censoring_or_truncation(dist, kind):
    # Censoring makes this the Tobit model; truncation introduces a Phi
    # normaliser. Neither has an analytic solution.
    rng = np.random.default_rng(SEED)
    x = (
        rng.normal(10, 2, 200)
        if dist is Normal
        else rng.lognormal(2, 0.5, 200)
    )
    if kind == "right_censored":
        model = dist.fit(x, c=(rng.random(200) < 0.3).astype(int))
    else:
        tl = np.full(200, np.quantile(x, 0.1))
        keep = x > tl
        model = dist.fit(x[keep], tl=tl[keep])
    assert not _is_closed_form(model)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lfp": True},
        {"zi": True},
        {"offset": True},
        {"fixed": {"failure_rate": 0.5}},
    ],
    ids=["lfp", "zi", "offset", "fixed"],
)
def test_structural_requests_skip_the_closed_form(kwargs):
    # gamma / p / f0 and held parameters add structure the closed form
    # does not solve for. Before the gate existed these were silently
    # dropped and the closed-form answer returned regardless.
    rng = np.random.default_rng(SEED)
    x = rng.exponential(1 / 0.7, 200)
    c = (rng.random(200) < 0.3).astype(int)
    try:
        model = Exponential.fit(x, c=c, **kwargs)
    except ValueError:
        # Some combinations are rejected outright by validation; that is
        # also "not silently ignored".
        return
    assert not _is_closed_form(model)


def test_limited_failure_population_is_actually_estimated():
    # The sharp version of the above: with a genuinely cured fraction the
    # fitted p must move away from 1, which the closed form could never
    # report.
    rng = np.random.default_rng(SEED)
    n = 400
    fails = rng.exponential(2.0, n)
    cured = rng.random(n) < 0.35
    x = np.where(cured, 12.0, np.minimum(fails, 12.0))
    c = (cured | (fails > 12.0)).astype(int)
    model = Exponential.fit(x, c=c, lfp=True)
    assert not _is_closed_form(model)
    assert 0.5 < model.p < 0.8


def test_fixed_parameter_is_honoured():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(2.0, 8.0, 200)
    model = Uniform.fit(x, fixed={"a": 1.5})
    assert np.isclose(model.params[0], 1.5)


def test_weibull_is_untouched():
    rng = np.random.default_rng(SEED)
    model = Weibull.fit(rng.weibull(2.0, 200) * 10)
    assert not _is_closed_form(model)


# -- the closed-form result is complete ---------------------------------


@pytest.mark.parametrize(
    "dist,gen",
    [
        (Exponential, lambda r: r.exponential(1 / 0.7, 300)),
        (Normal, lambda r: r.normal(10, 2, 300)),
        (LogNormal, lambda r: r.lognormal(2, 0.5, 300)),
    ],
    ids=["Exponential", "Normal", "LogNormal"],
)
def test_closed_form_model_supports_the_usual_methods(dist, gen):
    rng = np.random.default_rng(SEED)
    model = dist.fit(gen(rng))
    assert _is_closed_form(model)
    for method in ("neg_ll", "aic", "bic", "aic_c"):
        assert np.isfinite(getattr(model, method)())
    grid = np.quantile(model.data["x"], [0.25, 0.5, 0.75])
    bounds = np.asarray(model.cb(grid, alpha_ci=0.05), dtype=float)
    assert np.isfinite(bounds).all()
    assert model.cov_matrix is not None


@pytest.mark.parametrize(
    "dist,gen",
    [
        (Exponential, lambda r: r.exponential(1 / 0.7, 300)),
        (Normal, lambda r: r.normal(10, 2, 300)),
        (LogNormal, lambda r: r.lognormal(2, 0.5, 300)),
    ],
    ids=["Exponential", "Normal", "LogNormal"],
)
def test_closed_form_fits_no_worse_than_the_optimiser(dist, gen, monkeypatch):
    # The strongest available check: the closed form is the exact
    # maximiser, so its log-likelihood cannot be beaten by the optimiser
    # ladder it replaced. Disabling the closed form sends the identical
    # data down the identical path the optimiser used before.
    rng = np.random.default_rng(SEED)
    x = gen(rng)

    closed = dist.fit(x)
    assert _is_closed_form(closed)

    monkeypatch.setattr(
        type(dist), "_closed_form_mle", lambda self, data: None
    )
    optimised = dist.fit(x)
    assert not _is_closed_form(optimised)

    assert closed.neg_ll() <= optimised.neg_ll() + 1e-9
    # ... and lands in the same place, to the optimiser's own tolerance.
    np.testing.assert_allclose(
        closed.params, optimised.params, rtol=1e-5, atol=0
    )


# -- Uniform: the pre-existing closed form -------------------------------


def test_uniform_likelihood_is_finite():
    # The generic log_df identity log(hf) - Hf is nan at the upper support
    # edge, where sf is 0 -- and the MLE puts `b` exactly there, so the
    # whole log-likelihood was nan and neg_ll/aic/bic raised.
    rng = np.random.default_rng(SEED)
    x = rng.uniform(2.0, 8.0, 300)
    model = Uniform.fit(x)
    assert _is_closed_form(model)
    expected = len(x) * np.log(x.max() - x.min())
    np.testing.assert_allclose(model.neg_ll(), expected, rtol=1e-12, atol=0)
    for method in ("aic", "bic", "aic_c"):
        assert np.isfinite(getattr(model, method)())


def test_uniform_offers_no_covariance():
    # The Uniform MLE is an order statistic sitting on the support edge,
    # not an interior stationary point, so the observed information is not
    # positive definite. Reporting its inverse would give negative
    # variances and silent nan bounds.
    rng = np.random.default_rng(SEED)
    model = Uniform.fit(rng.uniform(2.0, 8.0, 300))
    assert model.cov_matrix is None
    with pytest.raises(ValueError, match="covariance"):
        model.cb([3.0, 5.0])
