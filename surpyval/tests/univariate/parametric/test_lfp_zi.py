import numpy as np
import pytest

import surpyval as surv
from surpyval import (
    Beta,
    Gamma,
    Gumbel,
    GumbelLEV,
    Logistic,
    LogLogistic,
    LogNormal,
    Normal,
    Rayleigh,
    Weibull,
)
from surpyval.tests._helpers import no_warnings
from surpyval.univariate.parametric.parametric_fitter import (
    OutsideSupportError,
)


def test_zi():
    np.random.seed(42)
    for dist in [Beta, Weibull, LogNormal, LogLogistic, Gamma]:
        for zeros in [1, 10, 100, 1000, 10000]:
            x = dist.random(100, 10, 2)
            x = np.concatenate((x, np.zeros(zeros)))
            model = dist.fit(x, zi=True)
            assert model.res.success
            # The zero-inflation estimate must match the actual
            # proportion of zeros
            f0_true = zeros / (100 + zeros)
            assert abs(model.f0 - f0_true) < 0.05


def test_lfp():
    np.random.seed(42)
    for dist in [
        Beta,
        Gamma,
        Gumbel,
        GumbelLEV,
        Logistic,
        LogLogistic,
        LogNormal,
        Normal,
        Weibull,
    ]:
        for censored in [1, 10, 100, 1000, 10000]:
            x = dist.random(100, 10, 2)
            c = np.concatenate((np.zeros_like(x), np.ones(censored)))
            x = np.concatenate((x, x.max() * np.ones(censored)))
            model = dist.fit(x, c=c, lfp=True)
            assert model.res.success
            # All failures are observed before the censor time, so the
            # max proportion estimate must match the failing fraction
            p_true = 100 / (100 + censored)
            assert abs(model.lfp_p - p_true) < 0.1


def test_lfp_zi():
    np.random.seed(42)
    for dist in [Gamma, Weibull, LogNormal, LogLogistic]:
        for zi_lfp_values in [1, 10, 100]:
            for num_samples in [100, 1000, 10000]:
                x = dist.random(num_samples, 10, 2)
                c = np.concatenate(
                    (
                        np.zeros_like(x),
                        np.zeros(zi_lfp_values),
                        np.ones(zi_lfp_values),
                    )
                )
                x = np.concatenate(
                    (
                        x,
                        np.zeros(zi_lfp_values),
                        x.max() * np.ones(zi_lfp_values) + 1,
                    )
                )
                model = dist.fit(x, c=c, zi=True, lfp=True)
                if not model.res.success:
                    raise ValueError(model, model.res)
                total = num_samples + 2 * zi_lfp_values
                f0_true = zi_lfp_values / total
                p_true = (num_samples + zi_lfp_values) / total
                assert abs(model.f0 - f0_true) < 0.05
                assert abs(model.lfp_p - p_true) < 0.1


def test_offset_lfp():
    np.random.seed(1)
    n = 2000
    x = Weibull.random(n, 10, 2) + 10
    c = np.zeros(n)
    never = np.random.uniform(size=n) > 0.7
    x[never] = x.max() + 1
    c[never] = 1

    model = Weibull.fit(x, c=c, lfp=True, offset=True)
    assert model.res.success
    assert abs(model.gamma - 10) < 1
    assert abs(model.lfp_p - 0.7) < 0.05


def test_offset_zi():
    # The offset bound and initial guess must come from the nonzero
    # observations; the zeros belong to the zero-inflation mass
    np.random.seed(2)
    x = np.concatenate([Weibull.random(1000, 10, 2) + 10, np.zeros(100)])

    model = Weibull.fit(x, zi=True, offset=True)
    assert model.res.success
    assert abs(model.gamma - 10) < 1
    assert abs(model.f0 - 100 / 1100) < 0.02


# --- quantile function for the mixture (LFP / zero-inflation / offset) -----


def test_qf_inverts_ff_for_lfp():
    # Below the cure ceiling p, the quantile inverts the failure function.
    model = Weibull.from_params([10.0, 2.0], lfp_p=0.6)
    u = np.array([0.05, 0.2, 0.4, 0.59])
    q = model.qf(u)
    assert np.all(np.isfinite(q))
    assert np.allclose(model.ff(q), u)


def test_qf_infinite_above_cure_fraction():
    # A cure fraction 1 - p never fails, so any quantile at or above p is
    # infinite -- and the median of a majority-cured population is infinite.
    model = Weibull.from_params([10.0, 2.0], lfp_p=0.6)
    assert np.isinf(model.qf(0.6))
    assert np.isinf(model.qf(0.85))
    cured = Weibull.from_params([10.0, 2.0], lfp_p=0.4)
    assert np.isinf(cured.qf(0.5))


def test_qf_inverts_ff_for_zero_inflation():
    # The zero-inflation mass f0 sits at 0 (with or without an offset), and
    # above it the quantile inverts the failure function.
    model = LogNormal.from_params([2.0, 0.4], f0=0.2)
    assert model.qf(0.1) == 0.0
    assert model.qf(0.2) == 0.0
    u = np.array([0.3, 0.5, 0.9])
    assert np.allclose(model.ff(model.qf(u)), u)


def test_qf_respects_offset_with_cure_and_inflation():
    # gamma + f0 + p all together: mass below f0 is the zero-inflation
    # point mass at 0 (consistent with ff(0) == f0, df's mass at x == 0 and
    # the likelihood, #256), the interior inverts ff, and u >= p is
    # infinite.
    model = Weibull.from_params([10.0, 2.0], gamma=5.0, lfp_p=0.7, f0=0.1)
    assert model.qf(0.05) == 0.0
    u = np.array([0.2, 0.4, 0.6])
    q = model.qf(u)
    assert np.all(q > 5.0)
    assert np.allclose(model.ff(q), u)
    assert np.isinf(model.qf(0.7))


def test_qf_scalar_and_array_shape():
    model = Weibull.from_params([10.0, 2.0], lfp_p=0.8)
    assert np.ndim(model.qf(0.3)) == 0
    out = model.qf([0.1, 0.3, 0.5])
    assert out.shape == (3,)


def test_qf_matches_plain_distribution_without_mixture():
    # With no offset, cure or inflation the quantile is exactly the base
    # distribution's, so ordinary models are unchanged.
    model = Weibull.from_params([10.0, 3.0])
    assert np.isclose(model.qf(0.2), Weibull.qf(0.2, 10.0, 3.0))


# --- moment for the mixture (LFP / zero-inflation / offset) ----------------


def test_moment_matches_plain_distribution_without_mixture():
    model = Weibull.from_params([10.0, 3.0])
    assert np.isclose(model.moment(2), Weibull.moment(2, 10.0, 3.0))
    assert np.isclose(model.moment(3), Weibull.moment(3, 10.0, 3.0))


def test_moment_one_equals_mean_across_mixtures():
    # moment(1) must be consistent with mean() for every configuration.
    for model in (
        Weibull.from_params([10.0, 2.0]),
        Weibull.from_params([10.0, 2.0], gamma=5.0),
        Weibull.from_params([10.0, 2.0], lfp_p=0.6),
        Weibull.from_params([10.0, 2.0], gamma=5.0, lfp_p=0.7),
    ):
        assert np.isclose(model.moment(1), model.mean())
        assert np.isclose(
            model.moment(1, defective=True), model.mean(defective=True)
        )


def test_offset_moment_includes_offset():
    # Regression: the offset previously dropped out of moment. E[(gamma+X)^2]
    # is strictly greater than E[X^2].
    base = Weibull.from_params([10.0, 2.0])
    offset = Weibull.from_params([10.0, 2.0], gamma=5.0)
    assert offset.moment(2) > base.moment(2)
    # exact binomial value: E[(g+X)^2] = g^2 + 2 g E[X] + E[X^2]
    g = 5.0
    expected = g**2 + 2 * g * base.moment(1) + base.moment(2)
    assert np.isclose(offset.moment(2), expected)


def test_lfp_moment_is_finite_and_defective():
    # With a cure fraction the moment of the lifetime is infinite (#404);
    # the defective moment is finite and equals the base moment scaled by
    # the failing proportion p (no offset).
    p = 0.6
    model = Weibull.from_params([10.0, 2.0], lfp_p=p)
    assert np.isinf(model.moment(2))
    assert np.isclose(
        model.moment(2, defective=True), p * Weibull.moment(2, 10.0, 2.0)
    )


def test_defective_moment_matches_monte_carlo():
    # cured units contribute nothing; offset shifts the failures.
    g, p, params = 6.0, 0.7, (10.0, 2.0)
    model = Weibull.from_params(list(params), gamma=g, lfp_p=p)
    rng = np.random.default_rng(0)
    n = 2_000_000
    fail = rng.uniform(size=n) < p
    t = np.where(fail, g + Weibull.random(n, *params), 0.0)
    assert np.isclose(model.moment(1, defective=True), t.mean(), rtol=0.02)
    assert np.isclose(
        model.moment(2, defective=True), (t**2).mean(), rtol=0.02
    )


# --- entropy for the mixture ----------------------------------------------


def test_entropy_is_offset_invariant():
    # Differential entropy is translation-invariant, so an offset model has
    # the same entropy as the un-offset one.
    base = Weibull.from_params([10.0, 2.0])
    offset = Weibull.from_params([10.0, 2.0], gamma=5.0)
    assert np.isclose(offset.entropy(), base.entropy())


def test_entropy_raises_with_a_probability_atom():
    # A cure fraction (mass at infinity) or zero-inflation (mass at 0)
    # leaves no single differential entropy.
    with pytest.raises(ValueError, match="probability atom"):
        Weibull.from_params([10.0, 2.0], lfp_p=0.6).entropy()
    with pytest.raises(ValueError, match="probability atom"):
        LogNormal.from_params([2.0, 0.4], f0=0.2).entropy()


# ---------------------------------------------------------------------------
# Rayleigh fits with LFP and ZI (#257).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("structural", ["lfp", "zi"])
def test_rayleigh_fits_with_lfp_and_zi(structural):
    np.random.seed(0)
    x = Rayleigh.random(200, 10.0)
    if structural == "zi":
        x = np.concatenate([x, np.zeros(10)])
    model = Rayleigh.fit(x, **{structural: True})
    # The sigma estimate is unaffected; the point is that it runs at all.
    assert model.params[0] == pytest.approx(9.92, abs=0.5)


# ---------------------------------------------------------------------------
# A distribution parameter named ``p`` (Geometric,
# NegativeBinomial) beside the LFP proportion.
# ---------------------------------------------------------------------------


def test_param_cb_on_a_distribution_parameter_named_p():
    np.random.seed(0)
    model = surv.Geometric.fit(surv.Geometric.random(200, 0.15))
    wald = model.param_cb("p")
    lr = model.param_cb("p", method="lr")
    p_hat = model.params[0]
    for bound in (wald, lr):
        assert bound[0] < p_hat < bound[1]
        assert 0 < bound[0] and bound[1] < 1


@pytest.mark.parametrize(
    "dist, x, c",
    [
        (surv.Geometric, [1, 2, 2, 3, 5, 8, 10], [0, 0, 0, 0, 0, 1, 1]),
        (
            surv.NegativeBinomial,
            [1, 2, 2, 3, 3, 4, 5, 6, 8, 10, 12, 12],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
        ),
    ],
)
def test_lfp_fit_for_a_distribution_parameter_named_p(dist, x, c):
    model = dist.fit(x, c=c, lfp=True)
    assert model.lfp_name == "lfp_p"
    assert len(model.params) == dist.k
    assert 0 < model.lfp_p < 1
    # the two p's are distinct parameters with distinct bounds
    assert model.param_cb("p")[0] < model.params[dist.param_map["p"]]
    lfp_bound = model.param_cb("lfp_p")
    assert lfp_bound[0] < model.lfp_p < lfp_bound[1]
    assert "Max Proportion (lfp_p)" in repr(model)


def test_lfp_proportion_can_be_fixed_by_its_own_name():
    model = surv.Geometric.fit(
        [1, 2, 2, 3, 5, 8, 10],
        c=[0, 0, 0, 0, 0, 1, 1],
        lfp=True,
        fixed={"lfp_p": 0.8},
    )
    assert model.lfp_p == pytest.approx(0.8)
    # every LFP proportion has the one name (#608)
    weibull = surv.Weibull.fit(
        [1, 2, 3, 4, 5, 6],
        c=[0, 0, 0, 0, 1, 1],
        lfp=True,
        fixed={"lfp_p": 0.8},
    )
    assert weibull.lfp_name == "lfp_p"
    assert weibull.lfp_p == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# LFP and ZI fit quietly without zeros.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


def test_lfp_and_zi_without_zeros_fit_quietly():
    no_warnings(
        W.fit,
        [1.0, 2, 3, 4, 5, 6, 7, 8],
        [0, 0, 0, 0, 0, 1, 1, 1],
        lfp=True,
        zi=True,
    )


# ---------------------------------------------------------------------------
# #269: LFP with left truncation uses the mixture survival in
# the truncation normaliser.
# ---------------------------------------------------------------------------


class TestLFPTruncation:
    def test_lfp_left_truncated_recovers_parameters(self):
        # 269: the old normaliser (p - f0) * (1 - F0(tl)) made the
        # likelihood unbounded; the fit returned alpha ~ 1e-42.
        np.random.seed(11)
        N = 30000
        is_mortal = np.random.uniform(size=N) < 0.6
        t = np.where(is_mortal, 10 * np.random.weibull(2, N), np.inf)
        entry = 3.0
        t_seen = t[t > entry]
        observed = t_seen < 20
        x = np.where(observed, t_seen, 20.0)
        c = (~observed).astype(int)

        model = Weibull.fit(x=x, c=c, tl=np.full(len(x), entry), lfp=True)
        alpha, beta = model.params
        assert alpha == pytest.approx(10.0, rel=0.05)
        assert beta == pytest.approx(2.0, rel=0.05)
        assert model.lfp_p == pytest.approx(0.6, abs=0.03)

    def test_plain_interval_likelihood_unchanged(self):
        # The f0 terms cancel for finite bounds: a plain interval-censored
        # fit must be unaffected by the 269 change.
        np.random.seed(12)
        t = 10 * np.random.weibull(2, 2000)
        xl = np.floor(t)
        xr = xl + 1.0
        model = Weibull.fit(x=np.column_stack([xl, xr]), c=np.full(2000, 2))
        assert model.params[0] == pytest.approx(10.0, rel=0.05)
        assert model.params[1] == pytest.approx(2.0, rel=0.1)


# ---------------------------------------------------------------------------
# #548: the zero-inflation mass sits at 0, so a left truncation below 0
# truncates nothing, and one at 0 excludes the mass.
# ---------------------------------------------------------------------------


def _zero_inflated_548():
    rng = np.random.default_rng(0)
    x = rng.weibull(1.5, 200) * 10
    zero = rng.random(200) < 0.13
    x[zero] = 0.0
    c = np.zeros(200, int)
    c[(rng.random(200) < 0.2) & ~zero] = 1
    return x, c, zero


@pytest.mark.parametrize("structural", [{}, {"lfp": True}, {"offset": True}])
@pytest.mark.parametrize("tl", [-1.0, -50.0])
def test_548_truncation_below_zero_is_no_truncation(structural, tl):
    x, c, zero = _zero_inflated_548()
    if structural.get("offset"):
        x[~zero] += 3.0
    plain = Weibull.fit(x, c=c, zi=True, **structural)
    # At tl = -1 the mass at 0 was counted as already gone: f0 ran from
    # 0.135 to 1 and the negative log-likelihood from 562.0 to -508.7
    truncated = no_warnings(Weibull.fit, x, c=c, zi=True, tl=tl, **structural)
    assert truncated.f0 == plain.f0
    np.testing.assert_array_equal(truncated.params, plain.params)
    assert truncated.gamma == plain.gamma and truncated.lfp_p == plain.lfp_p
    assert truncated._neg_ll == plain._neg_ll


def test_548_window_counts_the_mass_from_zero():
    # The likelihood's window (lo, hi]: the mass f0 at 0 is inside
    # (-1, 5] and outside (0, 5], as in the fitted model's ff
    model = Weibull.from_params([10, 1.5], f0=0.2)
    params = (10, 1.5, 0.0, 0.2, 1.0)
    for lo, hi in [(-1.0, 5.0), (0.0, 5.0), (-1.0, np.inf), (0.0, np.inf)]:
        log_window = Weibull.ll_interval_or_truncated(
            np.array([lo]), np.array([hi]), np.array([1]), *params
        )
        expected = (1.0 if np.isinf(hi) else model.ff(hi)) - model.ff(lo)
        assert np.exp(log_window) == pytest.approx(expected, rel=1e-12)
    assert model.ff(-1.0) == 0.0
    assert model.ff(0.0) == pytest.approx(0.2)


def _monthly_counts(months, counts, running, end=24.0):
    """Returns counted by month in service (interval-censored), and the
    units still running at ``end``."""
    months = np.asarray(months, dtype=float)
    return {
        "x": np.r_[np.column_stack([months - 1, months]), [[end, end]]],
        "c": np.r_[np.full(months.size, 2), 1],
        "n": np.r_[counts, running],
    }


def test_579_lfp_on_interval_counts_moves_p_off_its_bound():
    # #579: 20 000 units, 3% defective (Weibull 2 months, shape 0.7), the
    # rest wearing out (120 months, shape 3), returns counted by month for
    # 24 months. The default start ran p to 1, where p's searched value
    # no longer moves the likelihood (its gradient and curvature are
    # zero or rounding), and the fit reported a verified maximum at
    # neg_ll 5005.974; from a start near the answer the maximum is at
    # p = 0.059, neg_ll 5002.210.
    rng = np.random.default_rng(0)
    n_units = 20_000
    defective = rng.random(n_units) < 0.03
    t = np.where(
        defective,
        2 * rng.weibull(0.7, n_units),
        120 * rng.weibull(3, n_units),
    )
    month = np.ceil(t)
    returned = month <= 24
    months, counts = np.unique(month[returned], return_counts=True)
    data = _monthly_counts(months, counts, (~returned).sum())
    model = no_warnings(Weibull.fit, **data, lfp=True)
    near = Weibull.fit(**data, lfp=True, init=[2.0, 0.7, 0.05])
    assert model.maximum == "verified"
    assert model.neg_ll() <= near.neg_ll() + 1e-6
    assert model.neg_ll() == pytest.approx(5002.2097, abs=1e-3)
    assert model.lfp_p == pytest.approx(0.0592, abs=1e-3)


def test_579_p_on_its_bound_is_a_maximum_there():
    # 500 units counted by month, 11 returns: the likelihood is highest
    # at p = 1, falling as p moves off it. p's search ends where it is 1
    # in floating point, with a zero gradient and curvature, which the
    # check of the gradient and Hessian cannot pass: the fit warned that
    # it was not a verified maximum. p is now judged on its bound.
    data = _monthly_counts([1, 3, 8, 16, 23, 24], [5, 2, 1, 1, 1, 1], 489)
    model = no_warnings(Weibull.fit, **data, lfp=True)
    assert model.lfp_p == 1.0
    assert model.maximum == "verified"
    for p in (0.999, 0.9, 0.5):
        profile = Weibull.fit(**data, lfp=True, fixed={"lfp_p": p})
        assert profile.neg_ll() > model.neg_ll()


# ---------------------------------------------------------------------------
# #610: a zero-inflated fit checks the support as the plain fit does; the
# point mass makes 0 a possible failure time, but nothing lies below 0.
# ---------------------------------------------------------------------------

_ZERO_INFLATABLE = [Weibull, surv.Exponential, Gamma, LogNormal, LogLogistic]


@pytest.mark.parametrize("dist", _ZERO_INFLATABLE)
@pytest.mark.parametrize("lfp", [False, True])
@pytest.mark.parametrize(
    "data",
    [
        {"x": [-1.0, 0, 0, 1, 2, 3, 4, 5]},
        # left censored below 0
        {"x": [-1.0, 0, 0, 1, 2, 3, 4, 5], "c": [-1, 0, 0, 0, 0, 0, 1, 0]},
        # an interval reaching below 0
        {
            "xl": [-1.0, 0, 1, 2, 3],
            "xr": [1.0, 0, 1, 2, 3],
            "c": [2, 0, 0, 0, 0],
        },
    ],
)
def test_610_zero_inflated_fit_refuses_a_negative_time(dist, lfp, data):
    # The fit used to start the optimiser on a NaN likelihood and return
    # its start with "MLE Failed; returning the optimiser's start".
    with pytest.raises(OutsideSupportError, match="zero-inflated"):
        dist.fit(**data, zi=True, lfp=lfp)


@pytest.mark.parametrize("dist", _ZERO_INFLATABLE)
def test_610_zero_inflated_fit_takes_a_zero(dist):
    # A failure at 0, and a left censoring time of 0, are the point mass
    model = no_warnings(
        dist.fit,
        [0.0, 0, 1, 2, 3, 4, 5, 7],
        c=[0, -1, 0, 0, 1, 0, 0, 1],
        zi=True,
    )
    assert model.f0 == pytest.approx(0.25, rel=1e-4)


# ---------------------------------------------------------------------------
# #608: the limited-failure proportion is ``lfp_p``; ``p`` is a
# distribution's own parameter where it has one.
# ---------------------------------------------------------------------------
LFP_X = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LFP_C = [0] * 5 + [1] * 5


def _lfp_weibull(**kwargs):
    return no_warnings(Weibull.fit, LFP_X, LFP_C, lfp=True, **kwargs)


def test_608_the_proportion_is_lfp_p_and_p_is_its_deprecated_alias():
    model = _lfp_weibull()
    assert 0 < model.lfp_p < 1
    assert model.extras == {"lfp_p": model.lfp_p}
    with pytest.warns(DeprecationWarning, match="'Parametric.lfp_p'") as w:
        old = model.p
    assert old == model.lfp_p
    assert w[0].filename == __file__
    # Without lfp=True it is 1, as before
    assert no_warnings(Weibull.fit, LFP_X).lfp_p == 1


@pytest.mark.parametrize(
    "fit, index",
    [
        (lambda: surv.Bernoulli.fit([0, 1, 1, 0, 1]), 0),
        (lambda: surv.Binomial.fit([2, 3, 1, 4], n_trials=5), 1),
        (lambda: surv.FixedEventProbability.fit([0, 1, 1, 0, 1]), 0),
        (lambda: surv.Geometric.fit([1, 2, 3, 2, 1, 4]), 0),
    ],
)
def test_608_p_is_the_fitted_probability_where_the_distribution_has_one(
    fit, index
):
    # It read 1, the limited-failure proportion, on these models.
    model = fit()
    assert no_warnings(lambda: model.p) == model.params[index]
    assert model.lfp_p == 1
    with pytest.raises(AttributeError, match="params"):
        model.p = 0.5


def test_608_the_old_names_still_work_with_a_warning():
    new = _lfp_weibull(fixed={"lfp_p": 0.8})
    with pytest.warns(DeprecationWarning, match="fixed=\\{'lfp_p'"):
        old = Weibull.fit(LFP_X, LFP_C, lfp=True, fixed={"p": 0.8})
    np.testing.assert_array_equal(old.params, new.params)
    assert old.lfp_p == new.lfp_p == 0.8

    model = _lfp_weibull()
    with pytest.warns(DeprecationWarning, match="param_cb\\('lfp_p'\\)"):
        bound = model.param_cb("p")
    np.testing.assert_array_equal(bound, model.param_cb("lfp_p"))

    with pytest.warns(DeprecationWarning, match="use 'lfp_p'"):
        built = Weibull.from_params([10, 2], p=0.7)
    assert built.lfp_p == 0.7 and built.extras == {"lfp_p": 0.7}

    with pytest.raises(ValueError, match="'lfp_p' only"):
        Weibull.fit(LFP_X, LFP_C, lfp=True, fixed={"p": 0.8, "lfp_p": 0.8})


def test_608_saved_models_keep_their_format_and_old_ones_load():
    import pickle

    from surpyval import Parametric

    model = _lfp_weibull(fixed={"lfp_p": 0.8})
    saved = model.to_dict()
    # The dict's keys are as before v0.23, so every reader reads it.
    assert saved["p"] == 0.8 and saved["fixed"] == ["p"]
    restored = no_warnings(surv.from_dict, saved)
    assert restored.lfp_p == 0.8 and restored.extras == model.extras
    # A model pickled before v0.23 holds the proportion as ``p``.
    state = pickle.loads(pickle.dumps(model)).__dict__.copy()
    state["p"] = state.pop("lfp_p")
    old = Parametric.__new__(Parametric)
    old.__setstate__(state)
    assert old.lfp_p == 0.8
    np.testing.assert_array_equal(old.sf([5, 50]), model.sf([5, 50]))


def test_608_regression_models_spell_it_alike():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(40, 1))
    model = no_warnings(
        surv.WeibullPH.fit, Weibull.random(40, 10, 2, random_state=1), Z
    )
    assert model.lfp_p == 1.0
    with pytest.warns(DeprecationWarning, match="use 'lfp_p'"):
        assert model.p == 1.0
