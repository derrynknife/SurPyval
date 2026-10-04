"""
The offset / limited-failure-population / zero-inflation surface of a
parametric model as a system simulator uses it (#403 - #407):

- #407: the zero-inflation mass arrives at 0, not before it, with or
  without an offset, and ``Hf`` is +0 where nothing has failed;
- #406: ``extras`` and ``with_params`` rebuild a model with other
  parameters and the same ``gamma``, ``p`` and ``f0``;
- #405: ``df(x, continuous=True)`` is the continuous part alone;
- #404: ``mean``, ``moment`` and ``var`` are infinite with a cure
  fraction, the defective values under ``defective=True``;
- #403: ``random`` draws lifetimes, ``qf(u)`` from one uniform per draw,
  and ``random_data`` the survival data to refit.
"""

import numpy as np
import pytest
from scipy.integrate import quad

import surpyval as surv
from surpyval import Exponential, Geometric, LogNormal, Weibull

# -- #407: nothing fails before time 0 --------------------------------------


def test_zero_inflation_before_zero_with_an_offset():
    off = Weibull.from_params([100, 2], gamma=5.0, f0=0.1)
    x = [-1.0, 0.0, 2.0]
    np.testing.assert_allclose(off.ff(x), [0.0, 0.1, 0.1])
    np.testing.assert_allclose(off.sf(x), [1.0, 0.9, 0.9])
    np.testing.assert_allclose(off.Hf(x), [0.0, -np.log(0.9), -np.log(0.9)])
    assert off.hf(-1.0) == 0.0 and off.df(-1.0) == 0.0


@pytest.mark.parametrize(
    "extras",
    [{"f0": 0.1}, {"gamma": 5.0, "f0": 0.1}, {"lfp_p": 0.8, "f0": 0.1}],
)
def test_cumulative_hazard_before_zero_is_positive_zero(extras):
    # -log(1) is -0.0, which printed as "-0." before the mass at 0.
    model = Weibull.from_params([100, 2], **extras)
    out = model.Hf([-1.0, -0.5])
    assert np.all(out == 0.0) and not np.any(np.signbit(out))
    assert not np.signbit(model.Hf(-1.0))


# -- #406: extras and with_params --------------------------------------------


def test_extras_lists_what_the_model_carries():
    m = Weibull.from_params([100, 2], gamma=5.0, lfp_p=0.9, f0=0.1)
    assert m.extras == {"gamma": 5.0, "lfp_p": 0.9, "f0": 0.1}
    assert Weibull.from_params([100, 2]).extras == {}
    assert Weibull.from_params([100, 2], lfp_p=0.9).extras == {"lfp_p": 0.9}
    # a copy: changing it leaves the model alone
    m.extras["lfp_p"] = 0.5
    assert m.lfp_p == 0.9


def test_extras_rebuild_the_model():
    m = Weibull.from_params([100, 2], gamma=5.0, lfp_p=0.9, f0=0.1)
    # from_params(params) alone was a plain Weibull (0.7788, not 0.7533)
    rebuilt = Weibull.from_params(list(m.params), **m.extras)
    assert rebuilt.sf(50.0) == pytest.approx(0.7533491860784887)
    assert rebuilt.sf(50.0) == pytest.approx(m.sf(50.0))


def test_with_params_keeps_gamma_p_and_f0():
    m = Weibull.from_params([100, 2], gamma=5.0, lfp_p=0.9, f0=0.1)
    other = m.with_params([120, 2])
    np.testing.assert_array_equal(other.params, [120, 2])
    assert (other.gamma, other.lfp_p, other.f0) == (5.0, 0.9, 0.1)
    assert (other.offset, other.lfp, other.zi) == (True, True, True)
    x = np.array([-1.0, 0.0, 3.0, 50.0, 500.0])
    expected = Weibull.from_params([120, 2], gamma=5.0, lfp_p=0.9, f0=0.1)
    np.testing.assert_allclose(other.sf(x), expected.sf(x))
    # the same parameters give the same model
    np.testing.assert_allclose(m.with_params(m.params).sf(x), m.sf(x))


def test_with_params_of_a_fitted_model():
    np.random.seed(0)
    x = Weibull.random(200, 10, 2) + 3.0
    fitted = Weibull.fit(x, offset=True)
    assert fitted.extras == {"gamma": pytest.approx(fitted.gamma)}
    again = fitted.with_params(fitted.params)
    t = np.array([2.0, 5.0, 10.0])
    np.testing.assert_allclose(again.sf(t), fitted.sf(t))
    # a model of other parameters has no fit: no data, no covariance
    assert again.method == "given parameters" and again.data is None


def test_with_params_uses_p_for_a_distribution_with_its_own_p():
    # The Geometric's own parameter is p; its proportion is lfp_p, as
    # everywhere (#608), in from_params and extras alike.
    g = Geometric.from_params([0.3], lfp_p=0.8)
    assert g.extras == {"lfp_p": 0.8}
    other = g.with_params([0.2])
    assert other.lfp_p == 0.8 and other.params[0] == 0.2


def test_with_params_validates_like_from_params():
    m = Weibull.from_params([100, 2], lfp_p=0.9)
    with pytest.raises(ValueError, match="Must have 2 params"):
        m.with_params([1, 2, 3])
    with pytest.raises(ValueError, match="must be in bounds"):
        m.with_params([-1, 2])


def test_with_params_of_other_model_kinds():
    h = surv.Hypoexponential.from_params([0.5, 1.5, 3.0])
    assert h.with_params([1.0, 2.0, 3.0]).mean() == pytest.approx(
        1 + 1 / 2 + 1 / 3
    )
    d = surv.Discretize(Weibull).from_params([10, 2], lfp_p=0.8)
    assert d.with_params([11, 2]).extras == {"lfp_p": 0.8}
    assert surv.Binomial.from_params([5, 0.3]).with_params(
        [6, 0.3]
    ).mean() == pytest.approx(1.8)


# -- #405: the continuous part of a zero-inflated density ---------------------


def test_continuous_density_has_no_point_mass():
    zi = Weibull.from_params([100, 2], f0=0.1)
    assert zi.df(0.0) == pytest.approx(0.1)  # the point mass, as before
    assert zi.df(0.0, continuous=True) == 0.0
    t = np.linspace(0, 1000, 1001)
    # the grid used to count a spurious f0 * dx / 2: 0.95, not 0.9
    assert np.trapezoid(zi.df(t, continuous=True), t) == pytest.approx(
        0.9, abs=1e-4
    )
    # elsewhere the two agree
    np.testing.assert_array_equal(zi.df(t[1:], continuous=True), zi.df(t[1:]))


def test_continuous_density_integrates_to_p_minus_f0():
    m = LogNormal.from_params([2.0, 0.5], gamma=1.0, lfp_p=0.7, f0=0.2)
    total, _ = quad(lambda t: m.df(t, continuous=True), 1.0, np.inf)
    assert total == pytest.approx(0.5, abs=1e-6)
    with np.errstate(all="ignore"):  # the base density at a negative time
        below = m.df([-1.0, 0.0, 0.5], continuous=True)
    assert below.tolist() == [0, 0, 0]


def test_continuous_is_the_default_density_without_zero_inflation():
    x = np.array([0.0, 1.0, 5.0, 20.0])
    for m in (
        Weibull.from_params([10, 2]),
        Weibull.from_params([10, 2], gamma=2.0, lfp_p=0.6),
    ):
        np.testing.assert_array_equal(m.df(x, continuous=True), m.df(x))


def test_zero_inflated_density_of_a_scalar_is_a_scalar():
    zi = Weibull.from_params([100, 2], f0=0.1)
    assert np.ndim(zi.df(0.0)) == 0 and not isinstance(zi.df(0.0), np.ndarray)


# -- #404: the mean lifetime with a cure fraction is infinite ----------------


def test_mean_of_a_limited_failure_population_is_infinite():
    lfp = Weibull.from_params([100, 2], lfp_p=0.9)
    assert lfp.mean() == np.inf
    assert lfp.moment(1) == np.inf and lfp.moment(2) == np.inf
    assert lfp.var() == np.inf
    # the defective values of before
    assert lfp.mean(defective=True) == pytest.approx(79.76042329074821)
    assert lfp.moment(2, defective=True) == pytest.approx(9000.0)
    assert lfp.var(defective=True) == pytest.approx(
        9000.0 - 79.76042329074821**2
    )
    # the order-0 moment is not a lifetime moment and is left alone
    assert lfp.moment(0) == pytest.approx(0.9)


def test_mean_without_a_cure_fraction_is_unchanged():
    for m, mean in (
        (Weibull.from_params([10, 3]), 8.929795115692489),
        (Weibull.from_params([10, 3], gamma=2.0), 10.929795115692489),
        (Weibull.from_params([10, 3], f0=0.2), 0.8 * 8.929795115692489),
        (
            Weibull.from_params([10, 3], gamma=2.0, f0=0.2),
            0.8 * 10.929795115692489,
        ),
    ):
        assert m.mean() == pytest.approx(mean)
        assert m.mean(defective=True) == m.mean()
        assert m.moment(2, defective=True) == m.moment(2)
        assert m.var(defective=True) == m.var()


def test_defective_mean_is_cached_apart_from_the_lifetime_mean():
    lfp = Exponential.from_params([0.1], lfp_p=0.5)
    assert lfp.mean(defective=True) == pytest.approx(5.0)
    assert lfp.mean() == np.inf  # not the cached defective value
    assert lfp.mean(defective=True) == pytest.approx(5.0)


# -- #403: random draws lifetimes, random_data survival data ------------------


MODELS = {
    "plain": dict(),
    "offset": dict(gamma=5.0),
    "lfp": dict(lfp_p=0.8),
    "zi": dict(f0=0.1),
    "lfp+zi": dict(lfp_p=0.8, f0=0.1),
    "offset+lfp+zi": dict(gamma=5.0, lfp_p=0.8, f0=0.1),
}


@pytest.mark.parametrize("name", MODELS)
def test_random_is_qf_of_one_uniform_per_draw(name):
    model = Weibull.from_params([10, 3], **MODELS[name])
    np.random.seed(0)
    draws = model.random(500)
    np.random.seed(0)
    expected = model.qf(np.random.random_sample(500))
    assert isinstance(draws, np.ndarray) and draws.shape == (500,)
    np.testing.assert_array_equal(draws, expected)
    # blocks replay the same stream (common random numbers)
    np.random.seed(0)
    blocks = np.concatenate([model.random(200), model.random(300)])
    np.testing.assert_array_equal(blocks, draws)


def test_random_lifetimes_of_a_limited_failure_population():
    lfp = Weibull.from_params([100, 2], lfp_p=0.9)
    np.random.seed(1)
    x = lfp.random(1000)
    never = np.isinf(x)
    # (x, c, n, t) with the never-failing units censored, before
    assert never.mean() == pytest.approx(0.1, abs=0.03)
    assert np.all(np.isfinite(x[~never])) and np.all(x[~never] > 0)
    zi = Weibull.from_params([100, 2], lfp_p=0.9, f0=0.2)
    np.random.seed(1)
    y = zi.random(10_000)
    assert np.mean(y == 0) == pytest.approx(0.2, abs=0.02)
    assert np.mean(np.isinf(y)) == pytest.approx(0.1, abs=0.02)


def test_random_keeps_the_requested_shape():
    lfp = Weibull.from_params([10, 3], lfp_p=0.8, f0=0.1)
    assert lfp.random((2, 3)).shape == (2, 3)
    assert lfp.random(1).shape == (1,)


def test_random_of_a_plain_model_is_unchanged():
    # the doctest values of Parametric.random
    model = Weibull.from_params([10, 3])
    np.random.seed(1)
    np.testing.assert_allclose(model.random(1), [8.14127103])


@pytest.mark.parametrize("name", MODELS)
def test_random_data_is_the_same_draw_as_survival_data(name):
    model = Weibull.from_params([10, 3], **MODELS[name])
    np.random.seed(4)
    lifetimes = model.random(300)
    np.random.seed(4)
    x, c, n, t = model.random_data(300)
    assert n.sum() == 300 and set(np.unique(c)) <= {0, 1}
    failed = np.sort(lifetimes[np.isfinite(lifetimes)])
    np.testing.assert_allclose(np.repeat(x[c == 0], n[c == 0]), failed)
    # the never-failing units are right-censored after the last failure
    assert n[c == 1].sum() == np.isinf(lifetimes).sum()
    if np.any(c == 1):
        assert x[c == 1][0] > failed.max()
    assert np.all(np.isinf(t[:, 0])) and np.all(np.isinf(t[:, 1]))


def test_random_data_refits_the_model():
    truth = Weibull.from_params([10, 3], lfp_p=0.6)
    np.random.seed(5)
    x, c, n, _ = truth.random_data(3000)
    fitted = Weibull.fit(x, c, n, lfp=True)
    assert fitted.lfp_p == pytest.approx(0.6, abs=0.03)
    assert fitted.params == pytest.approx([10, 3], rel=0.05)


def test_random_data_records_the_truncation():
    model = Weibull.from_params([10, 3])
    np.random.seed(6)
    x, c, n, t = model.random_data(50, a=5.0, b=12.0)
    assert np.all((x >= 5.0) & (x <= 12.0)) and np.all(c == 0)
    np.testing.assert_array_equal(t, np.tile([5.0, 12.0], (len(x), 1)))


def test_truncated_draws_are_still_refused_with_lfp_or_zi():
    with pytest.raises(NotImplementedError):
        Weibull.from_params([10, 3], lfp_p=0.8).random(5, a=1.0)
    with pytest.raises(NotImplementedError):
        Weibull.from_params([10, 3], f0=0.1).random_data(5, b=20.0)


# ---------------------------------------------------------------------------
# ``var()`` of LFP / zero-inflated models.
# ---------------------------------------------------------------------------


def test_var_follows_the_defective_convention_of_mean():
    lfp = surv.Weibull.from_params([10.0, 2.0], lfp_p=0.7)
    # The lifetime's variance is infinite with a cure fraction (#404); the
    # defective one scores the cured units at 0.
    assert np.isinf(lfp.var())
    assert lfp.var(defective=True) == pytest.approx(
        lfp.moment(2, defective=True) - lfp.mean(defective=True) ** 2
    )
    zi = surv.Weibull.from_params([10.0, 2.0], f0=0.2)
    assert zi.var() == pytest.approx(zi.moment(2) - zi.mean() ** 2)
    # the zero-inflated variance is the mixture's, checked by simulation
    np.random.seed(0)
    draws = zi.random(400_000)
    assert zi.var() == pytest.approx(np.var(draws), rel=1e-2)
    # plain and offset models are unchanged
    plain = surv.Weibull.from_params([10.0, 3.0])
    assert plain.var() == pytest.approx(10.533288486847923)
    shifted = surv.Weibull.from_params([10.0, 3.0], gamma=5.0)
    assert shifted.var() == pytest.approx(10.533288486847923)


# ---------------------------------------------------------------------------
# The zero-inflation mass arrives at zero.
# ---------------------------------------------------------------------------


W, E, G = surv.Weibull, surv.Exponential, surv.Geometric


def test_zero_inflation_mass_arrives_at_zero():
    for dist, params in ((W, [10, 2]), (G, [0.3])):
        model = dist.from_params(params, f0=0.1)
        assert model.ff(-1) == 0.0
        assert model.sf(-1) == 1.0
        assert model.ff(0) == pytest.approx(0.1)
