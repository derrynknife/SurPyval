"""Accelerated Life (parameter-substitution) fitting.

Guards the life-parameter map that couples each distribution to its
stress-relationship life model, and checks that an Exponential Accelerated
Life model fits and recovers the underlying stress-life relationship (a
regression: its life parameter was mis-named ``"lambda"`` where the
distribution actually calls it ``"failure_rate"``, so the fit raised
``KeyError: 'lambda'``).
"""

import warnings

import numpy as np
import pytest

import surpyval
from surpyval import (
    AcceleratedLife,
    Exponential,
    Gamma,
    GammaFrailty,
    LogNormal,
    Weibull,
)
from surpyval.life_models import Power
from surpyval.tests._helpers import fitted_accelerated_life_model
from surpyval.univariate.regression import (
    DualPower,
    ExponentialLifeModel,
    InversePower,
    Linear,
)
from surpyval.univariate.regression.accelerated_life.accelerated_life import (
    _LIFE_PARAM_MAP,
)


def test_life_param_map_names_a_real_parameter():
    # Every distribution's declared life parameter must be one of that
    # distribution's actual parameters, or ``fit`` fails when it looks the
    # index up in ``param_map``.
    for dist_name, (life_param, _, _) in _LIFE_PARAM_MAP.items():
        dist = getattr(surpyval, dist_name)
        assert life_param in dist.parameter_names, (
            f"{dist_name} life parameter {life_param!r} is not in "
            f"{dist.parameter_names}"
        )


def _al_stress_data(phi, stresses, per=400, seed=0):
    rng = np.random.default_rng(seed)
    xs, Zs = [], []
    for s in stresses:
        xs.append(rng.exponential(phi(s), per))
        Zs.append(np.full(per, s))
    x = np.concatenate(xs)
    Z = np.concatenate(Zs).reshape(-1, 1)
    return x, Z


def test_exponential_accelerated_life_fits_and_recovers():
    # Exponential lifetimes whose mean life follows a power law in the stress.
    stresses = [1.0, 2.0, 4.0, 8.0]

    def true_life(s):
        return 2000.0 * s**-1.5

    x, Z = _al_stress_data(true_life, stresses)

    # Must not raise (previously KeyError: 'lambda').
    model = AcceleratedLife(Exponential, Power).fit(x=x, Z=Z)

    # The fitted model's mean life at each stress (= 1 / failure_rate) should
    # recover the true power-law life within sampling error.
    for s in stresses:
        rate = np.ravel(
            model.model.param_transform(
                model.reg_model.phi(np.array([[s]]), *model.phi_params)
            )
        )[0]
        assert np.isclose(1.0 / rate, true_life(s), rtol=0.15)


def test_exponential_accelerated_life_round_trips():
    stresses = [1.0, 2.0, 4.0, 8.0]
    x, Z = _al_stress_data(lambda s: 2000.0 * s**-1.5, stresses, seed=1)
    model = AcceleratedLife(Exponential, Power).fit(x=x, Z=Z)

    restored = surpyval.from_dict(model.to_dict())
    xq = np.linspace(1.0, 500.0, 10)
    Zq = np.full((xq.size, 1), 2.0)
    assert np.allclose(model.sf(xq, Zq), restored.sf(xq, Zq), equal_nan=True)


def test_weibull_accelerated_life_still_fits():
    # A guard that the fix did not disturb the other (already-working)
    # distributions.
    stresses = [1.0, 2.0, 4.0, 8.0]
    rng = np.random.default_rng(2)
    xs, Zs = [], []
    for s in stresses:
        life = 500.0 * s**-1.0
        xs.append(life * rng.weibull(2.2, 200))
        Zs.append(np.full(200, s))
    x = np.concatenate(xs)
    Z = np.concatenate(Zs).reshape(-1, 1)

    model = AcceleratedLife(Weibull, Power).fit(x=x, Z=Z)
    assert np.all(np.isfinite(model.params))


def test_gamma_life_is_the_reciprocal_of_its_rate():
    # Gamma's beta is a rate, like the Exponential's failure_rate, so the
    # life model must enter through 1 / life. With shape 1 the Gamma is
    # the Exponential, so on exponential data the two agree.
    from surpyval import AcceleratedLife, Exponential, Gamma
    from surpyval.life_models import InversePower

    rng = np.random.default_rng(0)
    stress = np.repeat([1.0, 2.0, 4.0], 60)
    x = rng.exponential(1000 * stress**-1.5)
    expo = AcceleratedLife(Exponential, InversePower).fit(x, Z=stress)
    gamma = AcceleratedLife(Gamma, InversePower).fit(x, Z=stress)
    assert gamma.params[0] == pytest.approx(1.0, abs=0.05)
    assert gamma.params[2:] == pytest.approx(expo.params[1:], rel=0.02)


# -- #489: the substituted life parameter, parameter_names, one stress level


def _weibull_power(levels=(20.0, 30.0, 40.0)):
    rng = np.random.default_rng(1)
    stress = np.repeat(levels, 40)
    x = 10 * rng.weibull(3, stress.size) * (100.0 / stress)
    return x, stress


def test_repr_does_not_show_the_life_parameter_as_a_fitted_value():
    # It printed "alpha: 1.0", read as a characteristic life of 1
    x, stress = _weibull_power()
    model = AcceleratedLife(Weibull, Power).fit(x, Z=stress)
    text = repr(model)
    assert "alpha: 1.0" not in text
    assert "alpha: L(Z) of the Power life model" in text
    # The fitted parameters are the rows of the tables (#484).
    index = model.summary().index
    assert ("baseline", "beta") in index and ("life model", "n") in index
    assert ("baseline", "alpha") not in index
    lognormal = AcceleratedLife(surpyval.LogNormal, Power).fit(x, Z=stress)
    assert "mu: ln L(Z) of the Power life model" in repr(lognormal)
    expo = AcceleratedLife(Exponential, Power).fit(x, Z=stress)
    assert "failure_rate: 1 / L(Z)" in repr(expo)


def test_parameter_names_and_life_parameter():
    x, stress = _weibull_power()
    model = AcceleratedLife(Weibull, Power).fit(x, Z=stress)
    assert model.parameter_names == ["alpha", "beta", "a", "n"]
    assert len(model.parameter_names) == model.params.size
    assert model.life_parameter == "alpha"
    restored = surpyval.from_dict(model.to_dict())
    assert restored.parameter_names == model.parameter_names
    assert restored.life_parameter == "alpha"
    assert "alpha: L(Z)" in repr(restored)
    # the other regression families have parameter_names and no life parameter
    ph = surpyval.WeibullPH.fit(x, stress.reshape(-1, 1))
    assert ph.parameter_names == ["alpha", "beta", "beta_0"]
    assert ph.life_parameter is None


def test_fitter_param_names_is_the_deprecated_alias():
    # The accelerated-life fitter lost its ``param_names`` when the other
    # fitters got the deprecated alias (an AttributeError); it now mirrors
    # its distribution like them (consolidation sweep).
    fitter = AcceleratedLife(Weibull, Power)
    with pytest.warns(DeprecationWarning, match="param_names is deprecated"):
        names = fitter.param_names
    assert names == fitter.parameter_names == ["alpha", "beta"]


def test_param_cb_refuses_the_life_parameter():
    x, stress = _weibull_power()
    model = AcceleratedLife(Weibull, Power).fit(x, Z=stress)
    with pytest.raises(ValueError, match="not a parameter.*a, n"):
        model.param_cb("alpha")
    lo, hi = model.param_cb("n")
    assert lo < model.params[3] < hi


def test_one_stress_level_says_why():
    # It said "Insufficient data at separate Z values. Try manually
    # setting initial guess using 'init'"; no init identifies (a, n).
    x, stress = _weibull_power(levels=(20.0,))
    with pytest.raises(ValueError, match="at least two distinct stress"):
        AcceleratedLife(Weibull, Power).fit(x, Z=stress)
    with pytest.raises(ValueError, match="`init`"):
        AcceleratedLife(Weibull, Power).fit(x, Z=stress, fixed={"n": -1.0})


def test_one_stress_level_with_init_warns_not_identifiable():
    x, stress = _weibull_power(levels=(20.0,))
    with pytest.warns(UserWarning, match="not identifiable") as record:
        AcceleratedLife(Weibull, Power).fit(
            x, Z=stress, init=[1.0, 3.0, 500.0, -1.0]
        )
    assert [w.filename for w in record] == [__file__]
    # with n fixed, one level identifies a: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = AcceleratedLife(Weibull, Power).fit(
            x, Z=stress, init=[1.0, 3.0, 500.0, -1.0], fixed={"n": -1.0}
        )
    # the life at stress 20 is a * 20**n, about 10 * 100 / 20 = 50
    assert model.params[2] / 20.0 == pytest.approx(50.0, rel=0.1)


# ---------------------------------------------------------------------------
# Refit and ``fixed`` (#261).
# ---------------------------------------------------------------------------


def test_accelerated_life_refit_and_fixed():
    np.random.seed(4)
    stress = np.repeat([1.0, 2.0, 3.0, 4.0], 50)
    x = (100 / stress) * (-np.log(np.random.uniform(size=len(stress)))) ** (
        1 / 3
    )
    fitter = AcceleratedLife(Weibull, Power)
    m1 = fitter.fit(x, Z=stress)  # 1-D stress vector (#261)
    assert np.isfinite(np.atleast_1d(m1.sf([50.0], np.array([1.0])))).all()
    # A second fit on the same fitter instance used to corrupt the
    # parameter map; and user-fixed parameters were dropped from
    # ``model.fixed`` so SEs were reported for constrained parameters.
    m2 = fitter.fit(x, Z=stress, fixed={"beta": 3.0})
    assert "beta" in m2.fixed
    assert m2.params[1] == pytest.approx(3.0, abs=1e-9)


# ---------------------------------------------------------------------------
# ``random()`` size per stress; ``k`` excludes the placeholder.
# ---------------------------------------------------------------------------


def test_accelerated_life_random_size_per_stress():
    model = fitted_accelerated_life_model()
    # A pair of stresses gives size draws at each: 2 * size, never size**2
    # (the old uniform (low, high) option drew size stresses and then size
    # draws at each).
    for Zq in ((1.0, 2.0), [1.0, 2.0], [[1.0], [2.0]]):
        draws, rows = model.model.random(3, Zq, *model.params)
        assert draws.shape == (6,)
        assert rows.shape == (6, 1)
    draws, rows = model.random(3, 2.0)
    assert draws.shape == (3,)
    assert np.all(rows == 2.0)


def test_accelerated_life_k_excludes_placeholder():
    model = fitted_accelerated_life_model()
    assert len(model.params) == 4
    assert model.k == 3
    assert model.aic() == pytest.approx(2 * 3 + 2 * model.neg_ll())


# ---------------------------------------------------------------------------
# Covariate shapes, a tiny parameter's covariance, the life
# models' starts and stresses, and ragged ``x``.
# ---------------------------------------------------------------------------


def test_two_stress_accelerated_life_accepts_a_1d_row():
    rng = np.random.default_rng(0)
    T = np.repeat([300.0, 350.0, 400.0], 30)
    V = np.tile(np.repeat([1.0, 2.0, 3.0], 10), 3)
    x = 100 * rng.weibull(2.0, 90) * (T / 300) ** -2 * V**-1
    model = AcceleratedLife(Weibull, DualPower).fit(
        x, Z=np.column_stack([T, V])
    )
    row = [320.0, 2.0]
    np.testing.assert_allclose(model.sf([5.0], row), model.sf([5.0], [row]))
    np.testing.assert_allclose(model.hf([5.0], row), model.hf([5.0], [row]))
    np.testing.assert_allclose(model.cb([5.0], row), model.cb([5.0], [row]))
    draws, Z_out = model.random(3, row)
    assert draws.shape == (3,) and Z_out.shape == (3, 2)


def test_single_stress_accelerated_life_accepts_a_scalar_stress():
    stress = np.repeat([300.0, 350.0, 400.0], 30)
    rng = np.random.default_rng(0)
    x = rng.weibull(2.0, 90) * 1e-3 * np.exp(3000 / stress)
    model = AcceleratedLife(Weibull, ExponentialLifeModel).fit(x, Z=stress)
    np.testing.assert_allclose(
        model.sf([10.0, 5.0], 300.0), model.sf([10.0, 5.0], [300.0, 300.0])
    )
    np.testing.assert_allclose(
        model.cb([10.0, 5.0], 300.0), model.cb([10.0, 5.0], [300.0, 300.0])
    )


def _arrhenius_data() -> tuple:
    stress = np.repeat([300.0, 350.0, 400.0], 40)
    rng = np.random.default_rng(0)
    x = rng.weibull(2.0, 120) * 1e-3 * np.exp(3000 / stress)
    return x, stress


def test_accelerated_life_covariance_with_tiny_parameter():
    x, stress = _arrhenius_data()
    model = AcceleratedLife(Weibull, InversePower).fit(x, Z=stress)
    assert model.params[2] < 1e-15
    se = model.standard_errors()
    assert np.all(np.isfinite(se))
    power = AcceleratedLife(Weibull, Power).fit(x, Z=stress)
    # InversePower's n is Power's -n, with the same standard error.
    assert se[3] == pytest.approx(power.standard_errors()[3], rel=1e-2)
    restored = surpyval.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.standard_errors(), se)


def test_power_life_model_refuses_non_positive_stress(capfd):
    x, stress = _arrhenius_data()
    with pytest.raises(ValueError, match="strictly positive stresses"):
        AcceleratedLife(Weibull, Power).fit(x, Z=stress - 350)
    assert capfd.readouterr().err == ""


@pytest.mark.parametrize("dist", [Weibull, LogNormal, Exponential, Gamma])
def test_linear_life_model_finds_a_feasible_start(dist):
    x, stress = _arrhenius_data()
    model = AcceleratedLife(dist, Linear).fit(x, Z=stress)
    assert np.all(model.phi(np.array([300.0, 350.0, 400.0])) > 0)
    assert np.isfinite(model.neg_ll())


def test_accelerated_life_and_gamma_frailty_accept_ragged_x():
    x = [10, [11, 13], 12, 9, [20, 25], 30, 14, 15, 16]
    c = [0, 2, 0, 0, 2, 0, 0, 0, 0]
    model = AcceleratedLife(Weibull, Power).fit(
        x, Z=[1, 1, 1, 2, 2, 2, 3, 3, 3], c=c
    )
    assert np.all(np.isfinite(model.params))
    # The frailty likelihood has no interval term, so an interval row is
    # refused by name; the ragged form itself no longer crashes.
    with pytest.raises(ValueError, match="right-censored"):
        GammaFrailty.fit(x, groups=[0, 0, 0, 0, 0, 1, 1, 1, 1], c=c)
    exact = [10, 11, 12, 9, 20, 30, 14, 15, 16, 3]
    two_col = np.column_stack([exact, exact])
    groups = [0] * 5 + [1] * 5
    np.testing.assert_allclose(
        GammaFrailty.fit(two_col, groups=groups).dist_params,
        GammaFrailty.fit(exact, groups=groups).dist_params,
    )
