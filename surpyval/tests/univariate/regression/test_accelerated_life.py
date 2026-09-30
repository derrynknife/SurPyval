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
from surpyval import AcceleratedLife, Exponential, Power, Weibull
from surpyval.univariate.regression.accelerated_life.accelerated_life import (
    _LIFE_PARAM_MAP,
)


def test_life_param_map_names_a_real_parameter():
    # Every distribution's declared life parameter must be one of that
    # distribution's actual parameters, or ``fit`` fails when it looks the
    # index up in ``param_map``.
    for dist_name, (life_param, _, _) in _LIFE_PARAM_MAP.items():
        dist = getattr(surpyval, dist_name)
        assert life_param in dist.param_names, (
            f"{dist_name} life parameter {life_param!r} is not in "
            f"{dist.param_names}"
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
    from surpyval import AcceleratedLife, Exponential, Gamma, InversePower

    rng = np.random.default_rng(0)
    stress = np.repeat([1.0, 2.0, 4.0], 60)
    x = rng.exponential(1000 * stress**-1.5)
    expo = AcceleratedLife(Exponential, InversePower).fit(x, Z=stress)
    gamma = AcceleratedLife(Gamma, InversePower).fit(x, Z=stress)
    assert gamma.params[0] == pytest.approx(1.0, abs=0.05)
    assert gamma.params[2:] == pytest.approx(expo.params[1:], rel=0.02)


# -- #489: the substituted life parameter, param_names, one stress level ---


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


def test_param_names_and_life_parameter():
    x, stress = _weibull_power()
    model = AcceleratedLife(Weibull, Power).fit(x, Z=stress)
    assert model.param_names == ["alpha", "beta", "a", "n"]
    assert len(model.param_names) == model.params.size
    assert model.life_parameter == "alpha"
    restored = surpyval.from_dict(model.to_dict())
    assert restored.param_names == model.param_names
    assert restored.life_parameter == "alpha"
    assert "alpha: L(Z)" in repr(restored)
    # the other regression families have param_names and no life parameter
    ph = surpyval.WeibullPH.fit(x, stress.reshape(-1, 1))
    assert ph.param_names == ["alpha", "beta", "beta_0"]
    assert ph.life_parameter is None


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
