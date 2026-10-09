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
from surpyval.life_models import (
    Eyring,
    InverseExponential,
    InverseEyring,
    Power,
)
from surpyval.tests._helpers import (
    finite_difference_covariance,
    fitted_accelerated_life_model,
    no_warnings,
)
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
    assert ph.parameter_names == ["alpha", "beta", "coef_0"]
    assert ph.life_parameter is None


def test_fitter_parameter_names_mirror_the_distribution():
    # The accelerated-life fitter names its distribution's parameters like
    # the other fitters (consolidation sweep); ``param_names``, deprecated
    # in v0.22, is removed in v0.23.
    fitter = AcceleratedLife(Weibull, Power)
    assert fitter.parameter_names == ["alpha", "beta"]
    assert not hasattr(fitter, "param_names")


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


def _celsius_data(levels: list) -> tuple:
    TC = np.repeat(levels, 20)
    rng = np.random.default_rng(1)
    x = 1e-3 * np.exp(5000 / (TC + 273.15)) * rng.weibull(2.0, TC.size)
    return x, TC


@pytest.mark.parametrize(
    "life_model",
    [ExponentialLifeModel, InverseExponential, Eyring, InverseEyring],
)
@pytest.mark.parametrize("levels", [[0.0, 25.0, 50.0], [-40.0, 25.0, 85.0]])
def test_arrhenius_life_models_refuse_non_positive_kelvin(
    capfd, life_model, levels
):
    # A 0 degrees Celsius level crashed inside LAPACK; a negative one fitted
    # a nonsense activation energy in silence (#654).
    x, TC = _celsius_data(levels)
    with pytest.raises(ValueError, match="kelvin") as info:
        AcceleratedLife(Weibull, life_model).fit(x=x, Z=TC)
    assert "Z + 273.15" in str(info.value)
    assert capfd.readouterr().err == ""


def test_arrhenius_life_model_warns_below_200_kelvin():
    x, TC = _celsius_data([20.0, 85.0, 125.0])
    with pytest.warns(UserWarning, match="Did you pass degrees Celsius"):
        AcceleratedLife(Weibull, ExponentialLifeModel).fit(x=x, Z=TC)
    model = no_warnings(
        AcceleratedLife(Weibull, ExponentialLifeModel).fit, x=x, Z=TC + 273.15
    )
    assert model.params[2] == pytest.approx(5000, rel=0.1)


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


#: The warning of an Arrhenius life model whose stresses are all below
#: 200 K (#654), which these tests' unitless stresses give.
_BELOW_200_KELVIN = "Every stress in column"


def _separated_stress_data():
    """Two stress levels, no failure at the higher: the life there has no
    finite estimate, so neither has the life model's."""
    rng = np.random.default_rng(3)
    x = np.r_[Weibull.random(30, 10, 3, random_state=rng), np.full(30, 12.0)]
    c = np.r_[np.zeros(30), np.ones(30)]
    return x, np.repeat([20.0, 40.0], 30), c


@pytest.mark.parametrize(
    "life_model", [Power, InversePower, ExponentialLifeModel]
)
def test_555_separated_stresses_warn_no_finite_maximum_once(life_model):
    # The life-model parameters ran off (Power's n to 17, a to 3e-22)
    # silently, or with the unverified-maximum warning (InversePower).
    # They now warn as the other regressions do, once, at the caller.
    x, stress, c = _separated_stress_data()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # ExponentialLifeModel on a stress that is not a temperature (#654)
        warnings.filterwarnings("ignore", message=_BELOW_200_KELVIN)
        AcceleratedLife(Weibull, life_model).fit(x, Z=stress, c=c)
    assert len(w) == 1, [str(a.message) for a in w]
    assert str(w[0].message).startswith(
        "No finite maximum: the likelihood keeps increasing as "
        "coefficient(s) ["
    )
    assert w[0].filename == __file__


def _three_stresses():
    np.random.seed(1)
    stress = np.repeat([20.0, 30.0, 40.0], 40)
    return Weibull.random(120, 10, 3) * (100.0 / stress), stress


@pytest.mark.parametrize("dist", [Weibull, LogNormal])
@pytest.mark.parametrize(
    "life_model", [Power, InversePower, ExponentialLifeModel, Eyring]
)
def test_555_the_covariance_is_the_exact_information(dist, life_model):
    # The covariance was a numerical Hessian, 2e-5 to 2e-4 (relative to
    # the standard errors) from a Richardson-extrapolated one; it is now
    # the exact information of the fit, which agrees with that to 1e-7.
    # (Through np.where, autograd's Hessian of a LogNormal model was 6e-3
    # off: see ParameterSubstitutionFitter._dist_params_at.)
    x, stress = _three_stresses()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # ExponentialLifeModel on a stress that is not a temperature (#654)
        warnings.filterwarnings("ignore", message=_BELOW_200_KELVIN)
        model = AcceleratedLife(dist, life_model).fit(x, Z=stress)
    assert model._information is not None
    cov, ref = finite_difference_covariance(model)
    se = np.sqrt(np.diag(ref))
    np.testing.assert_allclose(
        cov / np.outer(se, se), ref / np.outer(se, se), rtol=0, atol=2e-6
    )


def _continuous_stress_data():
    rng = np.random.default_rng(3)
    stress = rng.uniform(20.0, 40.0, 120)
    x = Weibull.random(120, 10.0, 3.0, random_state=4) * (100.0 / stress)
    c = (rng.uniform(size=120) < 0.2).astype(int)
    return x, stress, c


def test_592_life_found_once_per_evaluation(monkeypatch):
    # Every row's life is found in one call of the life model, however
    # many distinct stresses there are: the distribution used to be
    # evaluated over every row once per distinct stress (120 here), and
    # the life model called as often.
    from surpyval.utils.surpyval_data import SurpyvalData

    x, stress, c = _continuous_stress_data()
    fitter = AcceleratedLife(Weibull, Power)
    data = SurpyvalData(x=x, c=c, group_and_sort=False)
    data.add_covariates(stress.reshape(-1, 1))
    calls = []
    original = fitter.phi

    def counted(Z, *params):
        calls.append(np.shape(Z))
        return original(Z, *params)

    monkeypatch.setattr(fitter, "phi", counted)
    value = fitter.neg_ll(data, 1.0, 3.0, 1000.0, -1.0)
    assert np.isfinite(value)
    # (log_df takes hf and Hf at the failures, log_sf Hf at the survivors)
    assert len(calls) <= 3


class _RowWiseLife(surpyval.life_models.LifeModel):
    """``Power`` written for a single stress row, as a custom life model
    may be: it reads the row's one stress as ``Z[0]``."""

    def __init__(self):
        super().__init__(
            "RowWise", {"a": 0, "n": 1}, ((0, None), (None, None))
        )

    def phi(self, Z, *params):
        return params[0] * Z[0] ** params[1]

    def phi_init(self, life, Z):
        return Power.phi_init(life, Z)


def test_592_a_custom_life_model_is_called_per_stress_row():
    # A custom life model is not given a matrix of rows (``phi_takes_rows``
    # is False): it gives the same likelihood, gradient and fit as the
    # built-in one.
    from autograd import jacobian

    from surpyval.utils.surpyval_data import SurpyvalData

    x, stress, c = _continuous_stress_data()
    built_in = AcceleratedLife(Weibull, Power)
    custom = AcceleratedLife(Weibull, _RowWiseLife())
    assert Power.phi_takes_rows and not _RowWiseLife().phi_takes_rows
    data = SurpyvalData(x=x, c=c, group_and_sort=False)
    data.add_covariates(stress.reshape(-1, 1))
    p = np.array([1.0, 3.0, 1000.0, -1.0])
    np.testing.assert_allclose(
        custom.neg_ll(data, *p), built_in.neg_ll(data, *p), rtol=1e-12
    )
    np.testing.assert_allclose(
        jacobian(lambda v: custom.neg_ll(data, *v))(p),
        jacobian(lambda v: built_in.neg_ll(data, *v))(p),
        rtol=1e-10,
    )
    a = built_in.fit(x, stress, c=c)
    b = custom.fit(x, stress, c=c)
    np.testing.assert_allclose(b.params, a.params, rtol=1e-6)


def test_592_a_missing_stress_is_nan_and_leaves_the_others():
    # A row with a missing stress is nan; the other rows, and the bounds
    # whose gradients pass through every row's life, are as without it.
    x, stress, c = _continuous_stress_data()
    model = AcceleratedLife(Weibull, Power).fit(x, stress, c=c)
    Z = np.array([25.0, np.nan, 35.0])
    q = np.array([2.0, 2.0, 3.0])
    sf = model.sf(q, Z)
    assert np.isnan(sf[1])
    np.testing.assert_array_equal(sf[[0, 2]], model.sf(q[[0, 2]], Z[[0, 2]]))
    cb = model.cb(q, Z)
    assert np.all(np.isnan(cb[1]))
    np.testing.assert_array_equal(cb[[0, 2]], model.cb(q[[0, 2]], Z[[0, 2]]))


def _arrhenius_b_data() -> tuple:
    k = 8.617333e-5
    a = 0.7 / k
    T = np.repeat([358.15, 378.15, 398.15], 30)
    b = 1000 * np.exp(-a / 398.15)
    rng = np.random.default_rng(1)
    x = b * np.exp(a / T) * rng.weibull(2.2, T.size)
    c = (x > 6000).astype(int)
    return np.minimum(x, 6000), c, T


def test_wald_bound_on_a_positive_life_model_parameter_is_log_scale():
    # Arrhenius's b is bounded (0, None): its Wald bound went below zero,
    # [-3.5e-06, 7.0e-06] for b = 1.75e-06 (#655).
    x, c, T = _arrhenius_b_data()
    model = AcceleratedLife(Weibull, ExponentialLifeModel).fit(x, Z=T, c=c)
    b = model.params[3]
    se = model.standard_errors()[3]
    lower, upper = model.param_cb("b")
    assert 0 < lower < b < upper
    # Symmetric on the log scale, with the delta-method se of log b.
    np.testing.assert_allclose(
        [np.log(lower), np.log(upper)],
        np.log(b) + np.array([-1, 1]) * 1.959963984540054 * se / b,
    )
    assert model.summary().loc[("life model", "b"), "coef lower 95%"] > 0
    # The unbounded a keeps its linear-scale bound.
    lo_a, hi_a = model.param_cb("a")
    assert (lo_a + hi_a) / 2 == pytest.approx(model.params[2])
    restored = surpyval.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.param_cb("b"), [lower, upper])


def test_regression_bounds_accept_method_none():
    # None means the default, as for the univariate models (#655).
    x, c, T = _arrhenius_b_data()
    model = AcceleratedLife(Weibull, ExponentialLifeModel).fit(x, Z=T, c=c)
    np.testing.assert_array_equal(
        model.param_cb("b", method=None), model.param_cb("b")
    )
    np.testing.assert_array_equal(
        model.cb(5000.0, 378.15, method=None), model.cb(5000.0, 378.15)
    )
    np.testing.assert_array_equal(
        model.quantile_cb(0.1, 378.15, method=None),
        model.quantile_cb(0.1, 378.15),
    )
    po = surpyval.ProportionalOdds.fit(x, T - 378.15, c=c)
    np.testing.assert_array_equal(
        po.param_cb("coef_0", method=None), po.param_cb("coef_0")
    )


@pytest.mark.parametrize("seed", range(6))
def test_746_the_fit_does_not_depend_on_the_row_order(seed):
    # The fit runs on its rows sorted by every column (canonical_order),
    # as the PH, AFT and PO fits do (#728), so it is the same to the last
    # digit in any order. Before, on small random designs like these, the
    # answers moved with the order by up to 1e-7 where verified, and by 5%
    # where a level with only censored rows ran off (seed 3).
    rng = np.random.default_rng(seed)
    dist = (Weibull, LogNormal, Exponential)[seed % 3]
    life_model = (Power, ExponentialLifeModel)[seed // 3]
    levels = rng.choice([300.0, 330.0, 360.0, 400.0], size=3, replace=False)
    Z = rng.choice(levels, 10)
    Z[:3] = levels
    x = np.round(rng.exponential(10, 10) * np.exp(1500 / Z - 1500 / 330), 1)
    x = x + 0.1
    c = (rng.uniform(size=10) < 0.3).astype(int)
    if seed % 2:
        c[Z == levels[0]] = 1
    n = rng.integers(1, 4, 10)
    fits = []
    for order in (np.arange(10), rng.permutation(10)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = AcceleratedLife(dist, life_model).fit(
                x[order], Z=Z[order], c=c[order], n=n[order]
            )
        fits.append((model, [str(m.message) for m in w]))
    (a, wa), (b, wb) = fits
    np.testing.assert_array_equal(a.params, b.params)
    assert (a.maximum, a.neg_ll(), wa) == (b.maximum, b.neg_ll(), wb)
    # model.data keeps the rows in the order the fit ran them
    np.testing.assert_array_equal(a.data.x, b.data.x)
    np.testing.assert_array_equal(a.data.Z, b.data.Z)
    assert np.all(np.diff(a.data.x) >= 0)
