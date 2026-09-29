"""The log-linear parametric regressions centre their covariates (#463).

``exp(beta'Z)`` on a covariate far from 0 (a year, a date as a day count)
overflowed inside the fit, and the optimiser came back, silently, with a
wrong answer: 2000 added to a N(0, 1) covariate turned a WeibullPH
coefficient of 0.907 into 0.0247, with a scale of 1.1e19 (principles 12
and 13). The fits now run on the covariates centred at their ``n``-weighted
means. For the family/link pairs where that is an exact
reparameterisation (``ORIGIN_MAPS``), the model is the same whatever the
covariates' origin, and a shift changes neither the coefficients nor any
prediction or bound. The baseline is reported at ``Z = 0``, as before,
whenever that is representable; otherwise the model keeps it at the means,
as ``center``, and says so.
"""

import copy
import json
import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.serialisation import required_schema
from surpyval.univariate.regression import StepSchedule
from surpyval.univariate.regression._fit_skeleton import ORIGIN_MAPS, Centring
from surpyval.univariate.regression.parametric_regression_model import (
    ParametricRegressionModel,
)

OFFSETS = [0.0, 300.0, 2000.0, 1e5]
TIMES = np.array([0.5, 2.0, 5.0, 10.0, 20.0])
# One of each kind with an exact map, and the families of the brief.
FITTERS = [
    "WeibullPH",
    "ExponentialPH",
    "GumbelPH",
    "WeibullAFT",
    "LogNormalAFT",
    "GammaAFT",
    "NormalAFT",
    "LogisticPO",
]
QUERY = np.array([[0.3, 1.0], [-1.2, 0.0], [1.5, 1.0]])


def _data(size=200, seed=1):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=size)
    g = rng.integers(0, 2, size).astype(float)
    t = rng.weibull(1.5, size) * 10 * np.exp(-(0.8 * z + 0.6 * g) / 1.5)
    cens = rng.uniform(0, 25, size)
    x = np.round(np.minimum(t, cens), 3) + 1e-3
    c = (cens < t).astype(int)
    return x, np.column_stack([z, g]), c


def _shift(offset):
    # The first column moved, the second (binary) left alone.
    return np.array([offset, 0.0])


def _fit_quietly(fit, *args, **kwargs):
    # No warning of any kind: before the fix a large offset gave none
    # either -- just a wrong answer -- and Fine-Gray leaked numpy's.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit(*args, **kwargs)


@pytest.fixture(scope="module")
def references():
    x, Z, c = _data()
    return {
        name: getattr(sp, name).fit(x, Z, c=c) for name in FITTERS
    }


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("name", FITTERS)
def test_offset_leaves_the_model_unchanged(name, offset, references):
    x, Z, c = _data()
    ref = references[name]
    s = _shift(offset)
    model = _fit_quietly(getattr(sp, name).fit, x, Z + s, c=c)
    k = model.k_dist

    np.testing.assert_allclose(
        model.params[k:], ref.params[k:], rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(-model.neg_ll(), -ref.neg_ll(), rtol=1e-7)
    for zq in QUERY:
        for fn in ("sf", "Hf", "hf", "df", "ff"):
            got = getattr(model, fn)(TIMES, zq + s)
            want = getattr(ref, fn)(TIMES, zq)
            assert np.all(np.isfinite(got)), fn
            # Optimiser tolerance: the Nelder-Mead rung of the AFT and PO
            # fits stops on a slightly different point for the same data
            # centred with different rounding; 5e-4 is its size in the
            # far tail (sf 3e-5), where Hf agrees to 3e-5.
            np.testing.assert_allclose(
                got, want, rtol=5e-4, atol=1e-10, err_msg=fn
            )
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for on in ("sf", "Hf", "hf"):
                np.testing.assert_allclose(
                    model.cb(TIMES, zq + s, on=on),
                    ref.cb(TIMES, zq, on=on),
                    rtol=2e-3,
                    err_msg=on,
                )
    for coef in ("beta_0", "beta_1"):
        np.testing.assert_allclose(
            model.param_cb(coef), ref.param_cb(coef), rtol=1e-3, atol=1e-5
        )

    if not model._has_center():
        # Reported at Z = 0, which is Z = -s in the reference's frame: the
        # baseline moved from there to the reference's origin, s in this
        # frame, is the reference's. (Moved with the model's own
        # coefficients: an optimiser-tolerance difference in beta is
        # multiplied by the offset in the reported baseline.)
        entry = ORIGIN_MAPS[(model.kind, model.distribution.name)]
        moved = Centring(s, k, entry[1]).from_origin(model.params)
        np.testing.assert_allclose(
            np.asarray(moved, float)[:k], ref.params[:k], rtol=2e-4
        )


def test_the_baseline_is_at_zero_whenever_it_can_be():
    # 300 on a coefficient of 0.9 moves log(alpha) by 200: representable,
    # so reported at Z = 0 as before; 2000 is not (exp(1800) overflows).
    x, Z, c = _data()
    near = sp.WeibullPH.fit(x, Z + _shift(300.0), c=c)
    far = sp.WeibullPH.fit(x, Z + _shift(2000.0), c=c)
    np.testing.assert_array_equal(near.center, [0.0, 0.0])
    assert "center" not in near.to_dict()
    assert "Baseline at" not in repr(near)
    # Kept at the n-weighted covariate means.
    np.testing.assert_allclose(far.center, np.mean(Z + _shift(2000.0), 0))
    np.testing.assert_allclose(
        far.Hf(TIMES, far.center),
        sp.Weibull.Hf(TIMES, *far.params[:2]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(far.phi(far.center), 1.0)
    assert "Baseline at         : the covariate means" in repr(far)


def test_counts_weight_the_center():
    x, Z, c = _data()
    n = np.random.default_rng(0).integers(1, 4, x.size)
    model = sp.WeibullAFT.fit(x, Z + _shift(1e5), c=c, n=n)
    np.testing.assert_allclose(
        model.center, np.average(Z + _shift(1e5), axis=0, weights=n)
    )


def test_bounds_match_those_of_the_uncentred_computation():
    # The covariance of a model reported at Z = 0 is the centred fit's,
    # carried over by the jacobian of the map; cb is computed in the
    # centred parameterisation. Both agree with the old computation (the
    # Hessian at the reported parameters, on the data as given) where
    # that is well conditioned.
    x, Z, c = _data()
    for name in FITTERS:
        model = getattr(sp, name).fit(x, Z + _shift(3.0), c=c)
        assert model._fit_centring is not None, name
        old = copy.copy(model)
        old._fit_centring = None
        cov, cov_old = model.covariance(), old.covariance()
        se = np.sqrt(np.diag(cov_old))
        # To the accuracy of the numerical Hessians.
        np.testing.assert_allclose(
            cov / np.outer(se, se), cov_old / np.outer(se, se), atol=1e-3
        )
        for p in model.parameter_names():
            np.testing.assert_allclose(
                model.param_cb(p), old.param_cb(p), rtol=1e-4, err_msg=p
            )
        zq = QUERY[0] + _shift(3.0)
        for on in ("sf", "Hf", "hf", "df"):
            np.testing.assert_allclose(
                model.cb(TIMES, zq, on=on),
                old.cb(TIMES, zq, on=on),
                rtol=1e-4,
                err_msg=f"{name} {on}",
            )


@pytest.mark.parametrize("name", ["WeibullPH", "LogNormalAFT", "LogisticPO"])
def test_save_and_load_a_baseline_kept_at_the_means(name):
    x, Z, c = _data()
    s = _shift(1e5)
    model = getattr(sp, name).fit(x, Z + s, c=c)
    assert model._has_center()
    d = json.loads(json.dumps(model.to_dict()))
    # A reader that ignored the centre would mispredict, so schema-1
    # readers refuse the dict.
    assert d["schema"] == 2 == required_schema(d)
    back = sp.from_dict(d)
    np.testing.assert_array_equal(back.center, model.center)
    for zq in QUERY:
        for fn in ("sf", "Hf", "hf", "df", "ff"):
            np.testing.assert_allclose(
                getattr(back, fn)(TIMES, zq + s),
                getattr(model, fn)(TIMES, zq + s),
                rtol=1e-13,
            )
        np.testing.assert_allclose(
            back.cb(TIMES, zq + s), model.cb(TIMES, zq + s), rtol=1e-12
        )
    for p in model.parameter_names():
        np.testing.assert_allclose(back.param_cb(p), model.param_cb(p))
    assert repr(back) == repr(model)


@pytest.mark.parametrize("name", ["WeibullPH", "LogNormalAFT", "LogisticPO"])
def test_save_and_load_a_baseline_at_zero(name):
    x, Z, c = _data()
    s = _shift(300.0 if name != "LogisticPO" else 30.0)
    model = getattr(sp, name).fit(x, Z + s, c=c)
    assert not model._has_center()
    d = json.loads(json.dumps(model.to_dict()))
    # The dict of before: no "center", schema 1.
    assert "center" not in d
    assert d["schema"] == 1 == required_schema(d)
    back = sp.from_dict(d)
    np.testing.assert_array_equal(back.center, [0.0, 0.0])
    zq = QUERY[0] + s
    np.testing.assert_allclose(
        back.sf(TIMES, zq), model.sf(TIMES, zq), rtol=1e-12
    )
    # The restored model has only the reported parameters and their
    # (carried-over) covariance, and computes its bounds there; they agree
    # with the fitted model's, which come from the centred fit.
    np.testing.assert_allclose(
        back.cb(TIMES, zq), model.cb(TIMES, zq), rtol=1e-4
    )


def test_a_dict_written_before_centring_loads_unchanged():
    # What SurPyval 0.21.0 wrote for a WeibullPH (no "center"): the
    # baseline at Z = 0.
    old = {
        "parameterization": "parametric-regression",
        "kind": "Proportional Hazard",
        "distribution": "Weibull",
        "phi_param_map": {"beta_0": 0},
        "reg_model_name": "Log Linear [e^(beta'Z)]",
        "params": [10.0, 1.5, 0.8],
        "k": 3,
        "k_dist": 2,
        "fixed": {},
        "gamma": 0.0,
        "p": 1.0,
        "f0": 0.0,
        "schema": 1,
    }
    back = sp.from_dict(old)
    np.testing.assert_array_equal(back.center, [0.0])
    np.testing.assert_allclose(
        back.sf(TIMES, [[0.5]]),
        np.exp(-np.exp(0.4) * (TIMES / 10.0) ** 1.5),
        rtol=1e-14,
    )


def test_a_center_of_the_wrong_length_is_refused():
    x, Z, c = _data()
    d = sp.WeibullPH.fit(x, Z + _shift(1e5), c=c).to_dict()
    d["center"] = [1.0]
    with pytest.raises(ValueError, match="'center' has 1 value"):
        sp.from_dict(d)


def test_random_draws_at_the_rows_given():
    x, Z, c = _data()
    s = _shift(1e5)
    model = sp.WeibullPH.fit(x, Z + s, c=c)
    ref = sp.WeibullPH.fit(x, Z, c=c)
    draws, rows = model.random(4, QUERY + s, random_state=0)
    ref_draws, ref_rows = ref.random(4, QUERY, random_state=0)
    np.testing.assert_allclose(rows, ref_rows + s)
    np.testing.assert_allclose(draws, ref_draws, rtol=1e-4)


def test_an_init_is_read_at_zero():
    # ``init`` is in the reported parameterisation (baseline at Z = 0).
    x, Z, c = _data()
    s = _shift(100.0)
    ref = sp.WeibullPH.fit(x, Z + s, c=c)
    model = _fit_quietly(sp.WeibullPH.fit, x, Z + s, c=c, init=ref.params)
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-6)


def test_fixing_a_parameter_the_centring_moves():
    # alpha fixed at Z = 0 is not a fixed alpha at the means, so that fit
    # runs on the covariates as given; fixing the shape, which the map
    # leaves alone, does not stop the centring.
    x, Z, c = _data()
    s = _shift(3.0)
    fixed_scale = sp.WeibullPH.fit(x, Z + s, c=c, fixed={"alpha": 12.0})
    assert fixed_scale._fit_centring is None
    assert fixed_scale.params[0] == 12.0
    fixed_shape = _fit_quietly(
        sp.WeibullPH.fit, x, Z + _shift(2000.0), c=c, fixed={"beta": 1.3}
    )
    ref = sp.WeibullPH.fit(x, Z, c=c, fixed={"beta": 1.3})
    np.testing.assert_allclose(fixed_shape.params[1:], ref.params[1:])
    np.testing.assert_allclose(
        fixed_shape.sf(TIMES, QUERY[0] + _shift(2000.0)),
        ref.sf(TIMES, QUERY[0]),
        rtol=1e-6,
    )


def test_pairs_without_an_exact_map_are_fitted_as_defined():
    # For LogNormal PH (and Weibull PO, ...) the model with its baseline at
    # the covariate means is not the model with its baseline at 0 -- a
    # shift of the covariates changes the maximum likelihood -- so these
    # fit the model as defined, on the covariates as given.
    x, Z, c = _data()
    for name in ("LogNormalPH", "WeibullPO", "WeibullAH", "GammaPO"):
        model = getattr(sp, name).fit(x, Z + _shift(1.0), c=c)
        assert model._fit_centring is None, name
        np.testing.assert_array_equal(model.center, [0.0, 0.0])


def test_a_custom_phi_is_not_centred():
    x, Z, c = _data()
    import autograd.numpy as anp

    fitter = sp.ProportionalHazardsFitter(
        "WeibullExpRR",
        sp.Weibull,
        lambda Z, *params: anp.exp(anp.dot(Z, anp.array(params))),
        "custom",
        phi_bounds=lambda Z: ((None, None),) * Z.shape[1],
        phi_param_map=lambda Z: {f"beta_{i}": i for i in range(Z.shape[1])},
    )
    model = fitter.fit(x, Z + _shift(1.0), c=c)
    assert model._fit_centring is None
    np.testing.assert_array_equal(model.center, [0.0, 0.0])


@pytest.mark.parametrize("name", ["WeibullPH", "WeibullPO", "WeibullAFT"])
@pytest.mark.parametrize("offset", [2000.0, 1e5])
def test_time_varying_covariates(name, offset):
    rng = np.random.default_rng(0)
    n = 200
    switch = rng.uniform(0.3, 1.5, n)
    t_low = rng.exponential(2.0, n)
    t_high = switch + rng.exponential(2.0 / np.e, n)
    T = np.where(t_low > switch, t_high, t_low)
    one = T <= switch
    i = np.r_[np.arange(n), np.flatnonzero(~one)]
    xl = np.r_[np.zeros(n), switch[~one]]
    xr = np.r_[np.where(one, T, switch), T[~one]]
    c = np.r_[np.where(one, 0, 1), np.zeros((~one).sum(), dtype=int)]
    Z = np.r_[np.zeros(n), np.ones((~one).sum())]
    fitter = getattr(sp, name)
    ref = fitter.fit_tvc(i, xl, xr, c, Z)
    model = _fit_quietly(fitter.fit_tvc, i, xl, xr, c, Z + offset)
    if name != "WeibullPO":  # no exact map: not centred, not invariant
        np.testing.assert_allclose(
            model.params[-1], ref.params[-1], rtol=1e-4
        )
        path = StepSchedule.from_changepoints([0, 1], [[0.0], [1.0]])
        moved = StepSchedule.from_changepoints(
            [0, 1], [[offset], [offset + 1.0]]
        )
        t = [0.5, 1.0, 2.0, 4.0]
        np.testing.assert_allclose(
            model.sf_tvc(t, moved), ref.sf_tvc(t, path), rtol=1e-4
        )
        np.testing.assert_allclose(
            model.cb(t, [[offset + 1.0]]), ref.cb(t, [[1.0]]), rtol=2e-3
        )


def test_the_reported_parameters_are_those_of_before():
    # Where the uncentred fit worked the reported parameters are
    # unchanged to optimiser tolerance: SurPyval 0.21.0's values on the
    # data of this module (the covariate means are -0.074 and 0.485).
    x, Z, c = _data()
    before = {
        "WeibullPH": [11.583534, 1.398521, 0.707308, 0.832909],
        "WeibullAFT": [11.583526, 1.398519, 0.505755, 0.595563],
        "LogNormalAFT": [2.040164, 0.94534, 0.484204, 0.498129],
        "LogisticPO": [9.256839, 3.161694, -0.857365, -1.066917],
        "GumbelPH": [14.019765, 5.644782, 0.74558, 1.02669],
        "GammaAFT": [1.708778, 0.161107, 0.505012, 0.582608],
    }
    for name, params in before.items():
        np.testing.assert_allclose(
            getattr(sp, name).fit(x, Z, c=c).params, params, rtol=2e-4
        )
