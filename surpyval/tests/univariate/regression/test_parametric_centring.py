"""The parametric regressions and a covariate far from 0 (#463).

``exp(beta'Z)`` on a covariate far from 0 (a year, a date as a day count)
overflowed inside the fit, and the optimiser came back, silently, with a
wrong answer: 2000 added to a N(0, 1) covariate turned a WeibullPH
coefficient of 0.707 into 0.0247, with a scale of 2.4e18 (principles 12
and 13). Now:

- ``center=True`` fits on the covariates centred at their ``n``-weighted
  means and keeps the baseline there (``model.center``), for every family:
  a shift of the covariates changes nothing.
- By default the baseline is reported at ``Z = 0``, as before. Where the
  family/link has an exact map between the two (``ORIGIN_MAPS``) the fit
  still runs centred and maps back, so it is the same model whatever the
  origin, and refuses (pointing to ``center=True``) where the baseline at
  0 cannot be represented. Elsewhere the fit is at ``Z = 0`` as it always
  was, and refuses covariates far from 0 that break it.
"""

import copy
import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.serialisation import required_schema
from surpyval.univariate.regression import StepSchedule
from surpyval.univariate.regression._fit_skeleton import ORIGIN_MAPS, Centring

OFFSETS = [0.0, 300.0, 2000.0, 1e5]
TIMES = np.array([0.5, 2.0, 5.0, 10.0, 20.0])
QUERY = np.array([[0.3, 1.0], [-1.2, 0.0], [1.5, 1.0]])
# The families with an exact map between the baseline at the means and at
# Z = 0, one or more of each kind.
MAPPED = [
    "WeibullPH",
    "ExponentialPH",
    "GumbelPH",
    "WeibullAFT",
    "LogNormalAFT",
    "GammaAFT",
    "NormalAFT",
    "LogisticPO",
]
# With center=True: some of those, and some without a map.
CENTRED = [
    "WeibullPH",
    "LogNormalPH",
    "WeibullAFT",
    "LogNormalAFT",
    "WeibullPO",
    "LogisticPO",
    "WeibullAH",
]


def _data(size=200, seed=1, effect=0.8):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=size)
    g = rng.integers(0, 2, size).astype(float)
    t = rng.weibull(1.5, size) * 10 * np.exp(-(effect * z + 0.6 * g) / 1.5)
    cens = rng.uniform(0, 25, size)
    x = np.round(np.minimum(t, cens), 3) + 1e-3
    c = (cens < t).astype(int)
    return x, np.column_stack([z, g]), c


def _shift(offset):
    # The first column moved, the second (binary) left alone.
    return np.array([offset, 0.0])


def _fit_quietly(fit, *args, **kwargs):
    # No warning of any kind: before the fix a large offset gave none
    # either -- just a wrong answer.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return fit(*args, **kwargs)


def _same_predictions(model, ref, s, cb=True, quiet=True):
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
        if cb:
            with warnings.catch_warnings():
                # (An additive hazards cb where the fitted H is negative
                # leaks numpy's overflow warning, centred or not: #465.)
                warnings.simplefilter("error" if quiet else "ignore")
                for on in ("sf", "Hf", "hf"):
                    np.testing.assert_allclose(
                        model.cb(TIMES, zq + s, on=on),
                        ref.cb(TIMES, zq, on=on),
                        rtol=2e-3,
                        err_msg=on,
                    )


@pytest.fixture(scope="module")
def centred_references():
    x, Z, c = _data()
    return {
        name: getattr(sp, name).fit(x, Z, c=c, center=True) for name in CENTRED
    }


@pytest.mark.parametrize("offset", OFFSETS)
@pytest.mark.parametrize("name", CENTRED)
def test_center_true_leaves_the_model_unchanged(
    name, offset, centred_references
):
    x, Z, c = _data()
    ref = centred_references[name]
    s = _shift(offset)
    model = _fit_quietly(getattr(sp, name).fit, x, Z + s, c=c, center=True)
    np.testing.assert_allclose(model.center, ref.center + s, rtol=1e-12)
    # The baseline is at the means, so every parameter is the same.
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-4, atol=1e-6)
    np.testing.assert_allclose(-model.neg_ll(), -ref.neg_ll(), rtol=1e-7)
    _same_predictions(model, ref, s, quiet=name != "WeibullAH")
    for p in model.parameter_names():
        np.testing.assert_allclose(
            model.param_cb(p), ref.param_cb(p), rtol=1e-3, atol=1e-5
        )


@pytest.fixture(scope="module")
def references():
    x, Z, c = _data()
    return {name: getattr(sp, name).fit(x, Z, c=c) for name in MAPPED}


@pytest.mark.parametrize("offset", [0.0, 300.0])
@pytest.mark.parametrize("name", MAPPED)
def test_by_default_a_mapped_family_is_the_same_model(
    name, offset, references
):
    x, Z, c = _data()
    ref = references[name]
    s = _shift(offset)
    model = _fit_quietly(getattr(sp, name).fit, x, Z + s, c=c)
    k = model.k_dist
    np.testing.assert_array_equal(model.center, [0.0, 0.0])
    np.testing.assert_allclose(
        model.params[k:], ref.params[k:], rtol=1e-4, atol=1e-6
    )
    np.testing.assert_allclose(-model.neg_ll(), -ref.neg_ll(), rtol=1e-7)
    _same_predictions(model, ref, s)
    for coef in ("beta_0", "beta_1"):
        np.testing.assert_allclose(
            model.param_cb(coef), ref.param_cb(coef), rtol=1e-3, atol=1e-5
        )
    # The baseline at Z = 0 of this frame is the reference's moved from its
    # own Z = 0, which is Z = s here. (Moved with the model's own
    # coefficients: an optimiser-tolerance difference in beta is
    # multiplied by the offset in the reported baseline.)
    entry = ORIGIN_MAPS[(model.kind, model.distribution.name)]
    moved = Centring(s, k, entry[1]).from_origin(model.params)
    np.testing.assert_allclose(
        np.asarray(moved, float)[:k], ref.params[:k], rtol=2e-4
    )


@pytest.mark.parametrize("name", ["WeibullPH", "WeibullAFT", "LogNormalAFT"])
def test_the_reported_baseline_at_zero_across_offsets(name):
    # A smaller effect keeps the baseline at 0 representable at 2000 as
    # well (beta'center about 400): the model, and so the reported
    # parameters moved back to one origin, are the same at 0, 300 and 2000.
    x, Z, c = _data(effect=0.2)
    fitter = getattr(sp, name)
    ref = fitter.fit(x, Z, c=c)
    for offset in (300.0, 2000.0):
        s = _shift(offset)
        model = _fit_quietly(fitter.fit, x, Z + s, c=c)
        entry = ORIGIN_MAPS[(model.kind, model.distribution.name)]
        moved = Centring(s, model.k_dist, entry[1]).from_origin(model.params)
        np.testing.assert_allclose(moved, ref.params, rtol=2e-4, atol=1e-6)
        _same_predictions(model, ref, s, cb=False)


@pytest.mark.parametrize("name", MAPPED)
@pytest.mark.parametrize("offset", [2000.0, 1e5])
def test_by_default_a_baseline_that_cannot_be_represented_is_refused(
    name, offset
):
    x, Z, c = _data()
    with pytest.raises(ValueError, match="center=True"):
        getattr(sp, name).fit(x, Z + _shift(offset), c=c)


@pytest.mark.parametrize(
    "name, offset",
    [
        ("LogNormalPH", 300.0),
        ("GammaPH", 2000.0),
        ("NormalPH", 30.0),
        ("WeibullPO", 300.0),
        ("GammaPO", 2000.0),
        ("ExponentialPO", 1e5),
    ],
)
def test_by_default_a_family_without_a_map_refuses_what_breaks_it(
    name, offset
):
    # At Z = 0 these fits used to return a coefficient near 0 (LogNormal PH
    # at 300: 0.006 against 0.64 at the means), a log-likelihood of +3914
    # (Gamma PO at 2000), or a non-converged point with a warning.
    x, Z, c = _data()
    with pytest.raises(ValueError, match="center=True"):
        getattr(sp, name).fit(x, Z + _shift(offset), c=c)


def test_a_family_without_a_map_is_fitted_as_before_near_zero():
    # The default fit of a family without a map is at Z = 0 as it always
    # was (0.21.0's values), and not centred.
    x, Z, c = _data()
    before = {
        "WeibullPO": [11.875752, 1.46471, -0.906161, -1.141153],
        "LogNormalPH": [2.095023, 1.036038, 0.64372, 0.738699],
    }
    for name, params in before.items():
        model = getattr(sp, name).fit(x, Z, c=c)
        assert model._fit_centring is None
        np.testing.assert_array_equal(model.center, [0.0, 0.0])
        np.testing.assert_allclose(model.params, params, rtol=2e-4)


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


def test_the_baseline_at_the_means():
    x, Z, c = _data()
    n = np.random.default_rng(0).integers(1, 4, x.size)
    far = sp.WeibullPH.fit(x, Z + _shift(2000.0), c=c, n=n, center=True)
    # Kept at the n-weighted covariate means.
    np.testing.assert_allclose(
        far.center, np.average(Z + _shift(2000.0), axis=0, weights=n)
    )
    np.testing.assert_allclose(
        far.Hf(TIMES, far.center),
        sp.Weibull.Hf(TIMES, *far.params[:2]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(far.phi(far.center), 1.0)
    assert "Baseline at         : the covariate means" in repr(far)
    assert "Baseline at" not in repr(sp.WeibullPH.fit(x, Z, c=c))


def test_bounds_match_those_of_the_uncentred_computation():
    # The covariance of a model reported at Z = 0 is the centred fit's,
    # carried over by the jacobian of the map; cb is computed in the
    # centred parameterisation. Both agree with the old computation (the
    # Hessian at the reported parameters, on the data as given) where
    # that is well conditioned.
    x, Z, c = _data()
    for name in MAPPED:
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


@pytest.mark.parametrize("name", ["WeibullPH", "LogNormalPH", "WeibullAH"])
def test_save_and_load_a_baseline_at_the_means(name):
    x, Z, c = _data()
    s = _shift(1e5)
    model = getattr(sp, name).fit(x, Z + s, c=c, center=True)
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
    d = sp.WeibullPH.fit(x, Z + _shift(1e5), c=c, center=True).to_dict()
    d["center"] = [1.0]
    with pytest.raises(ValueError, match="'center' has 1 value"):
        sp.from_dict(d)


def test_random_draws_at_the_rows_given():
    x, Z, c = _data()
    s = _shift(1e5)
    model = sp.WeibullPH.fit(x, Z + s, c=c, center=True)
    ref = sp.WeibullPH.fit(x, Z, c=c)
    draws, rows = model.random(4, QUERY + s, random_state=0)
    ref_draws, ref_rows = ref.random(4, QUERY, random_state=0)
    np.testing.assert_allclose(rows, ref_rows + s)
    np.testing.assert_allclose(draws, ref_draws, rtol=1e-4)


def test_an_init_is_read_where_the_baseline_is_reported():
    x, Z, c = _data()
    s = _shift(100.0)
    # At Z = 0 by default (moved to the means for the search) ...
    ref = sp.WeibullPH.fit(x, Z + s, c=c)
    model = _fit_quietly(sp.WeibullPH.fit, x, Z + s, c=c, init=ref.params)
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-6)
    # ... and at the means with center=True.
    ref = sp.WeibullPH.fit(x, Z + s, c=c, center=True)
    model = _fit_quietly(
        sp.WeibullPH.fit, x, Z + s, c=c, init=ref.params, center=True
    )
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-6)


def test_fixing_a_parameter_the_map_moves():
    # alpha fixed at Z = 0 is not a fixed alpha at the means, so that fit
    # runs on the covariates as given; fixing the shape, which the map
    # leaves alone, does not stop the centring. With center=True a fixed
    # value is at the means.
    x, Z, c = _data()
    s = _shift(3.0)
    fixed_scale = sp.WeibullPH.fit(x, Z + s, c=c, fixed={"alpha": 12.0})
    assert fixed_scale._fit_centring is None
    assert fixed_scale.params[0] == 12.0
    fixed_shape = _fit_quietly(
        sp.WeibullPH.fit, x, Z + _shift(300.0), c=c, fixed={"beta": 1.3}
    )
    assert fixed_shape._fit_centring is not None
    ref = sp.WeibullPH.fit(x, Z, c=c, fixed={"beta": 1.3})
    np.testing.assert_allclose(fixed_shape.params[1:], ref.params[1:])
    np.testing.assert_allclose(
        fixed_shape.sf(TIMES, QUERY[0] + _shift(300.0)),
        ref.sf(TIMES, QUERY[0]),
        rtol=1e-6,
    )
    at_means = sp.WeibullPH.fit(
        x, Z + _shift(2000.0), c=c, fixed={"alpha": 9.0}, center=True
    )
    assert at_means.params[0] == 9.0
    np.testing.assert_allclose(
        at_means.Hf(TIMES, at_means.center),
        (TIMES / 9.0) ** at_means.params[1],
    )


def test_a_custom_phi_is_centred_only_on_request():
    import autograd.numpy as anp

    x, Z, c = _data()
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
    ref = fitter.fit(x, Z, c=c, center=True)
    far = _fit_quietly(fitter.fit, x, Z + _shift(1e5), c=c, center=True)
    np.testing.assert_allclose(far.params, ref.params, rtol=1e-6)


def test_fit_from_df_passes_center():
    x, Z, c = _data()
    df = pd.DataFrame({"x": x, "c": c, "z": Z[:, 0] + 1e5, "g": Z[:, 1]})
    model = sp.WeibullPH.fit_from_df(
        df, x_col="x", Z_cols=["z", "g"], c_col="c", center=True
    )
    ref = sp.WeibullPH.fit(x, Z + _shift(1e5), c=c, center=True)
    np.testing.assert_allclose(model.params, ref.params)
    np.testing.assert_allclose(
        model.sf(TIMES, df[["z", "g"]].head(1)),
        ref.sf(TIMES, (Z + _shift(1e5))[:1]),
    )
    # The accelerated life models have no linear predictor to centre.
    life = sp.AcceleratedLife(sp.Weibull, sp.Power)
    with pytest.raises(ValueError, match="center=True is not available"):
        life.fit_from_df(df.assign(z=df["z"] - 9e4), "x", "z", center=True)


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
    ref = fitter.fit_tvc(i, xl, xr, c, Z, center=True)
    model = _fit_quietly(fitter.fit_tvc, i, xl, xr, c, Z + offset, center=True)
    np.testing.assert_allclose(model.params, ref.params, rtol=1e-4)
    path = StepSchedule.from_changepoints([0, 1], [[0.0], [1.0]])
    moved = StepSchedule.from_changepoints([0, 1], [[offset], [offset + 1.0]])
    t = [0.5, 1.0, 2.0, 4.0]
    np.testing.assert_allclose(
        model.sf_tvc(t, moved), ref.sf_tvc(t, path), rtol=1e-4
    )
    np.testing.assert_allclose(
        model.cb(t, [[offset + 1.0]]), ref.cb(t, [[1.0]]), rtol=2e-3
    )
    if name != "WeibullPO":
        # By default the baseline at 0 is exp(1000s) away: refused.
        with pytest.raises(ValueError, match="center=True"):
            fitter.fit_tvc(i, xl, xr, c, Z + offset)
