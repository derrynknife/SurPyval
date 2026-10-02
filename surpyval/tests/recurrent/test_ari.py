import warnings

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import surpyval as sp  # noqa: E402
from surpyval.recurrent import (  # noqa: E402
    ARA,
    ARI,
    CoxLewis,
    CrowAMSAA,
    Duane,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
)
from surpyval.recurrent.renewal.ari import ari_reduction  # noqa: E402
from surpyval.utils.recurrent_utils import handle_xicn  # noqa: E402

X = np.array([3, 9, 20, 35, 56, 4, 11, 25, 44, 70], dtype=float)
I = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2])


def test_ari_reduction_helper():
    lams = [0.2, 0.5, 0.9]
    rho = 0.4
    # m = 1 keeps only the most recent failure intensity.
    assert np.isclose(ari_reduction(lams, rho, 1), rho * 0.9)
    # m = 2 keeps the last two.
    assert np.isclose(
        ari_reduction(lams, rho, 2), rho * (0.9 + (1 - rho) * 0.5)
    )
    # m = inf keeps the whole memory-weighted history.
    assert np.isclose(
        ari_reduction(lams, rho, np.inf),
        rho * (0.9 + (1 - rho) * 0.5 + (1 - rho) ** 2 * 0.2),
    )
    assert ari_reduction([], rho, 1) == 0.0


@pytest.mark.parametrize("dist", [CrowAMSAA, Duane])
def test_ari_rho_zero_matches_nhpp(dist):
    # With rho = 0 there is no intensity reduction, so the ARI log-likelihood
    # must equal the plain NHPP log-likelihood of the baseline intensity.
    data = handle_xicn(X, I, as_recurrent_data=True)
    ari_negll = ARI.create_negll_func(data, dist, m=1)
    nhpp_negll = dist.create_negll_func(data)
    for params in ([100.0, 1.3], [50.0, 1.4]):
        a = ari_negll([0.0, *params])
        b = nhpp_negll(params)
        assert np.isfinite(a) and np.isclose(a, b)


def test_ari_fit_and_information_criteria():
    model = ARI.fit(X, I, m=1, baseline=CrowAMSAA)
    assert 0.0 <= model.rho <= 1.0
    k = model._mle.size
    n = model._n_obs
    ll = model.log_likelihood
    assert np.isclose(ll, -model.res.fun)
    assert np.isclose(model.aic, 2 * k - 2 * ll)
    assert np.isclose(model.bic, k * np.log(n) - 2 * ll)
    assert model.parameter_names == ["rho", "alpha", "beta"]
    assert "ARI" in repr(model)


def test_ari_mcf_simulation_monotonic():
    model = ARI.fit_from_parameters(
        [60.0, 2.0], rho=0.3, m=1, baseline=CrowAMSAA
    )
    mcf = model.mcf(
        np.array([5.0, 10.0, 20.0, 30.0]), items=800, random_state=0
    )
    assert np.all(np.diff(mcf) >= -1e-9)
    assert np.all(mcf >= 0)


def test_ari_validates_memory():
    for bad in (0, -1, 2.5):
        with pytest.raises(ValueError, match="positive integer"):
            ARI.fit(X, I, m=bad)


def test_ari_rejects_unsupported_censoring():
    c = np.zeros_like(I)
    c[-1] = 2  # interval censoring not supported
    with pytest.raises(ValueError, match="censoring code"):
        ARI.fit(X, I, c=c, m=1)


def test_ari_inference_requires_fit_from_data():
    model = ARI.fit_from_parameters(
        [60.0, 2.0], rho=0.3, m=1, baseline=CrowAMSAA
    )
    with pytest.raises(ValueError, match="fitted from data"):
        model.aic


# -- the vectorised likelihood against the scalar original -------------
#
# The likelihood used to walk every event in Python, calling the
# baseline cif/iif and rebuilding the reduction from the failure history
# each step. That was ~10ms per event and made a 250-item fit take 19
# seconds. The replacement evaluates the whole sample at once, summing
# over the reduction *window offset* instead of over the events. The
# original is kept here as the oracle so the two cannot drift apart.


def _scalar_negll(data, dist, m, params):
    """The original per-event implementation, verbatim."""
    _, idx = np.unique(data.i, return_index=True)
    x_by_item = np.split(data.x, idx)[1:]
    c_by_item = np.split(data.c, idx)[1:]

    rho = params[0]
    dist_params = params[1:]

    ll = 0.0
    for x_item, c_item in zip(x_by_item, c_by_item):
        prev = 0.0
        reduction = 0.0
        history_iif = []
        for t, censor in zip(x_item, c_item):
            delta_cif = dist.cif(t, *dist_params) - dist.cif(
                prev, *dist_params
            )
            ll -= delta_cif - reduction * (t - prev)
            if censor == 0:
                intensity = dist.iif(t, *dist_params) - reduction
                if intensity <= 0:
                    return np.inf
                ll += np.log(intensity)
                history_iif.append(dist.iif(t, *dist_params))
                reduction = ari_reduction(history_iif, rho, m)
            prev = t
    return -ll


PARAM_GRID = [
    [0.0, 20.0, 1.5],
    [0.1, 20.0, 1.5],
    [0.5, 20.0, 1.5],
    [0.9, 20.0, 1.5],
    [0.999, 20.0, 1.5],
    [0.5, 5.0, 0.7],
    [0.3, 50.0, 2.5],
]


@pytest.mark.parametrize("m", [1, 2, 3, np.inf])
def test_vectorised_negll_matches_the_scalar_original(m):
    truth = ARI.fit_from_parameters([20.0, 1.5], 0.5, m=m, baseline=CrowAMSAA)
    data = truth.count_terminated_simulation_data(6, items=12, random_state=1)
    negll = ARI.create_negll_func(data, CrowAMSAA, m)
    for params in PARAM_GRID:
        want = _scalar_negll(data, CrowAMSAA, m, np.array(params))
        got = negll(np.array(params))
        if np.isinf(want):
            assert np.isinf(got) and np.sign(want) == np.sign(got)
        else:
            assert got == pytest.approx(want, rel=1e-12)


def test_vectorised_negll_matches_with_censoring_and_uneven_items():
    # Items of different lengths, some ending in a suspension, is where
    # the per-item bookkeeping has to be right: a reduction must not leak
    # from one item into the next, and a suspension consumes an interval
    # without updating the reduction.
    x = np.array([3, 9, 20, 35, 4, 11, 7, 15, 22, 40], dtype=float)
    i = np.array([1, 1, 1, 1, 2, 2, 3, 3, 3, 3])
    c = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1])
    data = handle_xicn(x, i, c)
    for m in (1, 2, np.inf):
        negll = ARI.create_negll_func(data, CrowAMSAA, m)
        for params in PARAM_GRID:
            want = _scalar_negll(data, CrowAMSAA, m, np.array(params))
            got = negll(np.array(params))
            if np.isinf(want):
                assert np.isinf(got)
            else:
                assert got == pytest.approx(want, rel=1e-12)


@pytest.mark.parametrize(
    "params",
    [
        [0.9, 20.0, 0.7],  # decreasing baseline: the reduction overtakes it
        [1.0, 20.0, 1.0],  # flat baseline, full reduction: intensity is 0
    ],
    ids=["overtaken", "exactly zero"],
)
def test_negll_is_infinite_when_the_intensity_is_not_positive(params):
    # A non-positive reduced intensity is outside the model's support.
    # The scalar loop returned early on it; the vectorised form has to
    # test before taking logs rather than warning its way to a nan. Note
    # a rising baseline (beta > 1) stays positive even at rho = 1, so
    # the case has to be built from a flat or falling one.
    truth = ARI.fit_from_parameters([20.0, 1.5], 0.5, m=1, baseline=CrowAMSAA)
    data = truth.count_terminated_simulation_data(6, items=10, random_state=2)
    negll = ARI.create_negll_func(data, CrowAMSAA, 1)
    got = negll(np.array(params))
    assert np.isinf(got) and got > 0
    # ... and the original agrees it is out of support.
    assert np.isinf(_scalar_negll(data, CrowAMSAA, 1, np.array(params)))


def test_fit_scales_to_many_items():
    # 250 items took 19 seconds under the per-event loop.
    truth = ARI.fit_from_parameters([20.0, 1.5], 0.5, m=1, baseline=CrowAMSAA)
    data = truth.count_terminated_simulation_data(8, items=250, random_state=5)
    model = ARI.fit_from_recurrent_data(data, baseline=CrowAMSAA, m=1)
    assert 0.0 <= model.rho <= 1.0
    assert np.isfinite(model.model.params).all()


# -- #495: ARI's dist is an intensity model, ARA's a lifetime distribution --


@pytest.mark.parametrize(
    "dist", [sp.Weibull, sp.Exponential, sp.LogNormal, sp.Gamma]
)
def test_ari_refuses_a_lifetime_distribution_clearly(dist):
    # It failed with AttributeError: 'Weibull_' object has no attribute
    # 'parameter_initialiser', from inside the fit.
    with pytest.raises(ValueError, match="baseline intensity model") as err:
        ARI.fit(X, I, baseline=dist)
    message = str(err.value)
    assert "CrowAMSAA" in message and "ARA" in message
    with pytest.raises(ValueError, match="baseline intensity model"):
        ARI.fit_from_recurrent_data(handle_xicn(X, I), baseline=dist)
    with pytest.raises(ValueError, match="baseline intensity model"):
        ARI.fit_from_parameters([10.0, 2.0], 0.5, baseline=dist)


def test_ari_refuses_anything_else_clearly():
    with pytest.raises(ValueError, match="must be a recurrence intensity"):
        ARI.fit(X, I, baseline="CrowAMSAA")


@pytest.mark.parametrize("fitter", [ARA, GeneralizedRenewal])
def test_lifetime_fitters_refuse_an_intensity_model_clearly(fitter):
    # ARA.fit(x, i, dist=CrowAMSAA) said "Item 0.0 has more than one
    # right censored time".
    with pytest.raises(ValueError, match="not an intensity model.*ARI"):
        fitter.fit(X, I, dist=CrowAMSAA)
    with pytest.raises(ValueError, match="not an intensity model"):
        fitter.fit_from_parameters([10.0, 2.0], 0.5, dist=CrowAMSAA)


def test_g1_refuses_an_intensity_model_clearly():
    with pytest.raises(ValueError, match="not an intensity model"):
        GeneralizedOneRenewal.fit(X, I, dist=Duane)
    with pytest.raises(ValueError, match="not an intensity model"):
        GeneralizedOneRenewal.fit_from_parameters([10.0, 2.0], 0.5, dist=Duane)


@pytest.mark.parametrize("dist", [CrowAMSAA, Duane, CoxLewis])
def test_ari_still_takes_every_intensity_model(dist):
    model = ARI.fit(X, I, baseline=dist)
    assert model.model.dist is dist
    assert 0.0 <= model.rho <= 1.0


# -- #507: ARI's baseline intensity is `baseline=`, not `dist=` --------------


def _old_name(call):
    # The result of ``call`` and the one DeprecationWarning it raised.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = call()
    deprecations = [w for w in caught if w.category is DeprecationWarning]
    assert len(deprecations) == 1, [str(w.message) for w in caught]
    return out, deprecations[0]


@pytest.mark.parametrize(
    "method", ["fit", "fit_from_recurrent_data", "fit_from_df"]
)
def test_ari_takes_its_baseline_as_baseline(method):
    # In ARA, GeneralizedRenewal and GeneralizedOneRenewal `dist` is a
    # lifetime distribution; in ARI it was the baseline intensity model,
    # so `dist=sp.Weibull` was a natural mistake. `dist=` still works,
    # warning, until v0.23, and gives the same fit.
    import pandas as pd

    def call(**kw):
        if method == "fit":
            return ARI.fit(X, I, m=1, **kw)
        if method == "fit_from_recurrent_data":
            return ARI.fit_from_recurrent_data(handle_xicn(X, I), m=1, **kw)
        df = pd.DataFrame({"x": X, "i": I})
        return ARI.fit_from_df(df, x_col="x", i_col="i", m=1, **kw)

    new = call(baseline=Duane)
    assert new.model.dist is Duane
    old, warning = _old_name(lambda: call(dist=Duane))
    message = str(warning.message)
    assert "'dist' is deprecated" in message and "'baseline'" in message
    assert "0.23" in message
    # The warning points at the caller's line, not into SurPyval.
    assert warning.filename == __file__
    np.testing.assert_array_equal(old.params, new.params)
    with pytest.raises(ValueError, match="pass 'baseline' only"):
        call(baseline=Duane, dist=Duane)


def test_ari_fit_from_parameters_takes_baseline():
    new = ARI.fit_from_parameters(
        baseline_params=[20.0, 1.5], rho=0.5, baseline=Duane
    )
    assert new.model.dist is Duane
    old, warning = _old_name(
        lambda: ARI.fit_from_parameters([20.0, 1.5], 0.5, dist=Duane)
    )
    assert "'baseline'" in str(warning.message)
    assert warning.filename == __file__
    old_params, warning = _old_name(
        lambda: ARI.fit_from_parameters(
            dist_params=[20.0, 1.5], rho=0.5, baseline=Duane
        )
    )
    assert "'baseline_params'" in str(warning.message)
    for model in (old, old_params):
        np.testing.assert_array_equal(model.params, new.params)


def test_ari_signatures_name_the_baseline():
    import inspect

    for method in ("fit", "fit_from_recurrent_data", "fit_from_parameters"):
        params = inspect.signature(getattr(ARI, method)).parameters
        assert "baseline" in params and "dist" not in params, method
        assert params["baseline"].default is CrowAMSAA


def test_ari_lifetime_distribution_error_names_baseline():
    with pytest.raises(ValueError, match="`baseline` is the baseline"):
        ARI.fit(X, I, baseline=sp.Weibull)


def test_ari_saved_before_the_rename_still_loads():
    # The layout written before #507 (and still written): the baseline's
    # name under "dist".
    from surpyval.recurrent.renewal.renewal_model import RenewalModel

    saved = {
        "model": "RenewalModel",
        "family": "ARI",
        "dist": "Duane",
        "params": [20.0, 1.5],
        "restoration": 0.5,
        "how": "from_params",
        "m": 1,
        "schema": 1,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        model = RenewalModel.from_dict(dict(saved))
        assert model.to_dict() == saved
    assert model.model.dist is Duane
    np.testing.assert_array_equal(model.params, [0.5, 20.0, 1.5])
