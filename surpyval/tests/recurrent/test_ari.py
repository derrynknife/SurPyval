import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from surpyval.recurrent import ARI, CrowAMSAA, Duane  # noqa: E402
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
    model = ARI.fit(X, I, m=1, dist=CrowAMSAA)
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
    model = ARI.fit_from_parameters([60.0, 2.0], rho=0.3, m=1, dist=CrowAMSAA)
    mcf = model.mcf(np.array([5.0, 10.0, 20.0, 30.0]), items=800, seed=0)
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
    model = ARI.fit_from_parameters([60.0, 2.0], rho=0.3, m=1, dist=CrowAMSAA)
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
    truth = ARI.fit_from_parameters([20.0, 1.5], 0.5, m=m, dist=CrowAMSAA)
    data = truth.count_terminated_simulation_data(6, items=12, seed=1)
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
    truth = ARI.fit_from_parameters([20.0, 1.5], 0.5, m=1, dist=CrowAMSAA)
    data = truth.count_terminated_simulation_data(6, items=10, seed=2)
    negll = ARI.create_negll_func(data, CrowAMSAA, 1)
    got = negll(np.array(params))
    assert np.isinf(got) and got > 0
    # ... and the original agrees it is out of support.
    assert np.isinf(_scalar_negll(data, CrowAMSAA, 1, np.array(params)))


def test_fit_scales_to_many_items():
    # 250 items took 19 seconds under the per-event loop.
    truth = ARI.fit_from_parameters([20.0, 1.5], 0.5, m=1, dist=CrowAMSAA)
    data = truth.count_terminated_simulation_data(8, items=250, seed=5)
    model = ARI.fit_from_recurrent_data(data, dist=CrowAMSAA, m=1)
    assert 0.0 <= model.rho <= 1.0
    assert np.isfinite(model.model.params).all()
