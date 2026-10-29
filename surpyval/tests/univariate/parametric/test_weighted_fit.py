"""Direct checks of a parametric fit with sampling weights ``w``.

The conformance suite (``conformance/test_weights.py``) checks the
properties every weighted fit keeps; these check the numbers themselves:
the robust (sandwich) covariance against one built by hand, the round trip
of a limited failure population fit, and the weighted plotting positions.
"""

import json
import warnings

import numdifftools as nd  # type: ignore
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import surpyval as sp
from surpyval.tests._helpers import quietly, weighted_reference_data
from surpyval.utils.weights import WeightedRefusal


def _hand_sandwich(x, c, w, tl, params):
    """``H^-1 M H^-1`` from the Weibull's own log density and log
    survival: ``l_i = log df(x_i)`` for a failure, ``log sf(x_i)`` for a
    right censored row, less ``log sf(tl_i)`` for an entry at ``tl_i``;
    ``H`` the negative Hessian of ``sum_i w_i l_i`` (the weights as given)
    and ``M = sum_i w_i^2 s_i s_i'``, ``s_i`` the gradient of ``l_i``."""

    def row(i, theta):
        alpha, beta = theta
        if c[i] == 0:
            value = sp.Weibull.log_df(x[i], alpha, beta)
        else:
            value = sp.Weibull.log_sf(x[i], alpha, beta)
        if tl is not None and tl[i] > 0:
            value = value - sp.Weibull.log_sf(tl[i], alpha, beta)
        return float(value)

    theta = np.asarray(params, dtype=float)
    rows = range(len(x))

    def total(th):
        return sum(w[i] * row(i, th) for i in rows)

    H = -nd.Hessian(total)(theta)
    scores = np.array(
        [nd.Gradient(lambda th: row(i, th))(theta) for i in rows]
    )
    M = (scores * (w**2)[:, None]).T @ scores
    H_inv = np.linalg.inv(H)
    return H_inv @ M @ H_inv


@pytest.mark.parametrize("truncated", [False, True], ids=["no tl", "tl"])
def test_the_sandwich_matches_one_built_by_hand(truncated):
    """audit W-C: the robust covariance of a weighted Weibull fit is
    ``H^-1 M H^-1`` built independently with numdifftools from the
    distribution's log density and log survival. It is free of the scale of
    the weights, so the library's own normalisation of them must not
    matter; a constant factor between the two would be a bug."""
    data = weighted_reference_data()
    x, c, w = data["x"], data["c"], data["w"]
    tl = data["tl"] if truncated else None
    model = sp.Weibull.fit(x, c, w=w, tl=tl)
    expected = _hand_sandwich(x, c, w, tl, model.params)
    assert_allclose(model.covariance(), expected, rtol=1e-4)


def test_the_round_trip_of_a_weighted_lfp_fit_keeps_everything():
    """audit W-B: a weighted limited failure population fit saved and
    restored keeps ``lfp_p``, ``weighted``, ``effective_n``, the 3x3 robust
    covariance and ``cb(sf)`` exactly."""
    data = weighted_reference_data()
    x, c, w, tl = data["x"], data["c"], data["w"], data["tl"]
    model = sp.Weibull.fit(
        np.minimum(x, 8.0), np.where(x > 8.0, 1, c), w=w, tl=tl, lfp=True
    )
    restored = sp.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.weighted is True
    assert restored.lfp_p == model.lfp_p
    assert restored.effective_n == model.effective_n
    assert model.covariance().shape == (3, 3)
    assert_array_equal(restored.covariance(), model.covariance())
    query = np.array([1.5, 3.0, 6.0])
    assert_array_equal(restored.cb(query, on="sf"), model.cb(query, on="sf"))


QUERY = np.array([1.5, 3.0, 6.0])


def _weighted_km(**kwargs):
    data = weighted_reference_data()
    return quietly(
        sp.KaplanMeier.fit, data["x"], data["c"], w=data["w"], **kwargs
    )


def test_a_lower_limit_prepends_a_certain_step():
    """audit W-F: ``set_lower_limit=0.0`` on a weighted Kaplan-Meier
    prepends a step at 0 with R = 1 and variance 0, and leaves the rest of
    the estimate as it was."""
    plain, limited = _weighted_km(), _weighted_km(set_lower_limit=0.0)
    assert limited.weighted
    assert limited.x[0] == 0.0 and limited.R[0] == 1.0
    assert limited.greenwood[0] == 0.0
    assert_array_equal(limited.x[1:], plain.x)
    assert_allclose(limited.R[1:], plain.R, rtol=1e-12)
    assert_allclose(limited.greenwood[1:], plain.greenwood, rtol=1e-12)
    assert_allclose(limited.cb(QUERY), plain.cb(QUERY), rtol=1e-12)


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_interpolated_bounds_contain_the_interpolated_estimate(interp):
    """audit W-F: a weighted Kaplan-Meier's interpolated bounds are finite
    and contain its interpolated ``sf``."""
    model = _weighted_km()
    bounds = model.cb(QUERY, interp=interp)
    sf = model.sf(QUERY, interp=interp)
    assert np.isfinite(bounds).all()
    assert (bounds[:, 0] <= sf).all() and (sf <= bounds[:, 1]).all()


def test_conditional_survival_is_the_ratio_of_weighted_sfs():
    """audit W-F: ``cs(x, given)`` of a weighted Kaplan-Meier is
    ``sf(given + x) / sf(given)``."""
    model = _weighted_km()
    expected = model.sf(np.array([2.5, 3.5])) / model.sf(1.5)
    assert_allclose(model.cs([1.0, 2.0], 1.5), expected, rtol=1e-12)


@pytest.mark.parametrize(
    "heuristic, estimator",
    [
        (None, sp.NelsonAalen),  # the default
        ("Kaplan-Meier", sp.KaplanMeier),
        ("Nelson-Aalen", sp.NelsonAalen),
        ("Fleming-Harrington", sp.FlemingHarrington),
    ],
)
def test_weighted_plotting_positions_are_the_weighted_estimates(
    heuristic, estimator
):
    """audit W-P: the plotting positions of a weighted fit are ``1 - S``
    of the weighted non-parametric estimate of the same name, at every
    time of its ladder (the default heuristic is Nelson-Aalen's)."""
    data = weighted_reference_data()
    x, c, w = data["x"], data["c"], data["w"]
    model = quietly(sp.Weibull.fit, x, c, w=w)
    kwargs = {} if heuristic is None else {"heuristic": heuristic}
    plot = model.get_plot_data(**kwargs)
    assert_array_equal(plot["x_"], np.unique(x))
    estimate = quietly(estimator.fit, x, c, w=w)
    assert_allclose(plot["F"], 1 - estimate.sf(plot["x_"]), rtol=1e-12)


def test_weighted_interval_censored_data_refuse_a_plot():
    """audit W-P: a weighted fit of interval censored data has no weighted
    (Turnbull) plotting positions yet, so ``get_plot_data`` and ``plot``
    refuse, naming Turnbull."""
    data = weighted_reference_data()
    x, w = data["x"], data["w"]
    model = quietly(sp.Weibull.fit, xl=x, xr=x + 1, c=np.full(len(x), 2), w=w)
    for call in (model.get_plot_data, model.plot):
        with pytest.raises(WeightedRefusal, match="Turnbull"):
            # (plot first says the heuristic changed to Turnbull)
            quietly(call)


def test_a_weighted_plot_draws_without_warnings():
    """audit W-P: ``plot()`` of a weighted right censored fit draws with
    no warning at all."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = weighted_reference_data()
    model = quietly(sp.Weibull.fit, data["x"], data["c"], w=data["w"])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ax = model.plot()
    assert ax is not None
    plt.close("all")
