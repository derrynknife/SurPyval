"""The unconstrained-scale maps (``surpyval.bayes.transforms``) that
every grid posterior is laid on: the round trip, the Jacobian, saving,
and what happens on or past a bound."""

import numpy as np
import pytest

from surpyval.bayes.transforms import Transform

# (bounds, kind, values of theta inside the bounds)
CASES = [
    ((0, None), "log", [0.3, 12.0]),
    ((None, None), "identity", [-1.0, 4.0]),
    ((0, 1), "logit", [0.3, 0.7]),
    ((None, 5.0), "neglog", [-1.0, 4.0]),
    ((-2.0, 3.0), "logit", [-1.0, 2.0]),
]
IDS = ["log", "identity", "logit01", "neglog", "logit_ab"]


@pytest.mark.parametrize("bounds, kind, theta", CASES, ids=IDS)
def test_kind_and_scale_name(bounds, kind, theta):
    """audit B20: each kind of bounds gets its map and a scale in words."""
    t = Transform.for_bounds(bounds)
    assert t.kind == kind
    assert isinstance(t.scale_name, str) and t.scale_name


@pytest.mark.parametrize("bounds, kind, theta", CASES, ids=IDS)
def test_to_theta_inverts_to_phi(bounds, kind, theta):
    """audit B20: ``to_theta(to_phi(theta)) == theta`` inside the
    bounds."""
    t = Transform.for_bounds(bounds)
    theta = np.asarray(theta)
    np.testing.assert_allclose(t.to_theta(t.to_phi(theta)), theta, rtol=1e-12)


@pytest.mark.parametrize("bounds, kind, theta", CASES, ids=IDS)
def test_log_jacobian_is_the_log_derivative(bounds, kind, theta):
    """audit B20: ``log_jacobian(phi)`` is ``log |d theta / d phi|``,
    checked by a central difference (the absolute value because the
    upper-bounded map is decreasing)."""
    t = Transform.for_bounds(bounds)
    phi = t.to_phi(np.asarray(theta))
    h = 1e-6
    slope = (t.to_theta(phi + h) - t.to_theta(phi - h)) / (2 * h)
    np.testing.assert_allclose(
        t.log_jacobian(phi), np.log(np.abs(slope)), atol=1e-6
    )


@pytest.mark.parametrize("bounds, kind, theta", CASES, ids=IDS)
def test_to_dict_round_trip(bounds, kind, theta):
    """audit B20: a transform saved and restored is the same map."""
    t = Transform.for_bounds(bounds)
    again = Transform.from_dict(t.to_dict())
    assert again.kind == t.kind
    assert again == t
    phi = t.to_phi(np.asarray(theta))
    np.testing.assert_array_equal(again.to_phi(theta), phi)
    np.testing.assert_array_equal(again.to_theta(phi), t.to_theta(phi))


@pytest.mark.parametrize(
    "bounds",
    [b for b, k, _ in CASES if k != "identity"],
    ids=IDS[:1] + IDS[2:],
)
def test_on_or_past_a_bound_phi_is_not_finite(bounds):
    """audit B20: a theta on a bound maps to an infinite phi and one past
    it to NaN; neither raises (the caller refuses non-finite values)."""
    t = Transform.for_bounds(bounds)
    edges = [b for b in bounds if b is not None]
    past = [edges[0] - 1.0] if bounds[0] is not None else []
    past += [bounds[1] + 1.0] if bounds[1] is not None else []
    on = t.to_phi(edges)
    assert np.all(np.isinf(on))
    assert np.all(np.isnan(t.to_phi(past)))
