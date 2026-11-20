"""The predictions of ``surpyval.torch``: ``sf`` against SurPyval's own
models row by row, and ``to_surpyval`` round-tripping it.

Everything is in double precision on the CPU; the references are
SurPyval's own models, so nothing depends on the platform. These run
only with ``--run-torch`` (and PyTorch installed): in CI, the ``torch``
job.
"""

import numpy as np
import pytest

import surpyval as sp

torch = pytest.importorskip("torch")

import surpyval.torch as spt  # noqa: E402
from surpyval.tests.torch.test_torch_loss import PARAMS  # noqa: E402
from surpyval.tests.torch.test_torch_mixture import MIX, W  # noqa: E402

F64 = torch.float64
INF = np.inf


def _dist(name):
    return getattr(sp, name)


def _times(dist):
    """Times over the support and past both its ends, with some in the
    far upper tail."""
    if dist.support[0] == 0:
        return np.array([-1.0, 0.0, 1e-3, 1.0, 5.0, 10.0, 25.0, 60.0, INF])
    return np.array([-INF, -20.0, 0.0, 5.0, 11.0, 30.0, 60.0, INF])


def _per_row(name, rows=4):
    """Each family's parameters with the first varying by row."""
    dist = _dist(name)
    base = PARAMS[name]
    params = {
        p: torch.tensor(v, dtype=F64)
        for p, v in zip(dist.parameter_names, base)
    }
    first = base[0] * np.linspace(0.7, 1.3, rows)
    params[dist.parameter_names[0]] = torch.tensor(first, dtype=F64)
    values = np.tile(base, (rows, 1))
    values[:, 0] = first
    return params, values


def _per_row_mixture(name, rows=4):
    """Each family's mixture with the first parameter of every component
    and the weights varying by row."""
    dist = _dist(name)
    base = np.asarray(MIX[name], dtype=float)
    values = np.repeat(base[None], rows, axis=0)
    values[:, :, 0] *= np.linspace(0.7, 1.3, rows)[:, None]
    weights = np.array([np.roll(W, i) for i in range(rows)])
    params = {
        p: torch.tensor(values[:, :, j], dtype=F64)
        for j, p in enumerate(dist.parameter_names)
    }
    return params, values, torch.tensor(np.log(weights), dtype=F64), weights


@pytest.mark.parametrize("name", list(PARAMS))
def test_sf_is_each_rows_own_model(name):
    dist = _dist(name)
    params, values = _per_row(name)
    times = _times(dist)
    got = spt.sf(dist, times, params)
    assert got.shape == (len(values), len(times))
    assert got.dtype == F64
    for i, row in enumerate(values):
        expected = dist.from_params(row).sf(times)
        np.testing.assert_allclose(
            got[i].numpy(), expected, rtol=1e-12, atol=0
        )


@pytest.mark.parametrize("name", list(MIX))
def test_a_mixtures_sf_is_each_rows_mixture_model(name):
    dist = _dist(name)
    params, values, logits, weights = _per_row_mixture(name)
    times = _times(dist)
    got = spt.sf(dist, times, params, mixture=logits)
    assert got.shape == (len(values), len(times))
    for i in range(len(values)):
        model = sp.MixtureModel.from_params(dist, values[i], weights[i])
        np.testing.assert_allclose(
            got[i].numpy(), model.sf(times), rtol=1e-12, atol=0
        )


def test_shared_parameters_give_one_row():
    params = {"alpha": 10.0, "beta": torch.tensor(2.0, dtype=F64)}
    got = spt.sf(sp.Weibull, [5.0, 10.0], params)
    assert got.shape == (1, 2)
    np.testing.assert_allclose(
        got[0].numpy(), sp.Weibull.from_params([10.0, 2.0]).sf([5.0, 10.0])
    )
    assert spt.sf(sp.Weibull, 10.0, params).shape == (1, 1)


@pytest.mark.parametrize("name", list(PARAMS))
def test_sf_has_a_finite_gradient_everywhere(name):
    # Including at the support's edges and past them, and in tails that
    # round to 0.
    dist = _dist(name)
    params, _ = _per_row(name)
    leaves = {p: v.clone().requires_grad_(True) for p, v in params.items()}
    times = np.r_[_times(dist), 1e4]
    spt.sf(dist, times, leaves).sum().backward()
    for value in leaves.values():
        assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize("name", ["Weibull", "Gamma", "LogNormal", "Gumbel"])
def test_sf_gradient_matches_finite_differences(name):
    dist = _dist(name)
    params, values, logits, _ = _per_row_mixture(name, rows=2)
    times = torch.tensor(
        [t for t in _times(dist) if np.isfinite(t)], dtype=F64
    )
    leaves = [v.clone().requires_grad_(True) for v in params.values()]
    names = list(params)
    logit_leaf = logits.clone().requires_grad_(True)

    def survival(*args):
        return spt.sf(
            dist, times, dict(zip(names, args[:-1])), mixture=args[-1]
        )

    # (first order only for the Gamma, whose shape's derivative is
    # numerical: gradcheck compares first derivatives)
    assert torch.autograd.gradcheck(
        survival, (*leaves, logit_leaf), eps=1e-6, atol=1e-6, rtol=1e-5
    )


def test_a_component_whose_tail_rounds_to_0_leaves_the_gradient_finite():
    # Q(2, 1500) is 0 in double precision.
    params = {
        "alpha": torch.tensor([2.0, 2.5], dtype=F64, requires_grad=True),
        "beta": torch.tensor([5.0, 0.05], dtype=F64, requires_grad=True),
    }
    logits = torch.zeros(2, dtype=F64, requires_grad=True)
    got = spt.sf(sp.Gamma, [300.0], params, mixture=logits)
    got.sum().backward()
    expected = 0.5 * sp.Gamma.from_params([2.5, 0.05]).sf(300.0)
    assert float(got.detach()) == pytest.approx(float(expected), rel=1e-12)
    for value in [*params.values(), logits]:
        assert torch.isfinite(value.grad).all()


def test_sf_follows_the_parameters_type():
    params = {
        "alpha": torch.tensor([10.0, 20.0], dtype=torch.float32),
        "beta": torch.tensor(2.0, dtype=torch.float32),
    }
    got = spt.sf(sp.Weibull, np.array([5.0, 10.0]), params)
    assert got.dtype == torch.float32


def test_sf_refuses_times_it_cannot_take():
    params = {"alpha": 10.0, "beta": 2.0}
    with pytest.raises(ValueError, match="times has NaN"):
        spt.sf(sp.Weibull, [1.0, np.nan], params)
    with pytest.raises(ValueError, match=r"times has shape \(2, 2\)"):
        spt.sf(sp.Weibull, [[1.0, 2.0], [3.0, 4.0]], params)
    with pytest.raises(TypeError, match="times must be numbers"):
        spt.sf(sp.Weibull, ["soon"], params)
    with pytest.raises(ValueError, match="different numbers of rows"):
        spt.sf(
            sp.Weibull,
            [1.0],
            {"alpha": torch.ones(3), "beta": torch.ones(2)},
        )
    with pytest.raises(ValueError, match="has no torch likelihood"):
        spt.sf(sp.Beta, [0.5], {"alpha": 1.0, "beta": 1.0})


@pytest.mark.parametrize("name", list(PARAMS))
def test_to_surpyval_round_trips_sf(name):
    dist = _dist(name)
    params, values = _per_row(name)
    times = _times(dist)
    curves = spt.sf(dist, times, params).numpy()
    for i in range(len(values)):
        model = spt.to_surpyval(dist, params, row=i)
        assert type(model) is type(dist.from_params(values[i]))
        np.testing.assert_array_equal(model.params, values[i])
        np.testing.assert_allclose(model.sf(times), curves[i], rtol=1e-12)
        np.testing.assert_allclose(
            model.ff(times), 1.0 - curves[i], rtol=1e-12, atol=1e-15
        )
        inside = (curves[i] > 1e-6) & (curves[i] < 1.0 - 1e-6)
        # (relative to the times' scale, 10: one of them is 0)
        np.testing.assert_allclose(
            model.qf(1.0 - curves[i][inside]),
            times[inside],
            rtol=1e-8,
            atol=1e-7,
        )


@pytest.mark.parametrize("name", list(MIX))
def test_to_surpyval_round_trips_a_mixtures_sf(name):
    dist = _dist(name)
    params, values, logits, weights = _per_row_mixture(name)
    times = _times(dist)
    curves = spt.sf(dist, times, params, mixture=logits).numpy()
    for i in range(len(values)):
        model = spt.to_surpyval(dist, params, row=i, mixture=logits)
        assert isinstance(model, sp.MixtureModel)
        assert model.m == 3 and model.dist is dist
        np.testing.assert_allclose(model.params, values[i], rtol=0, atol=0)
        np.testing.assert_allclose(model.w, weights[i], rtol=1e-15)
        np.testing.assert_allclose(model.sf(times), curves[i], rtol=1e-12)
        np.testing.assert_allclose(
            model.ff(times), 1.0 - curves[i], rtol=1e-12, atol=1e-15
        )
        inside = (curves[i] > 1e-6) & (curves[i] < 1.0 - 1e-6)
        # (relative to the times' scale, 10: one of them is 0)
        np.testing.assert_allclose(
            model.qf(1.0 - curves[i][inside]),
            times[inside],
            rtol=1e-8,
            atol=1e-7,
        )


def test_to_surpyval_takes_rows_from_the_end_and_float32():
    params = {
        "alpha": torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32),
        "beta": torch.tensor(2.5, dtype=torch.float32),
    }
    model = spt.to_surpyval(sp.Weibull, params, row=-1)
    assert model.params.dtype == np.float64
    np.testing.assert_array_equal(model.params, [30.0, 2.5])
    one = spt.to_surpyval(sp.Weibull, {"alpha": 10.0, "beta": 2.0})
    np.testing.assert_array_equal(one.params, [10.0, 2.0])


def test_to_surpyval_detaches_from_the_graph():
    raw = torch.zeros(2, 2, dtype=F64, requires_grad=True)
    model = spt.to_surpyval(sp.Weibull, spt.constrain(sp.Weibull, raw), row=0)
    np.testing.assert_allclose(model.params, [np.log(2.0)] * 2, rtol=1e-15)


def test_to_surpyval_refuses_a_row_out_of_range_naming_the_rows():
    params = {"alpha": torch.ones(3, dtype=F64), "beta": 2.0}
    with pytest.raises(ValueError) as err:
        spt.to_surpyval(sp.Weibull, params, row=3)
    assert "0 to 2" in str(err.value)
    assert "have 3 rows" in str(err.value)
    with pytest.raises(ValueError, match="have 3 rows"):
        spt.to_surpyval(sp.Weibull, params, row=-4)
    with pytest.raises(ValueError, match="row must be a whole number"):
        spt.to_surpyval(sp.Weibull, params, row=True)
    with pytest.raises(ValueError, match="row must be a whole number"):
        spt.to_surpyval(sp.Weibull, params, row=1.0)
    with pytest.raises(ValueError, match=r"3 rows; say which one .* row="):
        spt.to_surpyval(sp.Weibull, params)
    logits = torch.zeros(5, 2, dtype=F64)
    with pytest.raises(ValueError, match="have 5 rows"):
        spt.to_surpyval(
            sp.Weibull, {"alpha": 1.0, "beta": 1.0}, row=5, mixture=logits
        )


def test_to_surpyval_refuses_a_fitted_model_and_an_unsupported_family():
    with pytest.raises(TypeError, match="pass model.dist"):
        spt.to_surpyval(
            sp.Weibull.from_params([1.0, 1.0]), {"alpha": 1.0, "beta": 1.0}
        )
    with pytest.raises(ValueError, match="has no torch likelihood"):
        spt.to_surpyval(sp.Beta, {"alpha": 1.0, "beta": 1.0})
