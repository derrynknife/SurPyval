"""The torch loss of ``surpyval.torch`` against SurPyval's own likelihood.

At fixed parameters, ``nll`` must equal the distribution's
``_log_likelihood`` (what ``fit`` maximises) for every supported family,
every row type and every truncation pattern, to 1e-10, and its gradient
must agree with a central difference of SurPyval's. Everything is in
double precision on the CPU; the references are SurPyval's own values,
so nothing depends on the platform.

These run only with ``--run-torch`` (and PyTorch installed, which is not
a test dependency of SurPyval): in CI, the ``torch`` job.
"""

import numpy as np
import pytest

import surpyval as sp

torch = pytest.importorskip("torch")

import surpyval.torch as spt  # noqa: E402

INF = np.inf
F64 = torch.float64

# Parameters at which each family puts its mass over the times below
# (between 1 and 30), and where some windows lie in its upper tail.
PARAMS = {
    "Weibull": [10.0, 1.7],
    "Exponential": [0.09],
    "LogNormal": [2.2, 0.6],
    "Gamma": [2.5, 0.25],
    "LogLogistic": [9.0, 2.5],
    "Normal": [11.0, 6.0],
    "Gumbel": [14.0, 5.0],
    "Logistic": [11.0, 3.5],
    "Rayleigh": [9.0],
}

# Each row type alone, then mixed with counts. Interval [22, 27] sits in
# every family's upper tail (F(22) > 1/2), where the window is taken
# from the survival function.
ROWS = {
    "observed": dict(x=[2.0, 5.5, 9.0, 13.0, 21.0]),
    "right": dict(x=[3.0, 8.0, 15.0], c=[1, 1, 1]),
    "left": dict(x=[4.0, 7.5, 12.0], c=[-1, -1, -1]),
    "interval": dict(x=[[2.0, 6.0], [8.0, 9.0], [22.0, 27.0]], c=[2, 2, 2]),
    "mixed": dict(
        x=[2.0, 5.0, [6.0, 10.0], 9.0, 14.0, [22.0, 27.0], 18.0, 4.5],
        c=[0, 1, 2, -1, 0, 2, 1, -1],
        n=[1, 3, 2, 1, 4, 1, 2, 5],
    ),
}

# Truncation patterns, as (tl, tr) for every row.
TRUNCATION = {
    "none": (None, None),
    "left": (1.0, None),
    "right": (None, 29.0),
    "both": (1.0, 29.0),
    "left_at_support_edge": (0.0, 29.0),
}


def _arrays(rows, truncation):
    arrays = {k: list(v) for k, v in ROWS[rows].items()}
    tl, tr = TRUNCATION[truncation]
    k = len(arrays["x"])
    if tl is not None:
        arrays["tl"] = [tl] * k
    if tr is not None:
        arrays["tr"] = [tr] * k
    return arrays


def _surpyval_ll(dist, arrays, params):
    data = sp.SurpyvalData(**arrays)
    return float(dist._log_likelihood(data, *params, 0.0, 0.0, 1.0))


def _dist(name):
    return getattr(sp, name)


def _kw(dist, params, requires_grad=False):
    return {
        name: torch.tensor(value, dtype=F64, requires_grad=requires_grad)
        for name, value in zip(dist.parameter_names, params)
    }


def _nll(dist, arrays, params, **kwargs):
    kwargs.setdefault("reduction", "sum")
    return spt.nll(dist, spt.targets(**arrays), params, **kwargs)


def test_every_supported_family_is_tested():
    assert set(PARAMS) == set(spt.SUPPORTED)


@pytest.mark.parametrize("truncation", list(TRUNCATION))
@pytest.mark.parametrize("rows", list(ROWS))
@pytest.mark.parametrize("name", list(PARAMS))
def test_nll_equals_surpyval(name, rows, truncation):
    dist = _dist(name)
    if truncation == "left_at_support_edge" and dist.support[0] != 0:
        pytest.skip("only a positive family has its support edge at 0")
    params = PARAMS[name]
    arrays = _arrays(rows, truncation)
    expected = -_surpyval_ll(dist, arrays, params)
    got = float(_nll(dist, arrays, _kw(dist, params)))
    assert np.isfinite(expected)
    assert got == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("name", list(PARAMS))
def test_a_surpyval_data_and_the_arrays_give_the_same_targets(name):
    dist = _dist(name)
    params = PARAMS[name]
    arrays = _arrays("mixed", "both")
    # (not grouped or sorted, so its rows are the arrays')
    data = sp.SurpyvalData(**arrays, group_and_sort=False)
    from_data, from_arrays = spt.targets(data), spt.targets(**arrays)
    for a, b in zip(from_data, from_arrays):
        assert torch.equal(a, b)
    got = float(spt.nll(dist, from_data, _kw(dist, params), reduction="sum"))
    assert got == pytest.approx(
        -_surpyval_ll(dist, arrays, params), rel=1e-10, abs=1e-10
    )


def test_targets_are_the_rows_in_order():
    y = spt.targets(
        x=[2.0, 5.0, [6.0, 10.0], 9.0, 7.0, 3.0],
        c=[0, 1, 2, -1, 1, -1],
        n=[1, 3, 2, 1, 1, 4],
        tl=[1.0, -INF, 1.0, -INF, 0.5, 0.5],
        tr=[INF, INF, 29.0, 29.0, 29.0, INF],
    )
    assert isinstance(y, spt.Targets)
    assert y._fields == ("xl", "xr", "c", "n", "tl", "tr")
    # A censored row's open side ends at its truncation bound there.
    assert y.xl.tolist() == [2.0, 5.0, 6.0, -INF, 7.0, 0.5]
    assert y.xr.tolist() == [2.0, INF, 10.0, 9.0, 29.0, 3.0]
    assert y.c.tolist() == [0, 1, 2, -1, 1, -1]
    assert y.n.tolist() == [1.0, 3.0, 2.0, 1.0, 1.0, 4.0]
    assert y.tl.tolist() == [1.0, -INF, 1.0, -INF, 0.5, 0.5]
    assert y.tr.tolist() == [INF, INF, 29.0, 29.0, 29.0, INF]
    assert {v.dtype for v in y} == {F64, torch.int64}
    assert all(v.shape == (6,) for v in y)
    single = spt.targets(x=[1.0, 2.0], dtype=torch.float32)
    assert single.xl.dtype == torch.float32
    assert single.c.dtype == torch.int64


def test_targets_batch_through_a_dataloader():
    arrays = _arrays("mixed", "both")
    y = spt.targets(**arrays)
    X = torch.arange(len(y.xl), dtype=F64).reshape(-1, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X, *y), batch_size=3
    )
    params = _kw(sp.Weibull, PARAMS["Weibull"])
    total = sum(
        float(
            spt.nll(sp.Weibull, spt.Targets(*fields), params, reduction="sum")
        )
        for _, *fields in loader
    )
    whole = float(spt.nll(sp.Weibull, y, params, reduction="sum"))
    assert total == pytest.approx(whole, rel=1e-12)
    # The six tensors in order are taken as a Targets.
    as_list = float(spt.nll(sp.Weibull, list(y), params, reduction="sum"))
    assert as_list == whole


@pytest.mark.parametrize("name", list(PARAMS))
def test_per_row_parameters_give_each_row_its_own(name):
    # A parameter given per row equals each row's likelihood at its own
    # value: what a network's outputs rely on.
    dist = _dist(name)
    base = PARAMS[name]
    arrays = _arrays("mixed", "both")
    k = len(arrays["x"])
    first = base[0] * np.linspace(0.85, 1.15, k)
    params = _kw(dist, base)
    params[dist.parameter_names[0]] = torch.tensor(first, dtype=F64)
    got = _nll(dist, arrays, params, reduction="none")
    assert got.shape == (k,)
    for i in range(k):
        row = {key: [value[i]] for key, value in arrays.items()}
        expected = -_surpyval_ll(dist, row, [first[i], *base[1:]])
        assert float(got[i]) == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("name", list(PARAMS))
def test_gradient_matches_a_central_difference(name):
    dist = _dist(name)
    params = np.array(PARAMS[name])
    arrays = _arrays("mixed", "both")
    theta = _kw(dist, params, requires_grad=True)
    _nll(dist, arrays, theta).backward()
    for i, p in enumerate(dist.parameter_names):
        h = 1e-6 * abs(params[i])
        up, down = params.copy(), params.copy()
        up[i] += h
        down[i] -= h
        fd = -(
            _surpyval_ll(dist, arrays, up) - _surpyval_ll(dist, arrays, down)
        ) / (2 * h)
        grad = float(theta[p].grad)
        assert np.isfinite(grad)
        assert grad == pytest.approx(fd, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("name", list(PARAMS))
def test_per_row_gradients_are_each_rows_own(name):
    # The gradient in a per-row parameter is that row's term's alone.
    dist = _dist(name)
    base = PARAMS[name]
    arrays = _arrays("mixed", "both")
    k = len(arrays["x"])
    first = base[0] * np.linspace(0.85, 1.15, k)
    params = _kw(dist, base)
    leaf = torch.tensor(first, dtype=F64, requires_grad=True)
    params[dist.parameter_names[0]] = leaf
    _nll(dist, arrays, params).backward()
    for i in range(k):
        row = {key: [value[i]] for key, value in arrays.items()}
        h = 1e-6 * abs(first[i])
        fd = -(
            _surpyval_ll(dist, row, [first[i] + h, *base[1:]])
            - _surpyval_ll(dist, row, [first[i] - h, *base[1:]])
        ) / (2 * h)
        assert float(leaf.grad[i]) == pytest.approx(fd, rel=1e-6, abs=1e-6)


# The Gamma distribution's censored terms where torch's own incomplete
# gamma functions are accurate to 1e-9 only (a shape above 20, near the
# mean), and on both sides of that region's edges.
@pytest.mark.parametrize("shape", [19.4, 19.6, 20.5, 25.0, 300.0, 5000.0])
def test_gamma_with_a_large_shape_equals_surpyval(shape):
    rate = shape / 10.0  # the mean is 10
    spread = 10.0 / np.sqrt(shape)
    at = [10.0 + z * spread for z in (-3.0, -1.0, -0.1, 0.0, 0.4, 2.0, 4.0)]
    arrays = dict(
        x=[*at, [at[1], at[4]], [at[4], at[6]], 6.6, 6.8, 13.2, 13.4],
        c=[1, -1, 1, -1, 1, -1, 1, 2, 2, 1, -1, 1, -1],
        tl=[10.0 - 6.0 * spread] * 9 + [-INF] * 4,
    )
    params = [shape, rate]
    expected = -_surpyval_ll(sp.Gamma, arrays, params)
    theta = _kw(sp.Gamma, params, requires_grad=True)
    loss = _nll(sp.Gamma, arrays, theta)
    assert float(loss.detach()) == pytest.approx(
        expected, rel=1e-10, abs=1e-10
    )
    loss.backward()
    for i, p in enumerate(sp.Gamma.parameter_names):
        h = 1e-6 * params[i]
        up, down = list(params), list(params)
        up[i] += h
        down[i] -= h
        fd = -(
            _surpyval_ll(sp.Gamma, arrays, up)
            - _surpyval_ll(sp.Gamma, arrays, down)
        ) / (2 * h)
        assert float(theta[p].grad) == pytest.approx(fd, rel=1e-6, abs=1e-6)


def test_far_upper_tail_window_stays_finite():
    # A left truncation far in the upper tail, where F(tl) rounds to 1
    # and 1 - F would be 0: taken from log S, as SurPyval does.
    arrays = dict(x=[60.0, 70.0], c=[0, 1], tl=[50.0, 50.0], tr=[90.0, 90.0])
    expected = -_surpyval_ll(sp.Weibull, arrays, [5.0, 2.0])
    theta = _kw(sp.Weibull, [5.0, 2.0], requires_grad=True)
    loss = _nll(sp.Weibull, arrays, theta)
    assert np.isfinite(float(loss.detach()))
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-10)
    loss.backward()
    assert all(torch.isfinite(v.grad) for v in theta.values())


# Rows far in a tail of their own model, where a formula the row does
# not use would overflow or underflow: a failure where log S or log F is
# -inf, a suspension where F rounds to 0, a left-censored row where S
# does. Per-row parameters, as a network gives them.
TAILS = {
    "Weibull": dict(x=[400.0, 1e-9, 1e-9, 400.0], c=[0, 0, 1, -1]),
    "Exponential": dict(x=[2e4, 1e-300, 1e-300, 2e4], c=[0, 0, 1, -1]),
    "LogNormal": dict(x=[1e30, 1e-30, 1e-30, 1e30], c=[0, 0, 1, -1]),
    "Gamma": dict(x=[5e3, 1e-150, 1e-150, 5e3], c=[0, 0, 1, -1]),
    "LogLogistic": dict(x=[1e200, 1e-200, 1e-200, 1e200], c=[0, 0, 1, -1]),
    "Normal": dict(x=[500.0, -500.0, -500.0, 500.0], c=[0, 0, 1, -1]),
    "Gumbel": dict(x=[60.0, -4000.0, -4000.0, 60.0], c=[0, 0, 1, -1]),
    "Logistic": dict(x=[5e3, -5e3, -5e3, 5e3], c=[0, 0, 1, -1]),
    "Rayleigh": dict(x=[300.0, 1e-200, 1e-200, 300.0], c=[0, 0, 1, -1]),
}


@pytest.mark.parametrize("name", list(PARAMS))
def test_far_tails_are_finite_with_finite_gradients(name):
    dist = _dist(name)
    arrays = TAILS[name]
    raw = torch.zeros(4, len(PARAMS[name]), dtype=F64, requires_grad=True)
    shift = torch.tensor(PARAMS[name], dtype=F64)
    params = {
        p: shift[j] + raw[:, j] for j, p in enumerate(dist.parameter_names)
    }
    loss = _nll(dist, arrays, params, reduction="none")
    assert torch.isfinite(loss).all(), loss
    loss.sum().backward()
    assert torch.isfinite(raw.grad).all(), raw.grad
    for i in range(4):
        row = {key: [value[i]] for key, value in arrays.items()}
        expected = -_surpyval_ll(dist, row, PARAMS[name])
        assert loss[i].item() == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("name", list(PARAMS))
def test_tiny_windows_are_finite_with_finite_gradients(name):
    # Intervals a millionth wide, in the body and far in each tail, and
    # a truncation window barely wider than the data.
    dist = _dist(name)
    arrays = dict(
        x=[[9.0, 9.0 + 9e-6], [0.5, 0.5 + 5e-7], [28.0, 28.0 + 2.8e-5]],
        c=[2, 2, 2],
        tl=[0.4] * 3,
        tr=[28.5] * 3,
    )
    theta = _kw(dist, PARAMS[name], requires_grad=True)
    loss = _nll(dist, arrays, theta)
    expected = -_surpyval_ll(dist, arrays, PARAMS[name])
    assert float(loss.detach()) == pytest.approx(
        expected, rel=1e-10, abs=1e-10
    )
    loss.backward()
    assert all(torch.isfinite(v.grad) for v in theta.values())


def test_truncation_bounds_beyond_the_reach_of_double_precision():
    # A Gamma with shape 300 and mean 10: F(1) and S(40) round to 0, so
    # their logs are -inf. The window is then the whole line, as in
    # SurPyval, and its gradient is 0, not NaN.
    arrays = dict(
        x=[9.0, 10.5, [9.5, 11.0], 10.0],
        c=[0, 1, 2, -1],
        tl=[1.0, 1.0, 1.0, 9.0],
        tr=[40.0, INF, 40.0, 40.0],
    )
    params = [300.0, 30.0]
    theta = _kw(sp.Gamma, params, requires_grad=True)
    loss = _nll(sp.Gamma, arrays, theta)
    expected = -_surpyval_ll(sp.Gamma, arrays, params)
    assert float(loss.detach()) == pytest.approx(
        expected, rel=1e-10, abs=1e-10
    )
    loss.backward()
    assert all(torch.isfinite(v.grad) for v in theta.values())


def test_a_suspension_at_the_start_of_the_support_counts_nothing():
    # fit accepts a unit suspended at time 0: log S(0) = 0, with a zero
    # gradient (the formula there is 0 * inf).
    arrays = dict(x=[0.0, 5.0], c=[1, 0])
    theta = _kw(sp.Weibull, [10.0, 1.7], requires_grad=True)
    loss = _nll(sp.Weibull, arrays, theta, reduction="none")
    assert loss[0].item() == 0.0
    loss.sum().backward()
    assert all(torch.isfinite(v.grad) for v in theta.values())


def test_xl_xr_and_t_forms():
    xl, xr = [1.0, 4.0, 6.0], [3.0, 4.0, 9.0]
    t = [[0.5, 20.0]] * 3
    arrays = dict(xl=xl, xr=xr, t=t)
    expected = -_surpyval_ll(sp.Weibull, arrays, [6.0, 1.5])
    got = _nll(sp.Weibull, arrays, _kw(sp.Weibull, [6.0, 1.5]))
    assert float(got) == pytest.approx(expected, rel=1e-10)


def test_an_array_as_data_is_x():
    a = spt.targets([1.0, 2.0, 5.0])
    b = spt.targets(x=[1.0, 2.0, 5.0])
    assert all(torch.equal(u, v) for u, v in zip(a, b))


# constrain


def _inverse(dist, params):
    """The inverse of constrain's maps, written out here."""
    columns = []
    for name, (lo, hi) in zip(dist.parameter_names, dist.bounds):
        value = params[name]
        assert hi is None  # no supported family has an upper bound
        if lo is not None:
            value = torch.log(torch.expm1(value - lo))
        columns.append(value)
    return torch.stack(columns, dim=-1)


@pytest.mark.parametrize("name", list(PARAMS))
def test_constrain_maps_any_real_input_inside_the_range(name):
    dist = _dist(name)
    k = len(dist.parameter_names)
    column = torch.tensor(
        [-1e300, -800.0, -40.0, -1.0, 0.0, 1.0, 40.0, 800.0, 1e300],
        dtype=F64,
    )
    raw = column.repeat(k, 1).T.clone().requires_grad_(True)
    params = spt.constrain(dist, raw)
    assert list(params) == list(dist.parameter_names)
    for (lo, hi), value in zip(dist.bounds, params.values()):
        assert value.shape == (len(column),)
        assert not torch.isnan(value).any()
        if lo is not None:
            assert (value > lo).all()
        if hi is not None:
            assert (value < hi).all()
        assert (value[1:] >= value[:-1]).all()
    torch.stack(list(params.values())).sum().backward()
    assert torch.isfinite(raw.grad).all()


@pytest.mark.parametrize("name", list(PARAMS))
def test_constrain_round_trips_through_its_inverse(name):
    dist = _dist(name)
    k = len(dist.parameter_names)
    raw = torch.linspace(-30.0, 30.0, 13 * k, dtype=F64).reshape(13, k)
    back = _inverse(dist, spt.constrain(dist, raw))
    assert torch.allclose(back, raw, rtol=1e-10, atol=1e-10)
    # One set of parameters, shared by every row: scalars.
    shared = spt.constrain(dist, raw[4])
    assert all(v.ndim == 0 for v in shared.values())
    assert torch.allclose(_inverse(dist, shared), raw[4], rtol=1e-10)


def test_constrain_follows_the_bounds_of_the_distribution():
    # Each kind of range gets its map: read from dist.bounds, so a
    # distribution with other bounds is mapped by them.
    class Bounded:
        name = "Weibull"
        parameter_names = ["alpha", "beta", "gamma", "delta"]
        bounds = ((0, None), (None, None), (2.0, 5.0), (None, 1.0))

    raw = torch.tensor(
        [[-800.0] * 4, [0.0] * 4, [800.0] * 4], dtype=F64, requires_grad=True
    )
    out = spt.constrain(Bounded, raw)
    assert out["alpha"].tolist() == [
        np.nextafter(0.0, 1.0),
        pytest.approx(np.log(2.0), rel=1e-15),
        800.0,
    ]
    assert out["beta"].tolist() == [-800.0, 0.0, 800.0]
    assert out["gamma"][1].item() == 3.5
    assert ((out["gamma"] > 2.0) & (out["gamma"] < 5.0)).all()
    assert out["delta"][1].item() == pytest.approx(1.0 - np.log(2.0))
    assert (out["delta"] < 1.0).all()
    assert out["delta"][2].item() == np.nextafter(1.0, 0.0)
    assert out["gamma"].tolist() == [
        np.nextafter(2.0, 3.0),
        3.5,
        np.nextafter(5.0, 3.0),
    ]


def test_constrain_refuses_a_wrong_width_naming_the_parameters():
    with pytest.raises(ValueError) as err:
        spt.constrain(sp.Weibull, torch.zeros(5, 3, dtype=F64))
    message = str(err.value)
    assert "raw has shape (5, 3)" in message
    assert "2 parameters (alpha, beta, in that order)" in message
    assert "(N, 2)" in message
    with pytest.raises(ValueError, match=r"1 parameter \(failure_rate"):
        spt.constrain(sp.Exponential, torch.zeros(5, 2, dtype=F64))
    with pytest.raises(ValueError, match=r"shape \(2, 5, 2\)"):
        spt.constrain(sp.Weibull, torch.zeros(2, 5, 2, dtype=F64))
    with pytest.raises(TypeError, match="must be a torch tensor"):
        spt.constrain(sp.Weibull, [[0.0, 0.0]])
    with pytest.raises(TypeError, match="use raw.double()"):
        spt.constrain(sp.Weibull, torch.zeros(5, 2, dtype=torch.int64))


# reduction, NLLLoss, dtype


def test_reductions():
    arrays = _arrays("mixed", "both")
    y = spt.targets(**arrays)
    params = _kw(sp.Weibull, PARAMS["Weibull"])
    none = spt.nll(sp.Weibull, y, params, reduction="none")
    total = spt.nll(sp.Weibull, y, params, reduction="sum")
    mean = spt.nll(sp.Weibull, y, params, reduction="mean")
    assert none.shape == (8,)
    assert float(none.sum()) == pytest.approx(float(total), rel=1e-14)
    # The mean is over the units, sum(n) = 19, not the rows.
    assert float(mean) == pytest.approx(float(total) / 19.0, rel=1e-14)
    assert float(spt.nll(sp.Weibull, y, params)) == float(mean)
    with pytest.raises(ValueError, match="reduction must be 'mean'"):
        spt.nll(sp.Weibull, y, params, reduction="average")


def test_nll_loss_is_nll_as_a_module():
    y = spt.targets(**_arrays("mixed", "both"))
    params = _kw(sp.Weibull, PARAMS["Weibull"])
    for reduction in ("mean", "sum", "none"):
        loss_fn = spt.NLLLoss(sp.Weibull, reduction=reduction)
        assert isinstance(loss_fn, torch.nn.Module)
        assert torch.equal(
            loss_fn(y, params),
            spt.nll(sp.Weibull, y, params, reduction=reduction),
        )
    assert (
        repr(spt.NLLLoss(sp.Weibull)) == "NLLLoss(Weibull, reduction='mean')"
    )
    assert spt.NLLLoss is spt.NLLLoss
    assert spt.NLLLoss.__module__ == "surpyval.torch"
    assert list(spt.NLLLoss(sp.Weibull).parameters()) == []
    with pytest.raises(ValueError, match="reduction must be 'mean'"):
        spt.NLLLoss(sp.Weibull, reduction="average")
    with pytest.raises(ValueError, match="supports Weibull, Exponential"):
        spt.NLLLoss(sp.Beta)


def test_dtype_follows_the_parameters():
    y = spt.targets(**_arrays("mixed", "both"))
    numbers = dict(zip(sp.Weibull.parameter_names, PARAMS["Weibull"]))
    double = spt.nll(sp.Weibull, y, numbers)
    assert double.dtype == F64
    single = spt.nll(
        sp.Weibull,
        y,
        {k: torch.tensor(v, dtype=torch.float32) for k, v in numbers.items()},
    )
    assert single.dtype == torch.float32
    assert float(single) == pytest.approx(float(double), rel=1e-5)
    # A number beside a tensor takes the tensor's type.
    mixed = spt.nll(
        sp.Weibull,
        y,
        {"alpha": torch.tensor(10.0, dtype=torch.float32), "beta": 1.7},
    )
    assert mixed.dtype == torch.float32


def test_training_a_linear_model_recovers_the_fit():
    # A Weibull whose log scale is linear in one covariate, right
    # censored at 60: full-batch L-BFGS from zero on nll must reach the
    # maximum likelihood, which WeibullAFT finds for the same model. The
    # seeds (data 20261120, torch 0) and the tolerance (1e-5 relative on
    # the scale and shape per row, 1e-8 on the likelihood) were set
    # before the first run.
    rng = np.random.default_rng(20261120)
    z = rng.uniform(-1.0, 1.0, 400)
    life = np.exp(3.0 + 0.5 * z) * rng.weibull(2.0, 400)
    x, c = np.minimum(life, 60.0), (life > 60.0).astype(int)
    torch.manual_seed(0)
    Z = torch.tensor(z, dtype=F64).reshape(-1, 1)
    y = spt.targets(x=x, c=c)
    net = torch.nn.Linear(1, 1).double()
    raw_beta = torch.nn.Parameter(torch.zeros((), dtype=F64))
    torch.nn.init.zeros_(net.weight)
    torch.nn.init.constant_(net.bias, float(np.log(np.mean(x))))
    loss_fn = spt.NLLLoss(sp.Weibull, reduction="sum")
    optimiser = torch.optim.LBFGS(
        [*net.parameters(), raw_beta],
        max_iter=500,
        tolerance_grad=1e-10,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def params():
        both = spt.constrain(sp.Weibull, torch.stack([raw_beta, raw_beta]))
        return {"alpha": torch.exp(net(Z).squeeze(-1)), "beta": both["beta"]}

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(y, params())
        loss.backward()
        return loss

    optimiser.step(closure)
    fitted = params()
    model = sp.WeibullAFT.fit(x=x, c=c, Z=z.reshape(-1, 1))
    assert loss_fn(y, fitted).item() == pytest.approx(
        -model.log_likelihood, rel=1e-8
    )
    beta, b0, b1 = _weibull_aft(model)
    assert fitted["beta"].item() == pytest.approx(beta, rel=1e-5)
    assert net.bias.item() == pytest.approx(b0, rel=1e-5)
    assert net.weight.item() == pytest.approx(b1, rel=1e-5)


def _weibull_aft(model):
    """The shape, and the intercept and slope of the log scale, of a
    fitted WeibullAFT, read from its predictions (so that this does not
    depend on how the model orders its parameters)."""
    at = np.array([[0.0], [1.0]])
    t = np.array([1.0, 2.0])
    h0 = -np.log(model.sf(t, np.array([[0.0], [0.0]])))
    beta = np.log(h0[1] / h0[0]) / np.log(2.0)
    scale = [
        (-np.log(model.sf(np.array([1.0]), at[[i]]))[0]) ** (-1.0 / beta)
        for i in range(2)
    ]
    return beta, np.log(scale[0]), np.log(scale[1] / scale[0])


# Refusals: they read as fit's.


@pytest.mark.parametrize(
    "arrays",
    [
        dict(x=[1.0, 2.0], c=[0, 3]),
        dict(x=[1.0, np.nan]),
        dict(x=[1.0, 2.0], n=[1, 0]),
        dict(x=[1.0, 5.0], tl=[2.0, 2.0]),
    ],
)
def test_invalid_data_is_refused_as_fit_refuses_it(arrays):
    with pytest.raises(ValueError) as fit_err:
        sp.Weibull.fit(**arrays)
    with pytest.raises(ValueError) as ours:
        spt.targets(**arrays)
    assert str(ours.value) == str(fit_err.value)


@pytest.mark.parametrize(
    "arrays",
    [
        dict(x=[-1.0, 2.0, 3.0]),
        dict(x=[0.0, 2.0, 3.0]),
        dict(x=[2.0, 0.0], c=[0, -1]),
        dict(x=[2.0, -1.0], c=[0, 1]),
        dict(x=[2.0, [-1.0, 3.0]], c=[0, 2]),
    ],
)
def test_data_outside_the_support_is_refused_where_fit_refuses_it(arrays):
    with pytest.raises(ValueError, match="outside the support"):
        # (the check every fit of the distribution runs)
        sp.Weibull._check_inside_support(
            sp.SurpyvalData(**arrays), every_row=True
        )
    y = spt.targets(**arrays)
    with pytest.raises(ValueError) as err:
        spt.nll(sp.Weibull, y, {"alpha": 1.0, "beta": 1.0})
    message = str(err.value)
    assert "outside the support of the Weibull distribution" in message
    assert "first rows: [" in message
    assert "Remove or correct those rows" in message
    # The same rows are inside the support of a distribution on the line
    assert torch.isfinite(spt.nll(sp.Normal, y, {"mu": 1.0, "sigma": 1.0}))


def test_an_unsupported_family_is_refused_with_the_supported_list():
    y = spt.targets(x=[0.2, 0.5])
    with pytest.raises(ValueError, match="supports Weibull, Exponential"):
        spt.nll(sp.Beta, y, {"alpha": 1.0, "beta": 1.0})
    with pytest.raises(ValueError, match="no torch likelihood for Beta"):
        spt.constrain(sp.Beta, torch.zeros(2, 2, dtype=F64))


def test_a_fitted_model_is_refused():
    model = sp.Weibull.fit([1.0, 2.0, 4.0, 7.0])
    y = spt.targets(x=[1.0, 2.0])
    with pytest.raises(TypeError, match="not a fitted model"):
        spt.nll(model, y, {"alpha": 1.0, "beta": 1.0})


def test_missing_and_unknown_parameters_are_refused():
    y = spt.targets(x=[1.0, 2.0])
    with pytest.raises(TypeError, match="missing beta"):
        spt.nll(sp.Weibull, y, {"alpha": 1.0})
    with pytest.raises(TypeError, match="no parameter 'shape'"):
        spt.nll(sp.Weibull, y, {"alpha": 1.0, "beta": 1, "shape": 2})
    with pytest.raises(TypeError, match="params must be a dict"):
        spt.nll(sp.Weibull, y, torch.ones(2, 2, dtype=F64))


def test_a_parameter_of_the_wrong_shape_is_refused():
    y = spt.targets(x=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError) as err:
        spt.nll(
            sp.Weibull,
            y,
            {"alpha": torch.ones(2, dtype=F64), "beta": 1.0},
        )
    assert "alpha has shape (2,) and the target 3 rows" in str(err.value)
    with pytest.raises(ValueError, match=r"use \.squeeze\(-1\)"):
        spt.nll(
            sp.Weibull,
            y,
            {"alpha": torch.ones(3, 1, dtype=F64), "beta": 1.0},
        )


def test_a_target_that_is_not_one_is_refused():
    params = {"alpha": 1.0, "beta": 1.0}
    with pytest.raises(TypeError, match="target must be a Targets"):
        spt.nll(sp.Weibull, torch.ones(3, dtype=F64), params)
    with pytest.raises(TypeError, match="target must be a Targets"):
        spt.nll(sp.Weibull, None, params)
    y = spt.targets(x=[1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="all of one length"):
        spt.nll(sp.Weibull, y._replace(n=y.n[:2]), params)
    with pytest.raises(ValueError, match="all of one length"):
        spt.nll(sp.Weibull, y._replace(n=[1.0, 1.0, 1.0]), params)


def test_data_given_twice_is_refused():
    data = sp.SurpyvalData(x=[1.0, 2.0])
    with pytest.raises(TypeError, match="leave out 'c'"):
        spt.targets(data, c=[0, 1])
    with pytest.raises(TypeError, match="dtype must be a floating type"):
        spt.targets(data, dtype=torch.int64)
