"""The mixture loss of ``surpyval.torch`` (``nll(..., mixture=)``)
against ``MixtureModel``'s own likelihood.

At fixed parameters and weights, the loss of a mixture of ``K``
components of one family must equal the likelihood ``MixtureModel``
fits (``neg_ll_of``) for every supported family, every row type and
every truncation pattern, to 1e-10 in double precision; with ``K = 1``
it must be the loss without a mixture, bit for bit. The references are
SurPyval's own values, so nothing depends on the platform.

These run only with ``--run-torch`` (and PyTorch installed): in CI, the
``torch`` job.
"""

import numpy as np
import pytest

import surpyval as sp

torch = pytest.importorskip("torch")

import surpyval.torch as spt  # noqa: E402
from surpyval.tests.torch.test_torch_loss import (  # noqa: E402
    PARAMS,
    ROWS,
    TRUNCATION,
    _arrays,
)

F64 = torch.float64

# Three components of each family, with their mass spread over the times
# of ROWS (between 1 and 30), and the weights.
MIX = {
    "Weibull": [[6.0, 1.5], [18.0, 3.0], [11.0, 0.8]],
    "Exponential": [[0.3], [0.05], [0.1]],
    "LogNormal": [[1.2, 0.5], [2.8, 0.4], [2.2, 1.0]],
    "Gamma": [[2.0, 0.5], [9.0, 0.5], [2.5, 0.25]],
    "LogLogistic": [[5.0, 3.0], [20.0, 4.0], [9.0, 1.5]],
    "Normal": [[5.0, 3.0], [18.0, 4.0], [11.0, 8.0]],
    "Gumbel": [[8.0, 3.0], [20.0, 4.0], [14.0, 6.0]],
    "Logistic": [[5.0, 2.0], [18.0, 3.0], [11.0, 5.0]],
    "Rayleigh": [[4.0], [15.0], [9.0]],
    "Gompertz": [[0.3, 0.05], [0.15, 0.005], [0.05, 0.05]],
}
W = [0.2, 0.5, 0.3]


def _dist(name):
    return getattr(sp, name)


def _reference(dist, arrays, components, w):
    """``MixtureModel``'s negative log-likelihood of the data at these
    components and weights: the one ``fit`` minimises."""
    model = sp.MixtureModel.from_params(dist, components, w)
    model.data = sp.SurpyvalData(**arrays)
    model._truncated = bool(np.isfinite(model.data.t).any())
    return float(model.neg_ll_of(model.w, model.params))


def _params(dist, components, requires_grad=False):
    """The components' parameters as ``(K,)`` tensors by name."""
    columns = np.asarray(components, dtype=float).T
    return {
        name: torch.tensor(column, dtype=F64, requires_grad=requires_grad)
        for name, column in zip(dist.parameter_names, columns)
    }


def _logits(w, requires_grad=False):
    return torch.tensor(np.log(w), dtype=F64, requires_grad=requires_grad)


def test_every_supported_family_is_tested():
    assert set(MIX) == set(spt.SUPPORTED)


@pytest.mark.parametrize("truncation", list(TRUNCATION))
@pytest.mark.parametrize("rows", list(ROWS))
@pytest.mark.parametrize("name", list(MIX))
def test_mixture_nll_equals_the_mixture_model(name, rows, truncation):
    dist = _dist(name)
    if truncation == "left_at_support_edge" and dist.support[0] != 0:
        pytest.skip("only a positive family has its support edge at 0")
    arrays = _arrays(rows, truncation)
    expected = _reference(dist, arrays, MIX[name], W)
    got = spt.nll(
        dist,
        spt.targets(**arrays),
        _params(dist, MIX[name]),
        mixture=_logits(W),
        reduction="sum",
    )
    assert np.isfinite(expected)
    assert float(got) == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("name", list(MIX))
def test_each_row_has_its_own_components_and_weights(name):
    # (N, K) parameters and logits: each row's term is the mixture's
    # likelihood of that row alone, at its own components and weights.
    dist = _dist(name)
    arrays = _arrays("mixed", "both")
    rows = len(arrays["x"])
    base = np.asarray(MIX[name], dtype=float)
    scale = np.linspace(0.85, 1.15, rows)
    # The first parameter of every component varies by row; the weights
    # rotate.
    per_row = np.repeat(base[None], rows, axis=0)
    per_row[:, :, 0] *= scale[:, None]
    weights = np.array([np.roll(W, i) for i in range(rows)])
    params = {
        name_: torch.tensor(per_row[:, :, j], dtype=F64)
        for j, name_ in enumerate(dist.parameter_names)
    }
    got = spt.nll(
        dist,
        spt.targets(**arrays),
        params,
        mixture=torch.tensor(np.log(weights), dtype=F64),
        reduction="none",
    )
    assert got.shape == (rows,)
    for i in range(rows):
        row = {key: [value[i]] for key, value in arrays.items()}
        expected = _reference(dist, row, per_row[i], weights[i])
        assert float(got[i]) == pytest.approx(expected, rel=1e-10, abs=1e-10)


@pytest.mark.parametrize("name", list(PARAMS))
def test_one_component_is_the_loss_without_a_mixture(name):
    dist = _dist(name)
    arrays = _arrays("mixed", "both")
    y = spt.targets(**arrays)
    rows = len(arrays["x"])
    first = PARAMS[name][0] * np.linspace(0.85, 1.15, rows)
    plain = {
        p: torch.tensor(v, dtype=F64)
        for p, v in zip(dist.parameter_names, PARAMS[name])
    }
    plain[dist.parameter_names[0]] = torch.tensor(first, dtype=F64)
    one = {p: v.reshape(-1, 1) if v.ndim else v for p, v in plain.items()}
    per_row_logits = torch.linspace(-3.0, 3.0, rows, dtype=F64)
    for logits in (torch.zeros(1, dtype=F64), per_row_logits[:, None]):
        without = spt.nll(dist, y, plain, reduction="none")
        within = spt.nll(dist, y, one, mixture=logits, reduction="none")
        assert torch.equal(without, within)


@pytest.mark.parametrize("name", list(MIX))
def test_shared_parameters_are_every_rows(name):
    # (K,) parameters and logits are the same as each repeated per row,
    # and a scalar is the same as one value for every component.
    dist = _dist(name)
    arrays = _arrays("mixed", "both")
    y = spt.targets(**arrays)
    rows = len(arrays["x"])
    shared = _params(dist, MIX[name])
    logits = _logits(W)
    expanded = {p: v.expand(rows, 3) for p, v in shared.items()}
    a = spt.nll(dist, y, shared, mixture=logits, reduction="none")
    b = spt.nll(
        dist, y, expanded, mixture=logits.expand(rows, 3), reduction="none"
    )
    assert torch.allclose(a, b, rtol=1e-15, atol=0)
    last = dist.parameter_names[-1]
    value = float(shared[last][0])
    scalar = {**shared, last: value}
    same = {**shared, last: torch.full((3,), value, dtype=F64)}
    assert float(spt.nll(dist, y, scalar, mixture=logits)) == float(
        spt.nll(dist, y, same, mixture=logits)
    )


@pytest.mark.parametrize("name", list(MIX))
def test_mixture_gradient_matches_a_central_difference(name):
    # In every component's parameters and every logit, against a central
    # difference of MixtureModel's likelihood.
    dist = _dist(name)
    arrays = _arrays("mixed", "both")
    components = np.asarray(MIX[name], dtype=float)
    logit_values = np.log(W)
    params = _params(dist, components, requires_grad=True)
    logits = _logits(W, requires_grad=True)
    spt.nll(
        dist, spt.targets(**arrays), params, mixture=logits, reduction="sum"
    ).backward()

    def reference(c, lv):
        w = np.exp(lv - np.log(np.exp(lv).sum()))
        return _reference(dist, arrays, c, w)

    for j, p in enumerate(dist.parameter_names):
        for k in range(len(W)):
            h = 1e-6 * abs(components[k, j])
            up, down = components.copy(), components.copy()
            up[k, j] += h
            down[k, j] -= h
            fd = (
                reference(up, logit_values) - reference(down, logit_values)
            ) / (2 * h)
            grad = float(params[p].grad[k])
            assert np.isfinite(grad)
            assert grad == pytest.approx(fd, rel=1e-6, abs=1e-6)
    for k in range(len(W)):
        h = 1e-6
        up, down = logit_values.copy(), logit_values.copy()
        up[k] += h
        down[k] -= h
        fd = (reference(components, up) - reference(components, down)) / (
            2 * h
        )
        assert float(logits.grad[k]) == pytest.approx(fd, rel=1e-6, abs=1e-6)


@pytest.mark.parametrize("name", list(MIX))
def test_a_dominant_component_stays_finite(name):
    # A weight within 2e-35 of 1: the loss is still the mixture's, and
    # the gradient is finite in every parameter and logit. (Without
    # truncation: MixtureModel takes a censored row's window as a
    # difference of CDFs, which for the dominant Gumbel's [18, 29] loses
    # 4 digits; this loss, worked out from the survival function there,
    # agrees with 50-digit arithmetic.)
    dist = _dist(name)
    arrays = _arrays("mixed", "none")
    y = spt.targets(**arrays)
    params = _params(dist, MIX[name], requires_grad=True)
    logits = torch.tensor([40.0, -40.0, -40.0], dtype=F64, requires_grad=True)
    loss = spt.nll(dist, y, params, mixture=logits, reduction="sum")
    loss.backward()
    w = torch.softmax(logits.detach(), dim=0).numpy()
    expected = _reference(dist, arrays, MIX[name], w)
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-10)
    for value in [*params.values(), logits]:
        assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize(
    "name, components, arrays",
    [
        # Q(2, 1500) underflows to 0: the first component gives the
        # suspension at 300 probability 0 in double precision.
        ("Gamma", [[2.0, 5.0], [2.5, 0.05]], dict(x=[200.0, 300.0], c=[0, 1])),
        # (30 / 1) ** 400 overflows: the first component's density and
        # survival at 30 are 0.
        (
            "Weibull",
            [[1.0, 400.0], [20.0, 2.0]],
            dict(x=[2.0, 30.0, 30.0, [25.0, 35.0]], c=[0, 0, 1, 2]),
        ),
        # A truncation window the first component gives probability 0.
        (
            "Gamma",
            [[2.0, 5.0], [2.5, 0.05]],
            dict(x=[300.0, 400.0], c=[0, 1], tl=[250.0, 250.0]),
        ),
    ],
)
def test_a_component_with_probability_0_leaves_the_gradient_finite(
    name, components, arrays
):
    dist = _dist(name)
    y = spt.targets(**arrays)
    params = _params(dist, components, requires_grad=True)
    logits = _logits([0.5, 0.5], requires_grad=True)
    loss = spt.nll(dist, y, params, mixture=logits, reduction="sum")
    loss.backward()
    assert torch.isfinite(loss)
    for value in [*params.values(), logits]:
        assert torch.isfinite(value.grad).all()
    # The second component alone explains the rows (the first gives each
    # at most e^-900 of its probability), at half the weight.
    alone = spt.nll(
        dist,
        y,
        {p: v[1].detach() for p, v in params.items()},
        reduction="none",
    )
    per_row = spt.nll(dist, y, params, mixture=logits, reduction="none")
    lost = per_row.detach() - alone
    truncated = torch.isfinite(y.tl)
    # log 2 lost per row, but where the window takes it back.
    expected = torch.where(
        truncated, 0.0, torch.tensor(np.log(2.0), dtype=F64)
    )
    assert torch.allclose(lost, expected, rtol=0, atol=1e-12)


def test_nll_loss_takes_the_mixture():
    arrays = _arrays("mixed", "both")
    y = spt.targets(**arrays)
    params = _params(sp.Weibull, MIX["Weibull"])
    loss_fn = spt.NLLLoss(sp.Weibull, reduction="sum")
    assert float(loss_fn(y, params, mixture=_logits(W))) == float(
        spt.nll(sp.Weibull, y, params, mixture=_logits(W), reduction="sum")
    )
    assert float(loss_fn(y, params, _logits(W))) == float(
        loss_fn(y, params, mixture=_logits(W))
    )


def test_constrain_maps_a_mixtures_outputs():
    raw = torch.randn(5, 3, 2, dtype=F64)
    params = spt.constrain(sp.Weibull, raw)
    assert params["alpha"].shape == (5, 3)
    assert torch.equal(
        params["alpha"][2], spt.constrain(sp.Weibull, raw[2])["alpha"]
    )


def test_mixture_shapes_that_do_not_fit_are_refused():
    y = spt.targets(**_arrays("mixed", "none"))
    logits = torch.zeros(3, dtype=F64)
    with pytest.raises(ValueError) as err:
        spt.nll(
            sp.Weibull,
            y,
            {"alpha": torch.ones(8, dtype=F64), "beta": 1.0},
            mixture=logits,
        )
    message = str(err.value)
    assert "3 components" in message
    assert "target 8 rows" in message
    assert "alpha (8,)" in message
    assert "p[:, None]" in message
    with pytest.raises(ValueError, match=r"alpha \(8, 2\)"):
        spt.nll(
            sp.Weibull,
            y,
            {"alpha": torch.ones(8, 2, dtype=F64), "beta": 1.0},
            mixture=logits,
        )
    with pytest.raises(ValueError, match=r"mixture \(5, 3\)"):
        spt.nll(
            sp.Weibull,
            y,
            {"alpha": 1.0, "beta": 1.0},
            mixture=torch.zeros(5, 3, dtype=F64),
        )
    with pytest.raises(ValueError, match=r"mixture has shape \(8, 3, 1\)"):
        spt.nll(
            sp.Weibull,
            y,
            {"alpha": 1.0, "beta": 1.0},
            mixture=torch.zeros(8, 3, 1, dtype=F64),
        )
    with pytest.raises(TypeError, match="mixture must be a tensor"):
        spt.nll(sp.Weibull, y, {"alpha": 1.0, "beta": 1.0}, mixture=[0, 0])
    with pytest.raises(TypeError, match="use .double()"):
        spt.nll(
            sp.Weibull,
            y,
            {"alpha": 1.0, "beta": 1.0},
            mixture=torch.zeros(3, dtype=torch.int64),
        )


def test_per_row_parameters_without_a_mixture_point_to_it():
    y = spt.targets(**_arrays("mixed", "none"))
    with pytest.raises(ValueError, match="give its logits as mixture="):
        spt.nll(
            sp.Weibull, y, {"alpha": torch.ones(8, 3), "beta": torch.ones(8)}
        )


def test_the_type_follows_the_parameters_and_logits():
    y = spt.targets(**_arrays("mixed", "none"))
    params = {
        "alpha": torch.tensor([6.0, 18.0], dtype=torch.float32),
        "beta": torch.tensor([1.5, 3.0], dtype=torch.float32),
    }
    loss = spt.nll(
        sp.Weibull, y, params, mixture=torch.zeros(2, dtype=torch.float32)
    )
    assert loss.dtype == torch.float32
    loss = spt.nll(
        sp.Weibull, y, params, mixture=torch.zeros(2, dtype=torch.float64)
    )
    assert loss.dtype == torch.float64


def test_training_a_mixture_recovers_its_survival_curves():
    # Data from a mixture of two Weibulls whose weights depend on a
    # covariate (the first component's logit is 2.5 z), right censored at
    # 60. A linear layer gives the logits and the components' parameters
    # are shared; full-batch L-BFGS on the mixture loss. The seeds (data
    # 20261120, torch 0), the sample size (3000) and the tolerance (0.04
    # on every survival probability, at z = -0.8, 0 and 0.8 and times 2
    # to 50) were set before the first run. The fit's start is the
    # data's quartiles for the scales, not the truth.
    rng = np.random.default_rng(20261120)
    size = 3000
    z = rng.uniform(-1.0, 1.0, size)
    truth = np.array([[6.0, 3.0], [30.0, 2.0]])
    first = rng.uniform(size=size) < 1.0 / (1.0 + np.exp(-2.5 * z))
    which = np.where(first, 0, 1)
    life = truth[which, 0] * rng.weibull(truth[which, 1])
    x, c = np.minimum(life, 60.0), (life > 60.0).astype(int)

    torch.manual_seed(0)
    Z = torch.tensor(z, dtype=F64).reshape(-1, 1)
    y = spt.targets(x=x, c=c)
    gate = torch.nn.Linear(1, 2).double()
    start = np.log(np.expm1(np.r_[np.quantile(x, [0.25, 0.75]), 1.0, 1.0]))
    raw = torch.nn.Parameter(
        torch.tensor(start, dtype=F64).reshape(2, 2).T.contiguous()
    )
    loss_fn = spt.NLLLoss(sp.Weibull, reduction="sum")
    optimiser = torch.optim.LBFGS(
        [*gate.parameters(), raw],
        max_iter=2000,
        tolerance_grad=1e-9,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimiser.zero_grad()
        loss = loss_fn(y, spt.constrain(sp.Weibull, raw), mixture=gate(Z))
        loss.backward()
        return loss

    optimiser.step(closure)
    times = np.linspace(2.0, 50.0, 25)
    at = torch.tensor([[-0.8], [0.0], [0.8]], dtype=F64)
    with torch.no_grad():
        got = spt.sf(
            sp.Weibull, times, spt.constrain(sp.Weibull, raw), mixture=gate(at)
        ).numpy()
    for i, value in enumerate([-0.8, 0.0, 0.8]):
        weight = 1.0 / (1.0 + np.exp(-2.5 * value))
        expected = sp.MixtureModel.from_params(
            sp.Weibull, truth, [weight, 1.0 - weight]
        ).sf(times)
        error = np.abs(got[i] - expected).max()
        assert error < 0.04, (value, error)
