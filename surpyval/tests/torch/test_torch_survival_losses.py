"""The Cox partial likelihood and the discrete-time hazard loss of
``surpyval.torch`` against SurPyval's own likelihoods.

``cox_nll`` (``reduction="sum"``) must equal ``CoxPH``'s negative log
partial likelihood at the same coefficients, for each tie method offered
(Breslow, Efron), at the fitted coefficients and at two others, with ties,
counts and delayed entry, to 1e-8. ``discrete_time_nll`` must equal the
negative log-likelihood of ``DiscreteWeibull`` and ``Geometric`` (whose
hazards are the logits' sigmoid) on discrete data, and, for hazards of no
named family, a likelihood written directly in numpy. Gradients must
agree with central differences. The references are SurPyval's own values,
so nothing depends on the platform.

Run only with ``--run-torch`` (and PyTorch installed).
"""

import numpy as np
import pytest

import surpyval as sp

torch = pytest.importorskip("torch")

import surpyval.torch as spt  # noqa: E402

F64 = torch.float64


def _cox_data(n=70, seed=0, entry=False, counts=False):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    # rounded times: many ties
    x = np.round(rng.exponential(5, n) * np.exp(-0.5 * Z[:, 0]), 0) + 1
    c = (rng.uniform(size=n) < 0.3).astype(int)
    tl = np.full(n, -np.inf)
    if entry:
        tl = np.where(rng.uniform(size=n) < 0.4, rng.integers(0, 3, n), tl)
        tl = np.minimum(tl, x - 1.0)
    k = np.ones(n)
    if counts:
        k = rng.integers(1, 4, n).astype(float)
    return x, Z, c, k, tl


def _reference(x, Z, c, k, tl, ties):
    neg_ll, _ = sp.CoxPH._resolve_func_generator(ties)(x, Z, c, k, tl)
    return neg_ll


def _loss(x, Z, c, k, tl, ties, beta, reduction="sum"):
    kw = {"x": x, "c": c, "n": k}
    if np.isfinite(tl).any():
        kw["tl"] = np.where(np.isfinite(tl), tl, -np.inf)
    eta = torch.as_tensor(Z @ np.asarray(beta), dtype=F64)
    return spt.cox_nll(eta, spt.targets(**kw), ties=ties, reduction=reduction)


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize(
    "entry, counts",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_cox_nll_equals_coxph(ties, entry, counts):
    x, Z, c, k, tl = _cox_data(entry=entry, counts=counts)
    kw = {"x": x, "Z": Z, "c": c, "n": k, "tie_method": ties}
    if entry:
        kw["tl"] = tl
    fitted = sp.CoxPH.fit(**kw)
    reference = _reference(x, Z, c, k, tl, ties)
    # the fit centres Z; the partial likelihood is the same at the same
    # coefficients (a shift of Z is a constant added to every eta)
    for beta in (np.asarray(fitted.params), [0.3, -0.2], [1.5, 0.7]):
        expected = float(reference(np.asarray(beta, dtype=float)))
        got = float(_loss(x, Z, c, k, tl, ties, beta))
        assert got == pytest.approx(expected, rel=1e-8, abs=1e-8)
    assert float(_loss(x, Z, c, k, tl, ties, fitted.params)) == pytest.approx(
        fitted._neg_ll, rel=1e-8
    )


@pytest.mark.parametrize("ties", ["breslow", "efron"])
def test_cox_gradient_is_the_finite_difference(ties):
    x, Z, c, k, tl = _cox_data(entry=True, counts=True)
    beta = np.array([0.4, -0.3])
    eta = torch.as_tensor(Z @ beta, dtype=F64).requires_grad_(True)
    y = spt.targets(x=x, c=c, n=k, tl=tl)
    spt.cox_nll(eta, y, ties=ties, reduction="sum").backward()
    got = Z.T @ eta.grad.numpy()
    reference = _reference(x, Z, c, k, tl, ties)
    h = 1e-6
    expected = np.array(
        [
            (reference(beta + h * e) - reference(beta - h * e)) / (2 * h)
            for e in np.eye(2)
        ]
    )
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6)


def test_cox_without_ties_the_methods_agree_and_a_shift_changes_nothing():
    rng = np.random.default_rng(3)
    x = rng.uniform(1, 10, 40)
    c = (rng.uniform(size=40) < 0.3).astype(int)
    eta = torch.as_tensor(rng.normal(size=40), dtype=F64)
    y = spt.targets(x=x, c=c)
    a = spt.cox_nll(eta, y, ties="breslow", reduction="sum")
    b = spt.cox_nll(eta, y, ties="efron", reduction="sum")
    assert float(a) == pytest.approx(float(b), rel=1e-12)
    shifted = spt.cox_nll(eta + 7.0, y, ties="efron", reduction="sum")
    assert float(shifted) == pytest.approx(float(b), rel=1e-10)


def test_cox_mean_is_per_failure_and_large_risks_do_not_overflow():
    x, Z, c, k, tl = _cox_data(counts=True)
    total = float(_loss(x, Z, c, k, tl, "efron", [0.3, 0.1]))
    mean = float(_loss(x, Z, c, k, tl, "efron", [0.3, 0.1], "mean"))
    assert mean == pytest.approx(total / k[c == 0].sum(), rel=1e-12)
    big = spt.cox_nll(
        torch.as_tensor(Z @ [0.3, 0.1] + 800.0, dtype=F64),
        spt.targets(x=x, c=c, n=k),
        reduction="sum",
    )
    assert float(big) == pytest.approx(total, rel=1e-9)


@pytest.mark.parametrize(
    "dtype, spread", [(torch.float32, 110.0), (torch.float64, 800.0)]
)
def test_cox_efron_large_risk_spread_matches_exact_value_and_gradient(
    dtype, spread
):
    # Row 0 is censored before the tied failures at t = 1; its risk score
    # does not enter any risk set, so neither loss nor gradient depends on it.
    y = spt.targets(x=[0.5, 1.0, 1.0, 2.0], c=[1, 0, 0, 0])
    eta = torch.tensor(
        [spread, 0.0, 0.0, 0.0], dtype=dtype, requires_grad=True
    )
    loss = spt.cox_nll(eta, y, ties="efron", reduction="sum")
    loss.backward()
    assert float(loss.detach()) == pytest.approx(np.log(6.0), rel=1e-6)
    np.testing.assert_allclose(
        eta.grad.numpy(),
        [0.0, -5.0 / 12.0, -5.0 / 12.0, 10.0 / 12.0],
        rtol=1e-5,
        atol=1e-6,
    )


@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize(
    "dtype, spread", [(torch.float32, 20.0), (torch.float64, 50.0)]
)
def test_cox_delayed_entry_large_late_risk_avoids_cancellation(
    ties, dtype, spread
):
    # Row 1 enters at t = 2.5 with a much larger risk score than the units
    # at risk at t = 2.0, where subtracting the late-entry suffix sum cancels.
    x = np.array([2.0, 3.0, 3.0, 5.0, 8.0])
    c = np.array([0, 0, 0, 1, 0])
    tl = np.array([0.0, 2.5, 2.5, 0.0, 0.0])
    y = spt.targets(x=x, c=c, tl=tl)
    eta_vals = np.array([0.0, spread, 0.0, 0.0, 0.0])
    eta = torch.tensor(eta_vals, dtype=dtype, requires_grad=True)

    loss = spt.cox_nll(eta, y, ties=ties, reduction="sum")
    loss.backward()

    ref_loss = float(
        _reference(x, eta_vals[:, None], c, np.ones(len(x)), tl, ties)(
            np.array([1.0])
        )
    )
    loss_val = float(loss.detach())
    assert np.isfinite(loss_val)
    assert loss_val == pytest.approx(ref_loss, rel=1e-5, abs=1e-5)
    assert np.all(np.isfinite(eta.grad.numpy()))
    # Check gradient against float64 central differences on CoxPH's likelihood
    Z_id = np.eye(len(x))
    ref_fn = _reference(x, Z_id, c, np.ones(len(x)), tl, ties)
    h = 1e-6
    expected_grad = np.array(
        [
            (ref_fn(eta_vals + h * e) - ref_fn(eta_vals - h * e)) / (2 * h)
            for e in Z_id
        ]
    )
    np.testing.assert_allclose(
        eta.grad.numpy(), expected_grad, rtol=1e-5, atol=1e-5
    )


def test_cox_refusals_say_what_to_do():
    y = spt.targets(x=[1.0, 2.0, 3.0], c=[0, 0, 1])
    eta = torch.zeros(3, dtype=F64)
    with pytest.raises(ValueError, match="not offered"):
        spt.cox_nll(eta, y, ties="exact")
    with pytest.raises(ValueError, match="ties must be"):
        spt.cox_nll(eta, y, ties="best")
    with pytest.raises(ValueError, match="squeeze"):
        spt.cox_nll(eta[:, None], y)
    with pytest.raises(ValueError, match="same rows"):
        spt.cox_nll(eta[:2], y)
    with pytest.raises(ValueError, match="no row failed"):
        spt.cox_nll(eta, spt.targets(x=[1.0, 2.0, 3.0], c=[1, 1, 1]))
    with pytest.raises(ValueError, match="exact .* and right-censored"):
        spt.cox_nll(eta, spt.targets(x=[1.0, 2.0, 3.0], c=[0, -1, 0]))
    with pytest.raises(ValueError, match="right truncation"):
        spt.cox_nll(eta, spt.targets(x=[1.0, 2.0, 3.0], tr=[9.0] * 3))
    with pytest.raises(ValueError, match="reduction"):
        spt.cox_nll(eta, y, reduction="none")


# --- discrete time -----------------------------------------------------------
def _logits_from_sf(model, intervals):
    """The logit of each interval's hazard, intervals counted from 0:
    interval k is the discrete time k + 1."""
    t = np.arange(0, intervals + 1)
    s = np.asarray(model.sf(t), dtype=float)
    h = 1.0 - s[1:] / s[:-1]
    return np.log(h / (1.0 - h))


DATA = dict(
    x=[1, 2, 2, 3, 4, 5, 3, 6, 2, 4, 6, 1],
    c=[0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0],
    n=[1, 2, 1, 3, 1, 2, 1, 1, 2, 1, 4, 1],
)


@pytest.mark.parametrize(
    "dist, params",
    [
        (sp.DiscreteWeibull, [0.9, 1.3]),
        (sp.DiscreteWeibull, [0.8, 0.7]),
        (sp.Geometric, [0.2]),
    ],
)
def test_discrete_nll_equals_the_discrete_models_likelihood(dist, params):
    data = sp.SurpyvalData(**DATA)
    expected = -float(dist._log_likelihood(data, *params, 0.0, 0.0, 1.0))
    model = dist.from_params(params)
    logits = _logits_from_sf(model, 6)
    got = spt.discrete_time_nll(
        torch.as_tensor(logits, dtype=F64),
        np.asarray(DATA["x"]) - 1,
        c=DATA["c"],
        n=DATA["n"],
        reduction="sum",
    )
    assert float(got) == pytest.approx(expected, rel=1e-8, abs=1e-8)


def test_discrete_nll_at_the_fitted_model_and_per_row_logits():
    fitted = sp.DiscreteWeibull.fit(**DATA)
    logits = _logits_from_sf(fitted, 6)
    rows = len(DATA["x"])
    got = spt.discrete_time_nll(
        torch.as_tensor(np.tile(logits, (rows, 1)), dtype=F64),
        np.asarray(DATA["x"]) - 1,
        c=DATA["c"],
        n=DATA["n"],
        reduction="sum",
    )
    assert float(got) == pytest.approx(fitted.neg_ll(), rel=1e-8)


def test_discrete_nll_equals_a_likelihood_written_directly():
    rng = np.random.default_rng(1)
    rows, intervals = 25, 5
    z = rng.normal(size=(rows, intervals))
    k = rng.integers(0, intervals, rows)
    c = rng.integers(0, 2, rows)
    n = rng.integers(1, 4, rows).astype(float)
    h = 1.0 / (1.0 + np.exp(-z))
    ll = 0.0
    for i in range(rows):
        s = np.log(1.0 - h[i, : k[i]]).sum()
        ll += n[i] * (
            s + (np.log(h[i, k[i]]) if c[i] == 0 else np.log(1.0 - h[i, k[i]]))
        )
    got = spt.discrete_time_nll(
        torch.as_tensor(z, dtype=F64), k, c=c, n=n, reduction="sum"
    )
    assert float(got) == pytest.approx(-ll, rel=1e-12)
    mean = spt.discrete_time_nll(torch.as_tensor(z, dtype=F64), k, c=c, n=n)
    assert float(mean) == pytest.approx(-ll / n.sum(), rel=1e-12)


def test_discrete_gradient_is_the_finite_difference():
    z = torch.as_tensor(
        np.random.default_rng(2).normal(size=(6, 4)), dtype=F64
    ).requires_grad_(True)
    k, c, n = [0, 3, 1, 2, 3, 0], [0, 1, 0, 0, 1, 1], [1, 2, 1, 3, 1, 2]

    def f(a):
        return float(
            spt.discrete_time_nll(
                torch.as_tensor(a), k, c=c, n=n, reduction="sum"
            )
        )

    spt.discrete_time_nll(z, k, c=c, n=n, reduction="sum").backward()
    base = z.detach().numpy()
    h = 1e-6
    expected = np.zeros_like(base)
    for i in np.ndindex(*base.shape):
        up, down = base.copy(), base.copy()
        up[i] += h
        down[i] -= h
        expected[i] = (f(up) - f(down)) / (2 * h)
    np.testing.assert_allclose(z.grad.numpy(), expected, rtol=1e-6, atol=1e-7)


def test_discrete_refusals_say_what_to_do():
    z = torch.zeros(3, 4, dtype=F64)
    with pytest.raises(ValueError, match="between 0 and 3"):
        spt.discrete_time_nll(z, [0, 1, 4])
    with pytest.raises(ValueError, match="between 0 and 3"):
        spt.discrete_time_nll(z, [0, 1, -1])
    with pytest.raises(TypeError, match="whole numbers"):
        spt.discrete_time_nll(z, [0.5, 1.0, 2.0])
    with pytest.raises(ValueError, match="left and interval"):
        spt.discrete_time_nll(z, [0, 1, 2], c=[0, -1, 2])
    with pytest.raises(ValueError, match="same rows"):
        spt.discrete_time_nll(z, [0, 1])
    with pytest.raises(ValueError, match="reduction"):
        spt.discrete_time_nll(z, [0, 1, 2], reduction="none")
    with pytest.raises(ValueError, match=r"\(N, K\)"):
        spt.discrete_time_nll(z[None], [0])
    # logits shared by every row
    shared = spt.discrete_time_nll(
        torch.zeros(4, dtype=F64), [0, 2, 3], c=[0, 0, 1], reduction="sum"
    )
    assert float(shared) == pytest.approx(8 * np.log(2.0), rel=1e-12)
