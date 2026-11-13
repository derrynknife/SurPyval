"""The semi-parametric Cox model for interval-censored data
(``IntervalCoxPH``): its likelihood, every kind of row, counts, left
truncation, refusals, inference and saving. The comparison with R's
``icenReg::ic_sp`` is in ``tests/reference/test_interval_cox.py``."""

import json
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval import IntervalCoxPH, IntervalCoxPHModel


def _simulated(seed=1, N=120, exact_share=0.2, tl=False):
    """Weibull PH lives (shape 1.5, scale 10, log hazard ratio 0.7 z1 -
    0.4 z2) inspected every uniform(1, 3) from uniform(0, 2): left,
    interval and right censored, some seen exactly; with ``tl`` each
    unit entered at a uniform(0, 3) time it survived to."""
    rng = np.random.default_rng(seed)
    z1 = rng.binomial(1, 0.5, N).astype(float)
    z2 = rng.normal(size=N)
    life = 10.0 * (rng.exponential(size=N) / np.exp(0.7 * z1 - 0.4 * z2)) ** (
        1 / 1.5
    )
    entry = rng.uniform(0.0, 3.0, N) if tl else np.zeros(N)
    keep = life > entry
    x = np.empty((N, 2))
    c = np.empty(N, dtype=int)
    for i in range(N):
        start = entry[i] + rng.uniform(0.0, 2.0)
        visits = np.cumsum(np.r_[start, rng.uniform(1.0, 3.0, 30)])
        visits = visits[visits <= 25.0]
        if rng.random() < exact_share and life[i] < visits[-1]:
            x[i], c[i] = life[i], 0
        elif life[i] <= visits[0]:
            x[i], c[i] = visits[0], -1
        elif life[i] > visits[-1]:
            x[i], c[i] = visits[-1], 1
        else:
            k = np.searchsorted(visits, life[i]) - 1
            x[i], c[i] = (visits[k], visits[k + 1]), 2
    Z = np.column_stack([z1, z2])
    if tl:
        return x[keep], c[keep], Z[keep], entry[keep]
    return x, c, Z


def _log_likelihood(model, x, c, Z, n=None, tl=None):
    """The log-likelihood written out from the model's own predictions:
    each row's probability of what was seen, over its survival to
    entry."""
    n = np.ones(len(c)) if n is None else np.asarray(n, float)
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = np.column_stack([x, x])
    total = 0.0
    for i in range(len(c)):
        z = Z[i : i + 1]
        lo, hi = x[i]
        if c[i] == 0:
            # the drop at the exact time: the survival just before it
            before = model.sf(np.nextafter(lo, -np.inf), z)
            p = before - model.sf(lo, z)
        elif c[i] == 1:
            p = model.sf(lo, z)
        elif c[i] == -1:
            # before hi, and (with an entry) after the entry
            start = 1.0 if tl is None else model.sf(tl[i], z)
            p = start - model.sf(hi, z)
        else:
            p = model.sf(lo, z) - model.sf(hi, z)
        if tl is not None:
            p = p / model.sf(tl[i], z)
        total += n[i] * np.log(p)
    return float(total)


def test_the_likelihood_is_the_models_own():
    x, c, Z = _simulated()
    model = IntervalCoxPH.fit(x, Z, c=c)
    assert model.maximum == "verified"
    assert_allclose(model.log_likelihood, _log_likelihood(model, x, c, Z))


def test_other_coefficients_with_their_best_baseline_are_worse():
    x, c, Z = _simulated(2)
    model = IntervalCoxPH.fit(x, Z, c=c)
    best = _log_likelihood(model, x, c, Z)
    for step in ([0.05, 0.0], [0.0, -0.05], [-0.05, 0.05]):
        other = model.with_params(model.beta + np.array(step))
        assert _log_likelihood(other, x, c, Z) < best


def test_the_coefficients_are_near_the_truth_on_a_large_sample():
    x, c, Z = _simulated(3, N=2000)
    model = IntervalCoxPH.fit(x, Z, c=c)
    se = model.standard_errors()
    assert np.all(np.abs(model.beta - [0.7, -0.4]) < 4 * se)


def test_counts_are_repetition_and_order_does_not_matter():
    x, c, Z = _simulated(4, N=60)
    n = np.where(np.arange(60) % 3 == 0, 2, 1)
    with_counts = IntervalCoxPH.fit(x, Z, c=c, n=n)
    rep = np.repeat(np.arange(60), n)
    repeated = IntervalCoxPH.fit(x[rep], Z[rep], c=c[rep])
    assert_allclose(with_counts.beta, repeated.beta, rtol=1e-9)
    assert_allclose(with_counts.log_likelihood, repeated.log_likelihood)
    order = np.random.default_rng(0).permutation(60)
    shuffled = IntervalCoxPH.fit(x[order], Z[order], c=c[order], n=n[order])
    assert_allclose(shuffled.beta, with_counts.beta, rtol=1e-12)


def test_a_vacuous_entry_time_changes_nothing():
    x, c, Z = _simulated(5, N=80)
    plain = IntervalCoxPH.fit(x, Z, c=c)
    entered = IntervalCoxPH.fit(x, Z, c=c, tl=np.full(80, -1.0))
    assert_allclose(entered.beta, plain.beta, atol=1e-7)
    assert_allclose(entered.log_likelihood, plain.log_likelihood, atol=1e-8)


def test_left_truncation_is_conditional_on_survival_to_entry():
    x, c, Z, tl = _simulated(6, N=150, tl=True)
    model = IntervalCoxPH.fit(x, Z, c=c, tl=tl)
    assert model.maximum == "verified"
    best = _log_likelihood(model, x, c, Z, tl=tl)
    assert_allclose(model.log_likelihood, best)
    for step in ([0.05, 0.0], [0.0, 0.05]):
        other = model.with_params(model.beta + np.array(step))
        assert _log_likelihood(other, x, c, Z, tl=tl) < best


def test_exact_and_right_censored_data_are_close_to_coxph():
    """Without interval rows the model is the grouped-data PH model:
    close to Cox's partial likelihood, and closer with more data."""
    rng = np.random.default_rng(7)
    z = rng.normal(size=(800, 1))
    t = rng.exponential(np.exp(-0.5 * z[:, 0]))
    cens = rng.exponential(1.5, 800)
    x, c = np.minimum(t, cens), (cens < t).astype(int)
    ours = IntervalCoxPH.fit(x, z, c=c)
    cox = sp.CoxPH.fit(x, z, c=c)
    assert abs(ours.beta[0] - cox.beta[0]) < 0.2 * cox.standard_errors()[0]
    assert ours.beta[0] != cox.beta[0]


def test_predictions_and_their_identities():
    x, c, Z = _simulated(8, N=80)
    model = IntervalCoxPH.fit(x, Z, c=c)
    t = np.array([0.5, 3.0, 7.0, 12.0, 40.0])
    rows = np.repeat([[1.0, 0.3]], t.size, 0)
    sf = model.sf(t, rows)
    assert_allclose(sf + model.ff(t, rows), 1.0)
    with np.errstate(divide="ignore"):
        assert_allclose(model.Hf(t, rows), -np.log(sf))
    assert np.all(np.diff(sf) <= 0)
    assert sf[0] == 1.0 or sf[0] <= 1.0
    # S(x | z) = S0(x) ** exp(beta'z)
    s0 = model.sf(t, np.zeros((t.size, 2)))
    assert_allclose(sf, s0 ** np.exp(model.beta @ [1.0, 0.3]))
    assert_allclose(s0, model.baseline_sf(t))
    # The masses on the Turnbull intervals sum to 1
    assert_allclose(model.p0.sum(), 1.0)


def test_center_moves_only_the_baseline():
    x, c, Z = _simulated(9, N=60)
    Z = Z + [0.0, 50.0]
    plain = IntervalCoxPH.fit(x, Z, c=c)
    centred = IntervalCoxPH.fit(x, Z, c=c, center=True)
    assert_allclose(plain.beta, centred.beta)
    q = np.array([[1.0, 50.5], [0.0, 49.0]])
    assert_allclose(plain.sf([5.0, 9.0], q), centred.sf([5.0, 9.0], q))


def test_dataframe_with_interval_columns():
    import pandas as pd

    x, c, Z = _simulated(10, N=60)
    x2 = np.column_stack([x[:, 0], x[:, 1]])
    df = pd.DataFrame(
        {
            "lo": np.where(c == -1, 0.0, x2[:, 0]),
            "hi": np.where(c == 1, np.inf, x2[:, 1]),
            "z1": Z[:, 0],
            "z2": Z[:, 1],
        }
    )
    model = IntervalCoxPH.fit_from_df(
        df, xl_col="lo", xr_col="hi", Z_cols=["z1", "z2"]
    )
    direct = IntervalCoxPH.fit(x, Z, c=c)
    assert_allclose(model.beta, direct.beta, rtol=1e-9)
    assert model.parameter_names == ["z1", "z2"]
    with pytest.raises(ValueError, match="xr_col"):
        IntervalCoxPH.fit_from_df(df, xl_col="lo", Z_cols=["z1"])


def test_refusals_say_what_to_do():
    x, c, Z = _simulated(11, N=40)
    with pytest.raises(ValueError, match="WeibullPH"):
        IntervalCoxPH.fit(x, Z, c=c, tl=np.column_stack([np.zeros(40)] * 2))
    with pytest.raises(ValueError, match="right censored"):
        IntervalCoxPH.fit(x[:, 0], Z, c=np.ones(40))
    model = IntervalCoxPH.fit(x, Z, c=c)
    with pytest.raises(ValueError, match="param_cb"):
        model.cb(5.0, [[0.0, 0.0]])
    with pytest.raises(ValueError, match="method"):
        model.covariance(method="sandwich")


def test_a_separating_covariate_has_no_finite_maximum():
    x, c, Z = _simulated(12, N=60)
    late = Z[:, 0] == 1.0
    x = np.where(late[:, None], x + 100.0, x)
    with pytest.warns(UserWarning, match="No finite maximum"):
        model = IntervalCoxPH.fit(x, Z, c=c)
    assert model.maximum == "no finite maximum"


def test_bounds_by_every_method():
    x, c, Z = _simulated(13, N=100)
    model = IntervalCoxPH.fit(x, Z, c=c)
    wald = model.param_cb("coef_0")
    lr = model.param_cb("coef_0", method="lr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        boot = model.param_cb(
            "coef_0", method="bootstrap", n_boot=50, random_state=3
        )
        again = model.param_cb(
            "coef_0", method="bootstrap", n_boot=50, random_state=3
        )
    for lo, hi in (wald, lr, boot):
        assert lo < model.beta[0] < hi
    assert_allclose(boot, again)
    # The Wald interval is the profile standard error's
    se = model.standard_errors()[0]
    assert_allclose(wald, model.beta[0] + np.array([-1, 1]) * 1.959964 * se)


def test_saving_and_loading():
    x, c, Z = _simulated(14, N=60)
    model = IntervalCoxPH.fit(x, Z, c=c)
    text = json.dumps(model.to_dict(), allow_nan=False)
    restored = IntervalCoxPHModel.from_dict(json.loads(text))
    t = np.array([1.0, 5.0, 30.0])
    rows = np.zeros((3, 2))
    assert_allclose(restored.sf(t, rows), model.sf(t, rows))
    assert_allclose(restored.standard_errors(), model.standard_errors())
    assert restored.maximum == model.maximum
    with pytest.raises(ValueError, match="data the model was fitted to"):
        restored.param_cb("coef_0", method="lr")


def test_diagnose_refuses_with_what_to_use():
    x, c, Z = _simulated(15, N=40)
    model = IntervalCoxPH.fit(x, Z, c=c)
    with pytest.raises(NotImplementedError, match="WeibullPH"):
        model.diagnose()
