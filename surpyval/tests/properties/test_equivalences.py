"""Two ways of writing the same data give the same fit.

The conformance suite checks one option at a time (``registry_cases``)
and the hypothesis properties check order, counts and validity. These
tests add the equivalences between the *forms* the data can be written
in, and the combinations of the parametric options:

- a censoring flag against the same rows written as intervals
  (``c = 1`` is ``[x, inf]``, ``c = -1`` is ``[lo, x]``);
- a truncation bound at the edge of the support against none;
- integer weights against counts against repeated rows, with left
  truncation;
- ``offset``, ``lfp`` and ``zi`` together.

Each reference is the other form of the same fit, so no value is
pinned here; the tolerances are the agreement seen at commit e3412770
with margin, given beside each assertion.
"""

import json
import pickle
import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import (
    CoxPH,
    Exponential,
    Gamma,
    Gumbel,
    KaplanMeier,
    Logistic,
    LogLogistic,
    LogNormal,
    NelsonAalen,
    Normal,
    Turnbull,
    Weibull,
    WeibullAFT,
    WeibullPH,
)

SEEDS = [2, 3, 4]


def _name(v):
    return getattr(v, "name", str(v))


def _quiet(fit, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fit(*args, **kwargs)


def _data(seed):
    """The A1-A5 data, drawn in this order."""
    rng = np.random.default_rng(seed)
    x = np.round(rng.weibull(2.0, 40) * 100, 1)
    c = rng.choice([0, 1, -1], size=40, p=[0.5, 0.3, 0.2])
    Z = rng.normal(size=(40, 2))
    n = rng.integers(1, 4, size=40)
    tl = np.where(rng.random(40) < 0.3, x * 0.5, -np.inf)
    return x, c, Z, n, tl


def _as_intervals(x, c, lo):
    xl = np.where(c == -1, lo, x)
    xr = np.where(c == 1, np.inf, x)
    return np.c_[xl, xr], np.where(c == 0, 0, 2)


# --- A1-A3: a censoring flag against the interval form -------------------

POSITIVE = [Weibull, LogNormal, Gamma, Exponential, LogLogistic]
REAL_LINE = [Normal, Gumbel, Logistic]


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize(
    "dist, lo",
    [(d, 0.0) for d in POSITIVE] + [(d, -np.inf) for d in REAL_LINE],
    ids=_name,
)
def test_flags_equal_intervals_parametric(dist, lo, seed):
    x, c, *_ = _data(seed)
    X, c2 = _as_intervals(x, c, lo)
    a = _quiet(dist.fit, x, c=c)
    b = _quiet(dist.fit, X, c=c2)
    # Observed: neg_ll equal to 1e-12, params to 1.6e-7 relative.
    assert a.neg_ll() == pytest.approx(b.neg_ll(), abs=1e-8)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-5)


@pytest.mark.parametrize("seed", SEEDS)
def test_flags_equal_intervals_turnbull(seed):
    x, c, *_ = _data(seed)
    X, c2 = _as_intervals(x, c, 0.0)
    q = np.linspace(1, 300, 50)
    a = _quiet(Turnbull.fit, x, c=c)
    b = _quiet(Turnbull.fit, X, c=c2)
    # Observed: 3e-13.
    np.testing.assert_allclose(a.sf(q), b.sf(q), rtol=0, atol=1e-10)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("model", [WeibullPH, WeibullAFT], ids=_name)
def test_flags_equal_intervals_regression(model, seed):
    x, c, Z, *_ = _data(seed)
    X, c2 = _as_intervals(x, c, 0.0)
    a = _quiet(model.fit, x, Z, c)
    b = _quiet(model.fit, X, Z, c2)
    # Observed: PH 1.3e-9, AFT 6e-5 relative (the optimiser's tolerance).
    assert a.neg_ll() == pytest.approx(b.neg_ll(), abs=1e-6)
    np.testing.assert_allclose(a.params, b.params, rtol=1e-3)


# --- A4: truncation at the edge of the support is no truncation ----------

NO_OP_TRUNCATIONS = {
    "tl=0": dict(tl=0.0),
    "tr=inf": dict(tr=np.inf),
    "tl=-inf,tr=inf": dict(tl=-np.inf, tr=np.inf),
    "tl=1e-300": dict(tl=np.full(40, 1e-300)),
}


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("trunc", NO_OP_TRUNCATIONS, ids=_name)
@pytest.mark.parametrize("dist", [Weibull, LogNormal], ids=_name)
def test_no_op_truncation_parametric(dist, trunc, seed):
    x, c, *_ = _data(seed)
    cr = np.where(c == -1, 0, c)
    a = _quiet(dist.fit, x, c=cr)
    b = _quiet(dist.fit, x, c=cr, **NO_OP_TRUNCATIONS[trunc])
    # Observed: exactly equal.
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12)
    assert a.neg_ll() == pytest.approx(b.neg_ll(), rel=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("np_model", [KaplanMeier, Turnbull], ids=_name)
def test_no_op_truncation_nonparametric(np_model, seed):
    x, c, *_ = _data(seed)
    cr = np.where(c == -1, 0, c)
    q = np.array([20.0, 50.0, 100.0])
    a = _quiet(np_model.fit, x, c=cr)
    b = _quiet(np_model.fit, x, c=cr, tl=np.zeros(40))
    # Observed: 0 (Kaplan-Meier) and 1.2e-14 (Turnbull).
    np.testing.assert_allclose(a.sf(q), b.sf(q), rtol=0, atol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
def test_no_op_truncation_cox(seed):
    x, c, Z, *_ = _data(seed)
    cr = np.where(c == -1, 0, c)
    a = _quiet(CoxPH.fit, x, Z, cr)
    b = _quiet(CoxPH.fit, x, Z, cr, tl=np.zeros(40))
    # Observed: 0.
    np.testing.assert_allclose(a.params, b.params, rtol=0, atol=1e-12)


# --- A5: integer weights = counts = repeated rows, with truncation -------


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("dist", [Weibull, LogNormal, Exponential], ids=_name)
def test_weights_counts_rows_truncated_parametric(dist, seed):
    x, c, _, n, tl = _data(seed)
    cr = np.where(c == -1, 0, c)
    counted = _quiet(dist.fit, x, c=cr, n=n, tl=tl)
    weighted = _quiet(dist.fit, x, c=cr, w=n.astype(float), tl=tl)
    repeated = _quiet(
        dist.fit,
        np.repeat(x, n),
        c=np.repeat(cr, n),
        tl=np.repeat(tl, n),
    )
    # Observed: weights within 1e-6, repeated rows exactly equal.
    np.testing.assert_allclose(counted.params, weighted.params, rtol=1e-5)
    np.testing.assert_allclose(counted.params, repeated.params, rtol=1e-12)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("np_model", [KaplanMeier, NelsonAalen], ids=_name)
def test_weights_counts_truncated_nonparametric(np_model, seed):
    x, c, _, n, tl = _data(seed)
    cr = np.where(c == -1, 0, c)
    q = np.array([20.0, 50.0, 100.0])
    counted = _quiet(np_model.fit, x, c=cr, n=n, tl=tl)
    weighted = _quiet(np_model.fit, x, c=cr, w=n.astype(float), tl=tl)
    np.testing.assert_allclose(
        counted.sf(q), weighted.sf(q), rtol=0, atol=1e-12
    )


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("truncated", [False, True], ids=["no-tl", "tl"])
def test_weights_counts_cox_breslow(truncated, seed):
    # Breslow ties only: with Efron ties a weighted fit differs from
    # repeated rows by design (DECISIONS.md, "Ties in a weighted Cox fit").
    x, c, Z, n, tl = _data(seed)
    cr = np.where(c == -1, 0, c)
    extra = dict(tl=tl) if truncated else {}
    counted = _quiet(CoxPH.fit, x, Z, cr, n=n, tie_method="breslow", **extra)
    weighted = _quiet(
        CoxPH.fit,
        x,
        Z,
        cr,
        w=n.astype(float),
        tie_method="breslow",
        **extra,
    )
    q = np.array([20.0, 50.0, 100.0])
    # Observed: 8e-17 (params) and 0 (sf).
    np.testing.assert_allclose(
        counted.params, weighted.params, rtol=0, atol=1e-12
    )
    np.testing.assert_allclose(
        counted.sf(q, Z[:1]), weighted.sf(q, Z[:1]), rtol=0, atol=1e-12
    )


# --- A6: offset, lfp and zi together -------------------------------------

COMBOS = {
    "offset+lfp": dict(offset=True, lfp=True),
    "lfp+zi": dict(lfp=True, zi=True),
    "offset+zi": dict(offset=True, zi=True),
    "offset+lfp+zi": dict(offset=True, lfp=True, zi=True),
}
Q = np.array([5.0, 12.0, 30.0, 60.0, 200.0])


def _combo_data():
    rng = np.random.default_rng(7)
    N = 120
    life = rng.weibull(1.8, N) * 50 + 10
    immune = rng.random(N) < 0.25
    doa = rng.random(N) < 0.10
    x = life.copy()
    x[doa] = 0.0
    c = np.zeros(N, int)
    stopped = (x > 70.0) | immune
    c[stopped] = 1
    x[stopped] = 70.0
    n = rng.integers(1, 4, size=N)
    return x, c, n


_FITS: dict = {}


def _combo_fit(dist, combo):
    key = (dist.name, combo)
    if key not in _FITS:
        x, c, n = _combo_data()
        kw = COMBOS[combo]
        xx = x if kw.get("zi") else np.where(x == 0, 0.5, x)
        _FITS[key] = (xx, c, n, _quiet(dist.fit, xx, c=c, n=n, **kw))
    return _FITS[key]


COMBO_CASES = [
    pytest.param(d, k, id=f"{d.name}-{k}")
    for d in [Weibull, Exponential, LogNormal]
    for k in COMBOS
]


@pytest.mark.parametrize("dist, combo", COMBO_CASES)
def test_combo_identities(dist, combo):
    _, _, _, m = _combo_fit(dist, combo)
    kw = COMBOS[combo]
    sf, ff = m.sf(Q), m.ff(Q)
    np.testing.assert_allclose(sf + ff, 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m.Hf(Q), -np.log(sf), rtol=1e-9)
    np.testing.assert_allclose(m.df(Q), m.hf(Q) * sf, rtol=1e-9)
    inside = (ff > m.f0) & (ff < m.lfp_p)
    assert inside.any()
    np.testing.assert_allclose(m.qf(ff[inside]), Q[inside], rtol=1e-9)
    if kw.get("lfp"):
        assert 0 < m.lfp_p <= 1
    else:
        assert m.lfp_p == 1
    if kw.get("zi"):
        assert 0 < m.f0 < 1
    else:
        assert m.f0 == 0


@pytest.mark.parametrize("dist, combo", COMBO_CASES)
def test_combo_quantile_limits(dist, combo):
    _, _, _, m = _combo_fit(dist, combo)
    p = np.array([0.0, 0.01, 0.05, 0.3, 0.5, 0.79, 0.85, 0.95, 0.999])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        qp = m.qf(p)
    assert np.all(np.isinf(qp[p >= m.lfp_p]))
    if COMBOS[combo].get("zi"):
        # The point mass at zero; without zi, qf(0) is the offset gamma.
        assert np.all(qp[p <= m.f0] == 0)
    else:
        assert qp[0] == pytest.approx(m.gamma, abs=1e-9)


@pytest.mark.parametrize("dist, combo", COMBO_CASES)
def test_combo_counts_weights_units(dist, combo):
    xx, c, n, m = _combo_fit(dist, combo)
    kw = COMBOS[combo]
    repeated = _quiet(dist.fit, np.repeat(xx, n), c=np.repeat(c, n), **kw)
    np.testing.assert_allclose(repeated.params, m.params, rtol=1e-5)
    weighted = _quiet(dist.fit, xx, c=c, w=n.astype(float), **kw)
    np.testing.assert_allclose(weighted.params, m.params, rtol=1e-5)
    scaled = _quiet(dist.fit, xx * 24, c=c, n=n, **kw)
    np.testing.assert_allclose(scaled.sf(Q * 24), m.sf(Q), rtol=1e-6)


@pytest.mark.parametrize("dist, combo", COMBO_CASES)
def test_combo_round_trips(dist, combo):
    _, _, _, m = _combo_fit(dist, combo)
    via_json = surv.from_dict(json.loads(json.dumps(m.to_dict())))
    via_pickle = pickle.loads(pickle.dumps(m))
    np.testing.assert_array_equal(via_json.sf(Q), m.sf(Q))
    np.testing.assert_array_equal(via_pickle.sf(Q), m.sf(Q))


@pytest.mark.parametrize("dist, combo", COMBO_CASES)
def test_combo_random_and_mean(dist, combo):
    _, _, _, m = _combo_fit(dist, combo)
    kw = COMBOS[combo]
    r = m.random(20000, random_state=1)
    has_lfp = kw.get("lfp") and m.lfp_p < 1
    if has_lfp:
        assert np.isinf(r).mean() == pytest.approx(1 - m.lfp_p, abs=0.02)
        assert np.isinf(m.mean())
    else:
        assert not np.isinf(r).any()
    if kw.get("zi"):
        assert (r == 0).mean() == pytest.approx(m.f0, abs=0.02)
    else:
        assert not (r == 0).any()


def test_combo_lfp_zi_weibull_values():
    # Observed at commit e3412770 on this data; a regression pin.
    _, _, _, m = _combo_fit(Weibull, "lfp+zi")
    assert m.lfp_p == pytest.approx(0.7929, abs=1e-3)
    assert m.f0 == pytest.approx(0.0742, abs=1e-3)
