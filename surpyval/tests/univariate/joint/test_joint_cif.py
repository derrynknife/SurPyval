"""The chance of each failure mode for a running item: a joint model's
``predict(...).cif(x, mode)``, its bounds and its saved form.

The checks: the modes' chances and the survival add to 1; with the marker
effects 0 they are ``ParametricCompetingRisks``' incidences from the age
now; they agree with a brute-force two-dimensional integral over the
item's path from the model's parameters alone; the bounds hold the
estimate and stay in (0, 1); the saved prediction carries them; and the
calls refuse what they cannot do, saying what to use. The calibration of
the bounds is ``tests/calibration/test_coverage_joint_cif.py``; the R
``JM`` comparison is ``test_against_r_jm`` (it skips until its reference
values exist).
"""

import json
import pathlib
import warnings

import numpy as np
import pytest

from surpyval import ParametricCompetingRisks, Weibull, WeibullJointPH
from surpyval.tests.univariate.joint.test_joint_causes import (
    _model,
    _simpson,
    _with_params,
)

#: An item read four times, still running at 400.
ITEM = {
    "markers": {0.0: 3.1, 120.0: 3.4, 240.0: 3.6, 360.0: 3.9},
    "given": 400.0,
}
X = np.array([100.0, 400.0, 450.0, 600.0, 900.0, 1500.0])


@pytest.fixture(scope="module")
def model():
    return _model()


@pytest.fixture(scope="module")
def pred(model):
    return model.predict(**ITEM)


def test_the_chances_and_the_survival_add_to_one(model, pred):
    every = sum(pred.cif(X, mode) for mode in model.causes)
    assert np.max(np.abs(every + pred.sf(X) - 1.0)) <= 1e-10
    assert np.allclose(pred.total_cif(X), pred.ff(X), rtol=0, atol=0)
    assert np.allclose(pred.cif(X), pred.ff(X), rtol=0, atol=0)


def test_no_chance_up_to_the_age_now(model, pred):
    for mode in model.causes:
        assert np.all(pred.cif(X[X <= 400.0], mode) == 0.0)
        assert np.all(np.diff(pred.cif(X, mode)) >= 0)
    assert pred.cif(450.0, "bearing").shape == ()
    assert pred.cif([[450.0, 600.0]], "seal").shape == (1, 2)


def test_with_no_marker_effects_it_is_parametric_competing_risks(model):
    """With the marker effects 0 the modes' hazards do not depend on the
    path: each chance is ``ParametricCompetingRisks``' cumulative
    incidence from the age now, ``(CIF_k(x) - CIF_k(u)) / S(u)``."""
    flat = _with_params(
        model, **{"bearing: marker_effect": 0.0, "seal: marker_effect": 0.0}
    )
    p = dict(zip(flat.parameter_names, flat.params))
    crm = ParametricCompetingRisks.from_fitted(
        {
            cause: Weibull.from_params(
                [p[f"{cause}: alpha"], p[f"{cause}: beta"]]
            )
            for cause in flat.causes
        }
    )
    u = ITEM["given"]
    pred = flat.predict(**ITEM)
    after = X[X > u]
    for mode in flat.causes:
        expected = (crm.cif(after, mode) - crm.cif(u, mode)) / crm.sf(u)
        # 1.1e-11 relative is reached; 1e-9 leaves a margin
        assert pred.cif(after, mode) == pytest.approx(expected, rel=1e-9)


def _brute_force_cif(model, readings, u, x, cause):
    """The chance of failing by ``cause`` by each age of ``x``, from the
    model's parameters alone: a dense grid over the item's level and
    rate (121 points a dimension, 7 sds each way) weighted by the
    readings, the prior and the survival to ``u``; at each, the mode's
    hazard times the survival of both modes from ``u``, by Simpson's
    rule on 801 ages (the cumulative hazard to each by Simpson on the
    pairs before it)."""
    p = dict(zip(model.parameter_names, model.params))
    v0, v1, r = p["sd_a"] ** 2, p["sd_b"] ** 2, p["corr_a_b"]
    D = np.array([[v0, r * np.sqrt(v0 * v1)], [r * np.sqrt(v0 * v1), v1]])
    g0 = np.linspace(-7, 7, 121) * np.sqrt(v0)
    g1 = np.linspace(-7, 7, 121) * np.sqrt(v1)
    B0, B1 = (a.ravel() for a in np.meshgrid(g0, g1, indexing="ij"))
    b = np.stack([B0, B1], 1)
    log_prior = -0.5 * np.einsum("ni,ij,nj->n", b, np.linalg.inv(D), b)
    m0, m1 = p["a"] + B0, p["b"] + B1

    def hazard(c, t):
        t = np.atleast_1d(t)[None, :]
        alpha, beta = p[f"{c}: alpha"], p[f"{c}: beta"]
        h0 = (beta / alpha) * (t / alpha) ** (beta - 1)
        m = m0[:, None] + m1[:, None] * t
        return h0 * np.exp(
            p[f"{c}: marker_effect"] * (m - model.marker_reference)
        )

    def total(t):
        return sum(hazard(c, t) for c in model.causes)

    def cumhaz(lo, hi):
        s = np.linspace(lo, hi, 1001)
        return _simpson(total(s), s[1] - s[0])

    ages = np.array(list(readings), dtype=float)
    ys = np.array(list(readings.values()), dtype=float)
    resid = ys[None, :] - m0[:, None] - m1[:, None] * ages[None, :]
    log_w = log_prior - 0.5 * np.sum(resid**2, axis=1) / p["sd_noise"] ** 2
    log_w = log_w - cumhaz(0.0, u)
    w = np.exp(log_w - log_w.max())
    w = w / w.sum()
    out = []
    for hi in x:
        s = np.linspace(u, hi, 801)
        step = s[1] - s[0]
        h = total(s)
        pairs = (h[:, 0:-2:2] + 4 * h[:, 1:-1:2] + h[:, 2::2]) * step / 3
        H = np.concatenate([np.zeros((B0.size, 1)), np.cumsum(pairs, 1)], 1)
        even = s[::2]
        g = hazard(cause, even) * np.exp(-H)
        out.append(float(w @ _simpson(g, even[1] - even[0])))
    return np.array(out)


@pytest.mark.parametrize("moved", [False, True], ids=["fitted", "moved"])
def test_it_agrees_with_a_brute_force_integral(model, moved):
    if moved:
        # both modes following the marker, one of them down
        model = _with_params(
            model,
            **{
                "bearing: marker_effect": 1.4,
                "seal: marker_effect": -0.5,
                "seal: beta": 2.6,
                "corr_a_b": -0.5,
            },
        )
    pred = model.predict(**ITEM)
    after = np.array([450.0, 600.0, 900.0, 1500.0])
    for mode in model.causes:
        brute = _brute_force_cif(model, ITEM["markers"], 400.0, after, mode)
        assert pred.cif(after, mode) == pytest.approx(
            brute, rel=1e-3, abs=2e-5
        )


def test_the_bounds_hold_the_estimate(model, pred):
    after = np.array([450.0, 600.0, 900.0, 1500.0])
    for mode in model.causes:
        est = pred.cif(after, mode)
        band = pred.cb(after, on="cif", mode=mode)
        assert band.shape == (4, 2)
        assert np.all(band[:, 0] < est) and np.all(est < band[:, 1])
        assert np.all((band > 0) & (band < 1))
        lower = pred.cb(after, on="cif", mode=mode, bound="lower")
        assert lower == pytest.approx(
            pred.cb(after, on="cif", mode=mode, alpha_ci=0.1)[:, 0]
        )
    before = pred.cb([100.0, 400.0], on="cif", mode="seal")
    assert np.all(before == 0.0)
    # no mode: the bounds on ff
    assert np.allclose(pred.cb(after, on="cif"), pred.cb(after, on="ff"))


def test_the_bounds_are_the_delta_method_on_the_cifs_scale(model, pred):
    """The width on ``log(-log(1 - cif))``, by central differences of
    the estimate in each parameter, the nodes placed afresh: as the
    autograd derivative with the nodes held, to 1%."""
    from scipy.stats import norm

    x = np.array([900.0])
    est = np.asarray(model.params, dtype=float)
    keep = model._estimated
    cov = model.covariance()[np.ix_(keep, keep)]
    names = np.array(model.parameter_names)[keep]
    grads = []
    for j, name in enumerate(names):
        h = 1e-5 * max(abs(est[keep][j]), 1e-3)
        values = []
        for sign in (1.0, -1.0):
            moved = _with_params(model, **{name: est[keep][j] + sign * h})
            c = float(moved.predict(**ITEM).cif(x, "bearing")[0])
            values.append(np.log(-np.log1p(-c)))
        grads.append((values[0] - values[1]) / (2 * h))
    g = np.array(grads)
    se = float(np.sqrt(g @ cov @ g))
    c = float(pred.cif(x, "bearing")[0])
    centre = np.log(-np.log1p(-c))
    q = norm.ppf(0.975)
    expected = -np.expm1(-np.exp(centre + np.array([-q, q]) * se))
    assert pred.cb(x, on="cif", mode="bearing")[0] == pytest.approx(
        expected, rel=1e-2
    )


def test_the_saved_prediction_carries_the_modes(model, pred):
    d = pred.to_dict(x=[450.0, 900.0])
    json.dumps(d, allow_nan=False)
    assert set(d["cif"]) == {"bearing", "seal"}
    for mode in model.causes:
        entry = d["cif"][mode]
        assert entry["estimate"] == pytest.approx(
            pred.cif([450.0, 900.0], mode).tolist()
        )
        assert entry["lower"][1] < entry["estimate"][1] < entry["upper"][1]
    total = np.add(
        d["cif"]["bearing"]["estimate"], d["cif"]["seal"]["estimate"]
    )
    assert total + np.array(d["sf"]) == pytest.approx(1.0, abs=1e-10)


def test_a_model_of_one_way_of_failing(model):
    from surpyval.tests.univariate.joint._fleet import fleet

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = WeibullJointPH.fit(fleet(n=60, seed=5))
    p = one.predict(**ITEM)
    assert p.modes is None
    assert np.allclose(p.cif(X), p.ff(X))
    assert p.to_dict(x=[450.0])["cif"] is None
    with pytest.raises(ValueError, match="one way of failing"):
        p.cif(X, "bearing")
    with pytest.raises(ValueError, match="one way of failing"):
        p.cb(X, on="cif", mode="bearing")


def test_refusals_say_what_to_use(pred):
    with pytest.raises(ValueError, match="'mode' is one of"):
        pred.cif(500.0, "gearbox")
    with pytest.raises(ValueError, match="on='cif'"):
        pred.cb(500.0, on="sf", mode="seal")


def test_an_infinite_age_is_nan(pred):
    assert np.isnan(pred.cif(np.inf, "seal"))
    assert np.isnan(pred.cif(np.nan, "seal"))


REFERENCE = (
    pathlib.Path(__file__).parents[2] / "reference" / "data" / "r_jm_cif.json"
)


@pytest.mark.skipif(
    not REFERENCE.exists(),
    reason="R JM reference not generated (scripts/reference/"
    "reference_r_jm_cif.R; needs the R package JM)",
)
def test_against_r_jm():
    """PBC (death and transplant as two modes, log serum bilirubin the
    marker): JM's ``survfitJM`` cumulative incidences for patients still
    alive, against this model's. JM fits by a different quadrature and a
    piecewise-constant baseline, so the comparison is of the incidences
    to 0.03."""
    import pandas as pd

    from surpyval import ItemLifetime, SurpyvalData

    ref = json.loads(REFERENCE.read_text())
    long = pd.DataFrame(ref["data"])
    items = []
    for pid, rows in long.groupby("id"):
        first = rows.iloc[0]
        outcome = (
            {"observed": float(first["years"])}
            if first["status"] != "alive"
            else {"right_censored": float(first["years"])}
        )
        items.append(
            ItemLifetime(
                int(pid),
                **outcome,
                event_type=(
                    None if first["status"] == "alive" else first["status"]
                ),
                markers=dict(zip(rows["year"], np.log(rows["serBilir"]))),
            )
        )
    fitted = WeibullJointPH.fit(SurpyvalData.from_items(items))
    for case in ref["predictions"]:
        rows = long[long["id"] == case["id"]]
        readings = {
            a: v
            for a, v in zip(rows["year"], np.log(rows["serBilir"]))
            if a <= case["given"]
        }
        p = fitted.predict(markers=readings, given=case["given"])
        for mode, values in case["cif"].items():
            assert p.cif(case["x"], mode) == pytest.approx(values, abs=0.03)
