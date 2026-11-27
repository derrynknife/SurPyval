"""The edges of the items' spread in a joint model: a correlation of
+-1 between two varying parameters (one an exact multiple of the other
about their means), held on that edge; the adaptive nodes near it; and
the rule that keeps or holds a search off an edge."""

import json
import warnings

import numpy as np
import pytest

import surpyval
from surpyval import ItemLifetime, JointAFT, JointPH, Weibull
from surpyval.degradation import LinearPath
from surpyval.tests.conformance.registry_fixtures import (
    joint_causes_data,
    joint_data,
)
from surpyval.tests.univariate.joint._fleet import simulate_path
from surpyval.univariate.joint import _likelihood as lk
from surpyval.univariate.joint import joint_ph as jp
from surpyval.univariate.joint.joint_model import JointModel

#: The negative log-likelihood the search reached on ``rate_in_level``
#: before a correlation of +-1 was held on its edge: it stalled there,
#: unverified, with corr_a_b -0.998 and the part of b's spread that a's
#: does not explain at 5e-7 a day and still falling.
STALLED = 2432.2863701542383


@pytest.fixture(scope="module")
def rate_in_level():
    """150 items on a straight line, each with a level of its own and
    one rate (a rate spread of 0 in truth), and the fit with both
    varying: its estimated rate spread is a's exactly, times a
    multiple (a correlation of -1)."""
    data = simulate_path(
        LinearPath, [3.0, 1 / 365], [0.5, 0.0], 0.4, n=150, seed=0
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = JointPH(Weibull, LinearPath, varying=("a", "b")).fit(data)
    return data, model, caught


def _arrays_of(data):
    return {
        "x": data.x,
        "c": data.c,
        "n": data.n,
        "markers": [data.markers[i] for i in data.ids],
    }


def test_a_correlation_at_minus_one_is_held_on_its_edge(rate_in_level):
    _, model, caught = rate_in_level
    names = list(model.parameter_names)
    se = dict(zip(names, model.standard_errors()))
    assert model.maximum == "verified"
    assert not [w for w in caught if "verified" in str(w.message)]
    assert model.perfect_correlations == ["corr_a_b"]
    assert model.spreads_at_zero == []
    assert model.params[names.index("corr_a_b")] == -1.0
    assert np.isnan(se["corr_a_b"])
    assert all(np.isfinite(v) for k, v in se.items() if k != "corr_a_b")
    # the new number: above the stalled search's, which it was short of
    assert model.neg_ll() < STALLED
    assert model.neg_ll() == pytest.approx(2432.286303, abs=2e-5)
    assert "corr_a_b at its edge, -1 (b an exact multiple of a" in str(model)


def test_the_held_fit_is_the_full_model_at_its_edge(rate_in_level):
    # The fit on the edge integrates over a alone, b a multiple of it:
    # its likelihood is the limit of the full model's (both integrated)
    # as the part of b's spread that a's does not explain goes to 0.
    data, model, _ = rate_in_level
    arrays_of = _arrays_of(data)
    lik, arrays, u = jp.likelihood_parts(model, **arrays_of)
    held = float(
        lk.neg_ll(u, lik, arrays, lk.place(u, lik, arrays, (None, None))[0])
    )
    assert held == pytest.approx(model.neg_ll(), abs=1e-9)
    free = JointModel.from_dict(dict(model.to_dict()))
    free._proportional = None
    names = list(free.parameter_names)
    for own in (1e-4, 1e-6):
        r = -np.sqrt(1.0 - own**2)
        free.params = np.array(model.params, dtype=float)
        free.params[names.index("corr_a_b")] = r
        lik_f, arrays_f, u_f = jp.likelihood_parts(free, **arrays_of)
        assert lik_f.layout.loaded is None and lik_f.layout.q == 2
        placed, _ = lk.place(u_f, lik_f, arrays_f, (None, None))
        full = float(lk.neg_ll(u_f, lik_f, arrays_f, placed))
        assert full == pytest.approx(held, abs=1e-5)


def test_a_fit_held_on_its_edge_predicts_and_saves(rate_in_level):
    _, model, _ = rate_in_level
    item = ItemLifetime("u", right_censored=300, markers={0: 3.1, 300: 3.9})
    x = np.array([350.0, 500.0])
    sf = model.sf(x, item)
    bounds = model.cb(x, item)
    assert np.all((sf > 0) & (sf < 1)) and np.all(np.isfinite(bounds))
    assert np.all(bounds[:, 0] <= sf) and np.all(sf <= bounds[:, 1])
    restored = surpyval.from_dict(json.loads(json.dumps(model.to_dict())))
    assert restored.perfect_correlations == ["corr_a_b"]
    np.testing.assert_array_equal(restored.sf(x, item), sf)
    np.testing.assert_array_equal(restored.cb(x, item), bounds)


def _edge_pair(model, data):
    """The likelihood of ``model``'s fit with b made a multiple of a
    about their means (``corr_a_b`` at +-1), two ways: the layout that
    integrates over a alone (``lam``), and the full one with b's own
    part of its spread 1e-6 of its sd."""
    lik, arrays, u = jp.likelihood_parts(model, **data)
    theta = jp._theta_of(lik.layout, u)
    L = np.asarray(theta["L"])
    D = L @ L.T
    lam = D[1, 0] / D[0, 0]
    sd_b = np.sqrt(D[1, 1])
    full = jp._proportional_theta(theta, 1, 0, 1e-6 * sd_b)
    u_full = lik.layout.pack(full)
    edge = jp._with_proportional(lik, 1, 0, float(np.sign(lam)))
    u_edge = edge.layout.pack(dict(jp._project(theta, [0]), lam=lam))

    def value(m, v):
        placed, _ = lk.place(v, m, arrays, (None, None))
        return float(lk.neg_ll(v, m, arrays, placed))

    return value(lik, u_full), value(edge, u_edge)


@pytest.mark.parametrize(
    "fitter, data, how",
    [
        (JointPH(Weibull, LinearPath), joint_data, "grid"),
        (JointPH(Weibull, LinearPath), joint_data, "laplace"),
        (JointPH(Weibull, LinearPath, link="slope"), joint_data, "grid"),
        (JointAFT(Weibull, LinearPath), joint_data, "grid"),
        (JointPH(Weibull, LinearPath), joint_causes_data, "grid"),
    ],
    ids=["value", "laplace", "slope", "AFT", "modes"],
)
def test_the_edge_layout_is_the_limit_of_the_full_one(fitter, data, how):
    # every survival part and link the hold may meet: the likelihood
    # with b a multiple of a equals the full model's at that edge
    d = data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fitter.fit(**d, integration=how)
    full, edge = _edge_pair(model, d)
    assert edge == pytest.approx(full, abs=1e-5)


def test_the_reported_spread_of_the_edge_layout():
    # b = mu_b + lam (a - mu_a): sd_b = |lam| sd_a, corr_a_b = sign(lam)
    from surpyval.univariate.joint.joint_model import (
        internal_theta,
        reported_params,
    )

    layout = lk.Layout(2, 0, (0,), loaded=(1, 0, 1, -1.0))
    theta = {
        "mu": np.array([3.0, 0.01]),
        "Delta": np.zeros((2, 0)),
        "L": np.array([[0.5]]),
        "lam": -0.004,
        "sigma": 0.4,
        "alpha": 800.0,
        "beta": 2.0,
        "gamma": np.zeros(0),
        "zeta": np.array([0.9]),
    }
    refs = (4.0, np.zeros(0), 4.0)
    out = np.asarray(reported_params(theta, layout, refs))
    # a, b, sd_a, sd_b, corr_a_b, sd_noise, alpha, beta, effect
    np.testing.assert_allclose(out[2:5], [0.5, 0.002, -1.0], rtol=1e-15)
    est = np.delete(out, 4)  # the correlation is not estimated
    back = internal_theta(est, layout, refs)
    assert float(back["lam"]) == pytest.approx(-0.004, rel=1e-14)
    np.testing.assert_allclose(back["L"], [[0.5]], rtol=1e-15)


def _model_of(params):
    """``joint_data``'s quadratic fit with a, b and c varying, at the
    reported ``params`` (as a dict would save it)."""
    names = [
        "a", "a_coef_0", "b", "b_coef_0", "c", "c_coef_0",
        "sd_a", "sd_b", "sd_c", "corr_a_b", "corr_a_c", "corr_b_c",
        "sd_noise", "alpha", "beta", "coef_0", "marker_effect",
    ]  # fmt: skip
    return JointModel.from_dict(
        {
            "model": "JointModel",
            "schema": 1,
            "distribution": "Weibull",
            "path_model": "quadratic",
            "links": None,
            "varying": ["a", "b", "c"],
            "link": ["value"],
            "survival": "PH",
            "params": list(params),
            "parameter_names": names,
            "feature_names": ["coef_0"],
            "aliased": [],
            "path_covariates": None,
            "hazard_covariates": None,
            "marker_reference": 3.5946738609112705,
            "centre": 3.5946738609112705,
            "covariate_means": [0.5116279069767442],
            "n_items": 43.0,
            "n_events": 27.0,
            "n_readings": 417.0,
            "integration": "grid",
            "_neg_ll": None,
            "maximum": "unverified",
        }
    )


#: Where the search on ``joint_data``'s quadratic path with a, b and c
#: varying stopped before the nodes near the edge were placed in the
#: prior's coordinates (corr_a_b 0.997, within 1e-10 of a singular
#: spread): the likelihood by the 7-point nodes was 406.94 there, 422.5
#: by 15 and 437.8 by 31, against 494.19 at corr_a_b 1e-10 lower and by
#: Monte Carlo from the prior.
NEAR_EDGE = [
    2.964896051540007, -0.15025677977686658, 0.0037809177343823136,
    -0.00040104922095996273, -6.467728028585037e-07,
    5.748119709961593e-08, 0.5053730978880288, 0.0010508034441644863,
    1.4045541377100887e-06, 0.99718809587453, -0.27686413041233854,
    -0.20407574376010434, 0.38383381169965924, 6154.561592768178,
    0.8345892299339277, 0.6556922606203313, 1.3665014579557258,
]  # fmt: skip


def test_the_nodes_near_a_singular_spread_integrate_it():
    model = _model_of(NEAR_EDGE)
    lik, arrays, u = jp.likelihood_parts(model, **joint_data())
    values = []
    for points in (7, 15, 25):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(lk, "GH_FIT", points)
            placed, _ = lk.place(u, lik, arrays, (None, None))
            values.append(float(lk.neg_ll(u, lik, arrays, placed)))
    # the grid's answer no longer depends on its size...
    assert max(values) - min(values) < 1e-6
    # ... and is the integral: item 38's by Monte Carlo from the prior
    # (200,000 draws, seed 1; effective sample 6,381) is -9.374 +- 0.01
    theta = lk.numeric_theta(lik.layout.unpack(u))
    one = arrays.take(np.arange(len(arrays.w)) == 38)
    placed, _ = lk.place(u, lik, one, (None, None))
    gh = float(
        lk.log_integrals(
            theta, lik, one, placed[0], one.T, one.d, True, one.U
        )[0]
    )
    rng = np.random.default_rng(1)
    V = list(lik.layout.varying)
    mean = theta["mu"][V] + one.Z[0] @ theta["Delta"][V].T
    b = mean + rng.standard_normal((200_000, 3)) @ theta["L"].T
    log_prior = lk.log_prior(theta, lik.layout, one.Z, b[None])
    terms = lk.log_integrand(
        theta, lik, one, (b[None], -log_prior), one.T, one.d, True, one.U
    )[0]
    from scipy.special import logsumexp

    mc = float(logsumexp(terms) - np.log(b.shape[0]))
    assert gh == pytest.approx(mc, abs=0.05)
    assert gh < -9.0  # the 7 points' nodes as they were placed: -7.60


class _Stub:
    def __init__(self, layout):
        self.layout = layout


def _rule(monkeypatch, first, nested_value, off_value):
    """:func:`jp._held_or_off` with the searches stubbed: the first
    answer's value ``first``, the fit on the edge's ``nested_value``,
    and the whole model's search off it reaching ``off_value``; returns
    what it chose and where the search off the edge started."""
    layout = lk.Layout(2, 0, (0, 1))
    theta = {
        "mu": np.array([3.0, 0.01]),
        "Delta": np.zeros((2, 0)),
        "L": np.array([[0.5, 0.0], [0.001, 0.002]]),
        "sigma": 0.4,
        "alpha": 800.0,
        "beta": 2.0,
        "gamma": np.zeros(0),
        "zeta": np.array([0.9]),
    }
    u1 = layout.pack(theta)
    off_u = u1 + 0.5
    starts = []

    def fake(model, arrays, start):
        starts.append(layout.pack(start))
        return (u1 + 1.0, off_value, None, "unverified"), []

    monkeypatch.setattr(jp, "_maximised", fake)
    nested = _Stub(layout)
    got = ((u1 - 1.0, nested_value, None, "verified"), [])
    chosen = jp._held_or_off(
        _Stub(layout), None, (u1, first), nested, got, off_u
    )
    return chosen, starts[0], u1, off_u


def test_the_rise_off_an_edge_is_judged_in_absolute_terms(monkeypatch):
    # at a log-likelihood of 1e5 a rise of 0.05 is kept: one part in a
    # million of it (0.1) would have held the fit on the edge
    chosen, start, _, off_u = _rule(monkeypatch, 1e5 + 1.0, 1e5, 1e5 - 0.05)
    assert chosen is not None and chosen[1][1] == 1e5 - 0.05
    np.testing.assert_allclose(start, off_u)
    # a rise below 1e-4 is round-off: held on the edge
    chosen, _, _, _ = _rule(monkeypatch, 1e5 + 1.0, 1e5, 1e5 - 5e-5)
    assert chosen[1][1] == 1e5


def test_the_search_off_an_edge_starts_from_a_first_answer_above_it(
    monkeypatch,
):
    # the first answer already beats the fit on the edge: the search off
    # it starts there, not from the edge, and the better answer is kept
    chosen, start, u1, _ = _rule(monkeypatch, 99.0, 100.0, 98.5)
    np.testing.assert_allclose(start, u1)
    assert chosen[1][1] == 98.5
    # never worse than the first answer: it stands where the search
    # from it reaches less
    chosen, _, _, _ = _rule(monkeypatch, 99.0, 100.0, 99.5)
    assert chosen is None


def test_a_correlation_on_its_edge_is_not_tested_against_a_free_one(
    rate_in_level,
):
    _, model, _ = rate_in_level
    free = JointModel.from_dict(dict(model.to_dict()))
    free._proportional = None
    with pytest.raises(ValueError, match=r"corr_a_b\) against a free one"):
        model.likelihood_ratio_test(free)
    with pytest.raises(ValueError, match="held at"):
        free.likelihood_ratio_test(model)


def test_a_spread_beyond_the_valid_ones_is_refused_clearly():
    # a dict edited by hand: a correlation just past 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = JointPH(Weibull, LinearPath).fit(**joint_data())
    names = list(model.parameter_names)
    d = model.to_dict()
    d.pop("covariance")
    d["params"][names.index("corr_a_b")] = 1.0 + 1e-9
    edited = JointModel.from_dict(d)
    with pytest.raises(ValueError, match="not a valid covariance matrix"):
        edited.sf([400.0], [0.0], markers={0: 3.1, 300: 3.9}, given=300)
