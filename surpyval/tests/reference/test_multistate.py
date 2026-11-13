"""The multi-state models (``surpyval.multistate``) against R ``survival``
and ``mstate``.

The values are in ``data/r_multistate.json``, written by
``scripts/reference/reference_r_multistate.R`` (R 4.6.2, survival 3.8.12,
mstate 0.3.3), on two fixtures of ``fixtures.json``:

- "mgus2_multistate": ``survival::mgus2`` as an illness-death model (MGUS
  -> PCM -> death, MGUS -> death), ``load_mgus2(spells=True)``;
- "multistate": 300 units simulated by ``make_fixtures.py`` from an
  illness-death model with recovery, with delayed entry and tied times.

What is compared:

- ``survfit(Surv(tl, x, event) ~ 1, id =, istate =)``: the occupation
  probabilities and their infinitesimal-jackknife standard errors
  (``variance="robust"``), from the start with the states at risk then,
  and from ``s = 2`` with those states (the uncertainty of that starting
  distribution included) or from one state;
- ``mstate``'s ``probtrans(msfit(...))`` of the null Cox model with
  Breslow's ties (the Aalen-Johansen increments d / r): ``P(s, t)`` from
  every state with the Greenwood-type and Aalen-type standard errors;
- ``mstate`` with Cox models of the transitions (each covariate per
  transition, or ``male`` shared), Efron's and Breslow's ties: the
  coefficients, their standard errors, and ``P(s, t | z)`` with its
  standard errors (``probtrans``'s default, ``"aalen"``).

The estimates are the same arithmetic as R's: equal to rounding. The Cox
models' coefficients agree to the Newton-Raphson tolerance of the two
fits, which carries into the predictions and their standard errors.
"""

import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.datasets import load_mgus2
from surpyval.multistate import (
    MultiState,
    MultiStateProportionalHazards,
    TransitionMatrix,
)

from ._data import fixture, reference, values

SOURCE = "r_multistate"
STATES = {
    "mgus2_multistate": ("mgus", "pcm", "death"),
    "multistate": ("healthy", "ill", "dead"),
}
TRANSITIONS = {
    "mgus2_multistate": TransitionMatrix.illness_death(
        STATES["mgus2_multistate"]
    ),
    "multistate": TransitionMatrix.illness_death(
        STATES["multistate"], recovery=True
    ),
}


def _spells(name):
    """The fixture's spells, its state codes (1, 2, 3) as labels."""
    d = fixture(name)
    labels = STATES[name]
    return {
        "x": d["x"],
        "tl": d["tl"],
        "id": d["id"].astype(int),
        "state": [labels[int(v) - 1] for v in d["state"]],
        "e": [None if np.isnan(v) else labels[int(v) - 1] for v in d["e"]],
        "covariates": d,
    }


def _model(name):
    d = _spells(name)
    return MultiState.fit(
        d["x"],
        d["e"],
        d["state"],
        tl=d["tl"],
        i=d["id"],
        transitions=TRANSITIONS[name],
    )


MODELS = {name: _model(name) for name in STATES}


def test_mgus2_fixture_is_load_mgus2_spells():
    df = load_mgus2(spells=True)
    d = _spells("mgus2_multistate")
    assert list(df["id"]) == list(d["id"])
    assert list(df["state"]) == d["state"]
    assert list(df["e"]) == d["e"]
    assert_allclose(df["tl"], d["tl"], rtol=0, atol=0)
    assert_allclose(df["x"], d["x"], rtol=0, atol=0)


@pytest.mark.parametrize(
    "ref, s, from_state",
    [
        ("survfit_mgus2", None, None),
        ("survfit_sim", None, None),
        ("survfit_sim_s2", 2.0, None),
        ("survfit_sim_s2_ill", 2.0, "ill"),
    ],
)
def test_occupancy_and_robust_se_match_survfit(ref, s, from_state):
    v = values(SOURCE, ref)
    model = MODELS[reference(SOURCE, ref)["fixture"]]
    assert set(v["states"]) == set(model.states)
    t = v["times"]
    for state in v["states"]:
        p = model.occupancy(t, state, s=s, from_state=from_state)
        se = model.occupancy_se(
            t, state, s=s, from_state=from_state, variance="robust"
        )
        assert_allclose(p, v[f"p_{state}"], rtol=1e-12, atol=1e-14)
        assert_allclose(se, v[f"se_{state}"], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("variance", ["greenwood", "aalen"])
@pytest.mark.parametrize(
    "case, s",
    [("mgus2", 0.0), ("mgus2_s60", 60.0), ("sim", 0.0), ("sim_s2", 2.0)],
)
def test_transition_probabilities_and_se_match_mstate(case, s, variance):
    ref = f"mstate_{case}_{variance}"
    v = values(SOURCE, ref)
    model = MODELS[reference(SOURCE, ref)["fixture"]]
    t = v["times"]
    P = model.transition_probabilities(t, s=s)
    for a, from_state in enumerate(model.states):
        for b, to_state in enumerate(model.states):
            assert_allclose(
                P[:, a, b], v[f"P_{from_state}_{to_state}"], atol=1e-14
            )
            se = model.occupancy_se(
                t, to_state, s=s, from_state=from_state, variance=variance
            )
            want = v[f"se_{from_state}_{to_state}"]
            # mstate's square root of a variance that is negative by
            # rounding is NaN (and warns); it is 0 to rounding here. A
            # variance that is 0 to rounding (4e-19) has a square root of
            # 6e-10 where R's sums cancel exactly: hence atol.
            ok = ~np.isnan(want)
            assert np.all(se[~ok] < 1e-7)
            assert_allclose(se[ok], want[ok], rtol=1e-8, atol=1e-9)


def _cox(effects, ties):
    d = _spells("mgus2_multistate")
    Z = np.column_stack([d["covariates"]["z1"], d["covariates"]["male"]])
    return MultiStateProportionalHazards.fit(
        d["x"],
        Z,
        d["e"],
        d["state"],
        tl=d["tl"],
        i=d["id"],
        transitions=TRANSITIONS["mgus2_multistate"],
        shared=[1] if effects == "shared" else None,
        tie_method=ties,
    )


# mstate warns there too ("Negative diagonal elements of (I+dA)"); that
# the model warns is checked below.
@pytest.mark.filterwarnings("ignore:For the covariates")
@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("effects", ["specific", "shared"])
def test_cox_per_transition_matches_mstate(effects, ties):
    model = _cox(effects, ties)
    for s in (0, 60):
        v = values(SOURCE, f"mstate_cox_mgus2_{effects}_{ties}_s{s}")
        if s == 0:
            # R names the expanded columns z1.1 (z1 on transition 1), ...
            # in the same order as parameter_names
            assert len(v["coef_names"]) == len(model.parameter_names)
            assert_allclose(model.params, v["coef"], rtol=1e-8)
            assert_allclose(model.standard_errors(), v["se_coef"], rtol=1e-8)
        profile = [v["profile"]["z1"], v["profile"]["male"]]
        t = v["times"]
        for from_state in model.states:
            for to_state in model.states:
                p = model.occupancy(
                    t, profile, to_state, s=s, from_state=from_state
                )
                se = model.occupancy_se(
                    t, profile, to_state, s=s, from_state=from_state
                )
                key = f"{from_state}_{to_state}"
                assert_allclose(p, v[f"P_{key}"], rtol=1e-9, atol=1e-12)
                # Compared as variances. A probability that is 0 to
                # rounding (P(pcm -> pcm) at 30 years) has a variance of
                # 2e-12 that the recursion builds from terms the size of
                # the largest variance (0.02, an SE of 0.15): its rounding
                # is a few eps times that, 5e-18 in the variance, which is
                # 2e-12 in an SE of 1.4e-6 (relative 1.3e-6, platform
                # dependent). So the absolute tolerance is a small
                # multiple of eps times the largest variance in the
                # array; rtol 2e-6 on a variance is 1e-6 on an SE.
                want = v[f"se_{key}"] ** 2
                assert_allclose(
                    se**2,
                    want,
                    rtol=2e-6,
                    atol=64 * np.finfo(float).eps * np.nanmax(want),
                )


def test_coefficients_name_the_transitions():
    model = _cox("shared", "breslow")
    v = values(SOURCE, "mstate_cox_mgus2_shared_breslow_s0")
    assert v["coef_names"] == ["z1.1", "z1.2", "z1.3", "male"]
    assert model.parameter_names == [
        "z0: mgus -> pcm",
        "z0: mgus -> death",
        "z0: pcm -> death",
        "z1",
    ]


def test_cox_prediction_warns_where_the_hazards_add_up_past_one():
    # From 282 months a man of 75 has a predicted hazard of leaving MGUS
    # above 1 (two at risk), where mstate warns of a negative diagonal.
    model = _cox("specific", "efron")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.occupancy([120, 240], [0.5, 1.0], "pcm")
    with pytest.warns(UserWarning, match="add up to more than 1 at time 282"):
        model.occupancy([120, 300], [0.5, 1.0], "pcm")


# ---------------------------------------------------------------------------
# Effects shared by some transitions, and proportional baselines
# ---------------------------------------------------------------------------
# mstate's general expanded design (de Wreede, Fiocco and Putter, 2010):
# a column of the design applies a covariate to a set of transitions; the
# transitions with proportional baselines are one stratum, with the
# indicator of each but the first. The models below are those designs.
SIM_TRANSITIONS = TransitionMatrix(
    [
        ("healthy", "ill"),
        ("healthy", "dead"),
        ("ill", "healthy"),
        ("ill", "dead"),
    ],
    states=STATES["multistate"],
)


def _design(case, ties):
    if case == "sim_groups":
        d = _spells("multistate")
        unit = d["id"]
        Z = np.column_stack([(unit % 3 == 0).astype(float), (unit % 7) / 7])
        into_dead = [("healthy", "dead"), ("ill", "dead")]
        return MultiStateProportionalHazards.fit(
            d["x"],
            Z,
            d["e"],
            d["state"],
            tl=d["tl"],
            i=d["id"],
            transitions=SIM_TRANSITIONS,
            shared={0: into_dead, 1: "all"},
            proportional_baselines=into_dead,
            tie_method=ties,
        )
    d = _spells("mgus2_multistate")
    Z = np.column_stack([d["covariates"]["z1"], d["covariates"]["male"]])
    into_death = [("mgus", "death"), ("pcm", "death")]
    if case == "mgus2_groups":
        options = {"shared": {0: into_death}}
    else:
        options = {"shared": [1], "proportional_baselines": into_death}
    return MultiStateProportionalHazards.fit(
        d["x"],
        Z,
        d["e"],
        d["state"],
        tl=d["tl"],
        i=d["id"],
        transitions=TRANSITIONS["mgus2_multistate"],
        tie_method=ties,
        **options,
    )


DESIGNS = [
    ("mgus2_groups", "breslow", (0, 60)),
    ("mgus2_groups", "efron", (0, 60)),
    ("mgus2_proportional", "breslow", (0, 60)),
    ("mgus2_proportional", "efron", (0, 60)),
    ("sim_groups", "efron", (0, 2)),
]


def _compare_predictions(model, v, s, se_rtol):
    profile = list(v["profile"].values())
    t = v["times"]
    # The recursion builds every variance from terms the size of the
    # largest variance of any transition probability (see
    # test_cox_per_transition_matches_mstate), so that, not the largest in
    # one array, sets the rounding floor: an array whose variances are all
    # small (P(mgus -> pcm) from s = 60, at most 3.5e-4) still carries
    # rounding of a few eps times the largest (8.9e-18 against mstate's
    # 4.2e-19 on x86-64 Linux, both 0 to rounding).
    largest = max(
        np.nanmax(np.asarray(w, float) ** 2)
        for k, w in v.items()
        if k.startswith("se_")
    )
    for from_state in model.states:
        for to_state in model.states:
            key = f"{from_state}_{to_state}"
            p = model.occupancy(
                t, profile, to_state, s=s, from_state=from_state
            )
            se = model.occupancy_se(
                t, profile, to_state, s=s, from_state=from_state
            )
            assert_allclose(p, v[f"P_{key}"], rtol=1e-9, atol=1e-12)
            # As variances, for the reason given in
            # test_cox_per_transition_matches_mstate.
            # mstate's square root of a variance negative by rounding is
            # NaN; it is 0 to rounding here.
            want = v[f"se_{key}"] ** 2
            ok = ~np.isnan(want)
            assert np.all(se[~ok] < 1e-7)
            assert_allclose(
                se[ok] ** 2,
                want[ok],
                rtol=se_rtol,
                atol=64 * np.finfo(float).eps * largest,
            )


@pytest.mark.filterwarnings("ignore:For the covariates")
@pytest.mark.parametrize("case, ties, starts", DESIGNS)
def test_shared_groups_and_proportional_baselines_match_mstate(
    case, ties, starts
):
    model = _design(case, ties)
    for s in starts:
        v = values(SOURCE, f"mstate_cox_{case}_{ties}_s{s}")
        # the same columns in the same order
        assert len(v["coef_names"]) == len(model.parameter_names)
        assert_allclose(model.params, v["coef"], rtol=1e-8)
        assert_allclose(model.standard_errors(), v["se_coef"], rtol=1e-8)
        assert_allclose(model.log_likelihood, v["loglik"], rtol=1e-12)
        if "proportional" in case or case == "sim_groups":
            # mstate's msfit gives two transitions that share a baseline
            # the covariance phi_h^2 dN / S0^2 of their increments, h the
            # lower-numbered one; it is phi_h phi_g dN / S0^2 (the
            # baseline's variance, times each transition's relative
            # risk), which does not depend on the order of the
            # transitions. With mstate's term the standard errors agree;
            # with the right one they differ (by up to 13% here).
            model._mstate_cross = True
            model._cache = {}
            _compare_predictions(model, v, s, 2e-6)
            model._mstate_cross = False
            model._cache = {}
            # The default differs from mstate's, at least somewhere
            # (compared where the standard error is not 0 to rounding).
            profile = list(v["profile"].values())
            gap = 0.0
            for a in model.states:
                for b in model.states:
                    want = v[f"se_{a}_{b}"]
                    big = want > 1e-4
                    if np.any(big):
                        se = model.occupancy_se(
                            v["times"], profile, b, s=s, from_state=a
                        )
                        gap = max(gap, np.max(np.abs(se[big] / want[big] - 1)))
            assert 0.01 < gap < 0.2, gap
        else:
            _compare_predictions(model, v, s, 2e-6)


def test_shared_names_say_which_transitions():
    model = _design("mgus2_groups", "breslow")
    assert values(SOURCE, "mstate_cox_mgus2_groups_breslow_s0")[
        "coef_names"
    ] == ["z1.1", "z1.23", "male.1", "male.2", "male.3"]
    assert model.parameter_names == [
        "z0: mgus -> pcm",
        "z0: mgus -> death, pcm -> death",
        "z1: mgus -> pcm",
        "z1: mgus -> death",
        "z1: pcm -> death",
    ]
    prop = _design("mgus2_proportional", "breslow")
    assert (
        prop.parameter_names[-1] == "baseline: pcm -> death vs mgus -> death"
    )


def test_the_shared_baselines_covariance_does_not_depend_on_the_order():
    # The same model with its transitions listed in another order: the
    # standard errors are the same. mstate's term changes with the order.
    model = _design("sim_groups", "efron")
    d = _spells("multistate")
    unit = d["id"]
    Z = np.column_stack([(unit % 3 == 0).astype(float), (unit % 7) / 7])
    into_dead = [("ill", "dead"), ("healthy", "dead")]
    tm = TransitionMatrix(
        [
            ("ill", "dead"),
            ("ill", "healthy"),
            ("healthy", "dead"),
            ("healthy", "ill"),
        ],
        states=STATES["multistate"],
    )
    other = MultiStateProportionalHazards.fit(
        d["x"],
        Z,
        d["e"],
        d["state"],
        tl=d["tl"],
        i=d["id"],
        transitions=tm,
        shared={0: into_dead, 1: "all"},
        # the reference is now ill -> dead: the same model
        proportional_baselines=into_dead,
        tie_method="efron",
    )
    assert_allclose(other.log_likelihood, model.log_likelihood, rtol=1e-12)
    t = np.array([1.0, 3.0, 6.0])
    for a in ("healthy", "ill"):
        for b in model.states:
            args = (t, [1.0, 0.5], b)
            assert_allclose(
                other.occupancy_se(*args, from_state=a),
                model.occupancy_se(*args, from_state=a),
                rtol=1e-8,
            )
    model._mstate_cross = other._mstate_cross = True
    model._cache, other._cache = {}, {}
    differ = [
        np.max(
            np.abs(
                other.occupancy_se(t, [1.0, 0.5], b, from_state=a)
                - model.occupancy_se(t, [1.0, 0.5], b, from_state=a)
            )
        )
        for a in ("healthy", "ill")
        for b in model.states
    ]
    assert max(differ) > 1e-4
