"""The multi-state models against R ``survival`` and ``mstate``.

The data and R's results are in ``data/r_multistate.json``, written by
``scripts/reference/reference_r_multistate.R`` (R 4.6.2, survival 3.8.12,
mstate 0.3.3; the versions are stored in the file):

- "mgus2": ``survival::mgus2`` as an illness-death model (MGUS -> PCM ->
  death, MGUS -> death), which ``load_mgus2(spells=True)`` gives too;
- "sim": 300 units simulated in R from an illness-death model with
  recovery, with delayed entry and tied times.

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

import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.datasets import load_mgus2
from surpyval.multistate import (
    MultiState,
    MultiStateProportionalHazards,
    TransitionMatrix,
)

DATA = Path(__file__).parent / "data" / "r_multistate.json"


def _load():
    with open(DATA) as f:
        return json.load(f)


J = _load()

TRANSITIONS = {
    "mgus2": TransitionMatrix.illness_death(("mgus", "pcm", "death")),
    "sim": TransitionMatrix.illness_death(
        ("healthy", "ill", "dead"), recovery=True
    ),
}


def _array(values):
    return np.array([np.nan if v is None else v for v in values], float)


def _model(name):
    d = J["data"][name]
    return MultiState.fit(
        d["x"],
        d["e"],
        d["state"],
        tl=d["tl"],
        id=d["id"],
        transitions=TRANSITIONS[name],
    )


MODELS = {name: _model(name) for name in TRANSITIONS}


def test_the_file_records_its_software():
    assert J["generator"] == "scripts/reference/reference_r_multistate.R"
    assert J["software"] == {
        "R": "4.6.2",
        "survival": "3.8.12",
        "mstate": "0.3.3",
    }


def test_load_mgus2_spells_are_the_reference_data():
    df = load_mgus2(spells=True)
    d = J["data"]["mgus2"]
    assert list(df["id"]) == d["id"]
    assert list(df["state"]) == d["state"]
    assert list(df["e"]) == d["e"]
    assert_allclose(df["tl"], d["tl"], rtol=0, atol=0)
    assert_allclose(df["x"], d["x"], rtol=0, atol=0)
    assert_allclose(df["male"], d["male"], rtol=0, atol=0)
    assert_allclose((df["age"] - 70) / 10, d["z1"], rtol=1e-15)


@pytest.mark.parametrize(
    "ref, name, s, from_state",
    [
        ("survfit_mgus2", "mgus2", None, None),
        ("survfit_sim", "sim", None, None),
        ("survfit_sim_s2", "sim", 2.0, None),
        ("survfit_sim_s2_ill", "sim", 2.0, "ill"),
    ],
)
def test_occupancy_and_robust_se_match_survfit(ref, name, s, from_state):
    v = J["references"][ref]["values"]
    model = MODELS[name]
    t = np.array(v["times"], float)
    assert set(v["states"]) == set(model.states)
    for state in v["states"]:
        p = model.occupancy(t, state, s=s, from_state=from_state)
        se = model.occupancy_se(
            t, state, s=s, from_state=from_state, variance="robust"
        )
        assert_allclose(p, v[f"p_{state}"], rtol=1e-12, atol=1e-14)
        assert_allclose(se, v[f"se_{state}"], rtol=1e-10, atol=1e-14)


@pytest.mark.parametrize("variance", ["greenwood", "aalen"])
@pytest.mark.parametrize(
    "name, case, s",
    [
        ("mgus2", "mgus2", 0.0),
        ("mgus2", "mgus2_s60", 60.0),
        ("sim", "sim", 0.0),
        ("sim", "sim_s2", 2.0),
    ],
)
def test_transition_probabilities_and_se_match_mstate(name, case, s, variance):
    v = J["references"][f"mstate_{case}_{variance}"]["values"]
    model = MODELS[name]
    t = np.array(v["times"], float)
    P = model.transition_probabilities(t, s=s)
    for a, from_state in enumerate(model.states):
        for b, to_state in enumerate(model.states):
            assert_allclose(
                P[:, a, b], v[f"P_{from_state}_{to_state}"], atol=1e-14
            )
            se = model.occupancy_se(
                t, to_state, s=s, from_state=from_state, variance=variance
            )
            ref = _array(v[f"se_{from_state}_{to_state}"])
            # mstate's square root of a variance that is negative by
            # rounding is NaN (and warns); it is 0 to rounding here. A
            # variance that is 0 to rounding (4e-19) has a square root of
            # 6e-10 where R's sums cancel exactly: hence atol.
            ok = ~np.isnan(ref)
            assert np.all(se[~ok] < 1e-7)
            assert_allclose(se[ok], ref[ok], rtol=1e-8, atol=1e-9)


# mstate warns there too ("Negative diagonal elements of (I+dA)"); that
# the model warns is checked below.
@pytest.mark.filterwarnings("ignore:For the covariates")
@pytest.mark.parametrize("ties", ["breslow", "efron"])
@pytest.mark.parametrize("effects", ["specific", "shared"])
def test_cox_per_transition_matches_mstate(effects, ties):
    d = J["data"]["mgus2"]
    Z = np.column_stack([d["z1"], d["male"]])
    model = MultiStateProportionalHazards.fit(
        d["x"],
        Z,
        d["e"],
        d["state"],
        tl=d["tl"],
        id=d["id"],
        transitions=TRANSITIONS["mgus2"],
        shared=[1] if effects == "shared" else None,
        tie_method=ties,
    )
    for s in (0, 60):
        v = J["references"][f"mstate_cox_mgus2_{effects}_{ties}_s{s}"][
            "values"
        ]
        if s == 0:
            # R names the expanded columns z1.1 (z1 on transition 1), ...
            # in the same order as parameter_names
            assert len(v["coef_names"]) == len(model.parameter_names)
            assert_allclose(model.params, v["coef"], rtol=1e-8)
            assert_allclose(model.standard_errors(), v["se_coef"], rtol=1e-8)
        profile = [v["profile"]["z1"], v["profile"]["male"]]
        t = np.array(v["times"], float)
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
                assert_allclose(se, _array(v[f"se_{key}"]), rtol=1e-6)


def test_coefficients_name_the_transitions():
    d = J["data"]["mgus2"]
    model = MultiStateProportionalHazards.fit(
        d["x"],
        np.column_stack([d["z1"], d["male"]]),
        d["e"],
        d["state"],
        tl=d["tl"],
        id=d["id"],
        transitions=TRANSITIONS["mgus2"],
        shared=[1],
    )
    v = J["references"]["mstate_cox_mgus2_shared_breslow_s0"]["values"]
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
    d = J["data"]["mgus2"]
    model = MultiStateProportionalHazards.fit(
        d["x"],
        np.column_stack([d["z1"], d["male"]]),
        d["e"],
        d["state"],
        tl=d["tl"],
        id=d["id"],
        transitions=TRANSITIONS["mgus2"],
    )
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model.occupancy([120, 240], [0.5, 1.0], "pcm")
    with pytest.warns(UserWarning, match="add up to more than 1 at time 282"):
        model.occupancy([120, 300], [0.5, 1.0], "pcm")
