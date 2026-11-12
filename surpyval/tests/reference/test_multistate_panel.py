"""The panel-data multi-state models (``PanelMultiState``,
``PanelMultiStateRegression``) against R ``msm``.

The values are in ``data/r_multistate_panel.json``, written by
``scripts/reference/reference_r_multistate_panel.R`` (R 4.6.2, msm 1.9)
on the fixture ``multistate_panel``: 300 simulated units of a four-state
Markov model seen at visits, with the transitions of msm's
heart-transplant example, exactly timed deaths and last visits that saw
only "alive" (one of the three living states). msm's own ``cav`` data
are not copied into the repository (their licence, as data distributed
in a GPL package from a hospital registry, is unclear). The same checks
were run on them locally while building the model (not repeated here):
the model without covariates agrees with msm's log-likelihood to 1e-12,
its intensities to 3e-7 and their standard errors to 1.3e-6; with sex
on every intensity, the baseline intensities agree to 2e-7, and the
effect of sex on the move from state 2 to 4 has no finite maximum
(SurPyval warns; msm reports an intensity ratio of 1.2e-6).

What is compared, each fit with msm's optimiser run to ``reltol =
1e-14``:

- the maximised log-likelihood (to 1e-9 relative: msm's is the same
  likelihood, conditional on each unit's first state, with the density
  of an exactly timed death per unit of time);
- the intensities, their delta-method standard errors and 95% bounds
  (``qmatrix.msm``), to 1e-5 relative (the two optimisers' answers differ
  by up to 6e-7);
- ``pmatrix.msm`` at two times, to 1e-6;
- the mean stays (``sojourn.msm``) with their standard errors and bounds;
- with covariates (male, and w, which changes at each visit, so this
  also checks that a row's covariates hold over the gap that starts at
  it, as msm takes them): the intensity ratios and their bounds
  (``hazard.msm``) and the intensities at ``Z = 0``; one shared effect
  (msm's ``constraint``); and piecewise-constant intensities (``pci``).
- a fit to snapshots only (no censored states, no exact deaths).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.multistate import (
    PanelMultiState,
    PanelMultiStateRegression,
    TransitionMatrix,
)

from ._data import fixture, values

SOURCE = "r_multistate_panel"
TRANSITIONS = [(1, 2), (1, 4), (2, 1), (2, 3), (2, 4), (3, 2), (3, 4)]
TM = TransitionMatrix(TRANSITIONS, states=[1, 2, 3, 4])
H = len(TRANSITIONS)
RTOL = 1e-5


@pytest.fixture(scope="module")
def data():
    d = fixture("multistate_panel")
    return {
        "x": d["time"],
        "s": d["state"].astype(int),
        "i": d["id"].astype(int),
        "obstype": np.where(d["obstype"] == 3, "exact_entry", "panel"),
        "male": d["male"],
        "w": d["w"],
    }


def _kw(d):
    return dict(obstype=d["obstype"], transitions=TM, censored={99: [1, 2, 3]})


def _check_intensities(model, ref):
    assert model.maximum == "verified"
    assert_allclose(model.log_likelihood, ref["loglik"], rtol=1e-9)
    assert_allclose(model.params[:H], ref["q"], rtol=RTOL)
    assert_allclose(model.standard_errors()[:H], ref["q_se"], rtol=RTOL)
    cb = np.array([model.param_cb(n) for n in model.parameter_names[:H]])
    assert_allclose(cb[:, 0], ref["q_lower"], rtol=RTOL)
    assert_allclose(cb[:, 1], ref["q_upper"], rtol=RTOL)


def _check_ratios(model, ref, name, rows):
    """Each transition's intensity ratio and its bounds (exp of the
    coefficient's Wald bound), for the coefficients at ``rows``."""
    names = [model.parameter_names[k] for k in rows]
    est = np.exp(model.params[rows])
    cb = np.exp(np.array([model.param_cb(n) for n in names]))
    assert_allclose(est, ref[f"hr_{name}"], rtol=RTOL)
    assert_allclose(cb[:, 0], ref[f"hr_lower_{name}"], rtol=RTOL)
    assert_allclose(cb[:, 1], ref[f"hr_upper_{name}"], rtol=RTOL)


@pytest.fixture(scope="module")
def panel(data):
    return PanelMultiState.fit(data["x"], data["s"], data["i"], **_kw(data))


def test_panel_intensities(panel):
    _check_intensities(panel, values(SOURCE, "panel"))


def test_panel_transition_probabilities(panel):
    ref = values(SOURCE, "panel")
    for t, key in ((1.0, "P1"), (5.0, "P5")):
        assert_allclose(
            panel.transition_probabilities(t, s=0.0), ref[key], atol=1e-6
        )


def test_panel_mean_sojourn(panel):
    ref = values(SOURCE, "panel")
    states = [1, 2, 3]
    assert_allclose(
        [panel.mean_sojourn(st) for st in states], ref["sojourn"], rtol=RTOL
    )
    cb = np.array([panel.sojourn_cb(st) for st in states])
    assert_allclose(cb[:, 0], ref["sojourn_lower"], rtol=RTOL)
    assert_allclose(cb[:, 1], ref["sojourn_upper"], rtol=RTOL)
    # msm's standard error is on the time scale: the delta method from
    # the log scale's
    z = 1.959963984540054
    se = panel.mean_sojourn(1) * (np.log(cb[0, 1]) - np.log(cb[0, 0])) / 2 / z
    assert_allclose(se, ref["sojourn_se"][0], rtol=RTOL)


def test_snapshots_only(data):
    keep = data["s"] != 99
    model = PanelMultiState.fit(
        data["x"][keep], data["s"][keep], data["i"][keep], transitions=TM
    )
    ref = values(SOURCE, "panel_snapshot")
    _check_intensities(model, ref)
    assert_allclose(
        model.transition_probabilities(5.0, s=0.0), ref["P5"], atol=1e-6
    )


def test_covariates(data):
    Z = np.column_stack([data["male"], data["w"]])
    model = PanelMultiStateRegression.fit(
        data["x"], Z, data["s"], data["i"], **_kw(data)
    )
    ref = values(SOURCE, "covariates")
    _check_intensities(model, ref)
    _check_ratios(model, ref, "male", list(range(H, 2 * H)))
    _check_ratios(model, ref, "w", list(range(2 * H, 3 * H)))
    assert_allclose(
        model.transition_probabilities(5.0, [1.0, 0.5], s=0.0),
        ref["P5_male"],
        atol=1e-6,
    )


def test_shared_effect(data):
    model = PanelMultiStateRegression.fit(
        data["x"], data["male"], data["s"], data["i"], shared=0, **_kw(data)
    )
    ref = values(SOURCE, "shared")
    _check_intensities(model, ref)
    assert model.parameter_names[H:] == ["z0"]
    _check_ratios(model, ref, "male", [H] * H)


def test_piecewise_constant_intensities(data):
    model = PanelMultiState.fit(
        data["x"], data["s"], data["i"], pci=[3, 6], **_kw(data)
    )
    ref = values(SOURCE, "pci")
    _check_intensities(model, ref)
    _check_ratios(model, ref, "timeperiod[3,6)", list(range(H, 2 * H)))
    _check_ratios(model, ref, "timeperiod[6,Inf)", list(range(2 * H, 3 * H)))
