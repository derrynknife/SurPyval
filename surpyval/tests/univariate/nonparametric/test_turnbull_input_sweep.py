"""Turnbull across the full matrix of supported inputs and estimators.

The individual Turnbull tests each pin one regime (a truncation form, an
endpoint convention, a variance identity). This file is the breadth
counterpart: every combination of censoring type, truncation form and
hazard estimator is fitted and checked to produce a valid survival curve.

It exists so that a change anywhere upstream of Turnbull -- in the data
handlers, the risk-set conversion, or the EM ladder -- cannot silently
break one corner of the input space while the targeted tests keep
passing.
"""

import warnings

import numpy as np
import pytest

from surpyval import Turnbull

ESTIMATORS = ["Nelson-Aalen", "Kaplan-Meier", "Fleming-Harrington"]
GRID = np.array([0.5, 2.0, 4.0, 6.0, 8.0, 11.0])
N = 30
# The Turnbull EM converges slowly on interval-censored data; the targeted
# tests use the same allowance. A sweep should assert on converged fits,
# not on whatever the default iteration cap happens to reach.
MAX_ITER = 20_000


def _case(kind):
    """Build ``Turnbull.fit`` kwargs for one input regime.

    Deterministic: every case is generated from a fixed seed so a failure
    is reproducible.
    """
    rng = np.random.default_rng(24601)
    base = np.sort(rng.uniform(1.0, 10.0, N))
    right = (rng.random(N) < 0.35).astype(int)
    upper = base + rng.uniform(0.5, 2.0, N)

    if kind == "observed":
        return dict(x=base, c=np.zeros(N, dtype=int))
    if kind == "observed_right":
        return dict(x=base, c=right)
    if kind == "observed_left":
        c = np.where(np.arange(N) % 4 == 1, -1, 0)
        return dict(x=base, c=c)
    if kind == "interval":
        return dict(x=np.column_stack([base, upper]), c=np.full(N, 2))
    if kind == "all_censoring_types":
        c = rng.choice([0, 1, -1, 2], size=N)
        c[0] = 0  # cannot be entirely censored
        x = np.column_stack([base, np.where(c == 2, upper, base)])
        return dict(x=x, c=c)
    if kind == "left_truncated":
        return dict(x=base, c=right, tl=rng.uniform(0.0, 0.9, N))
    if kind == "right_truncated":
        return dict(
            x=base,
            c=np.zeros(N, dtype=int),
            tr=base + rng.uniform(1.0, 4.0, N),
        )
    if kind == "both_truncated":
        # Observed events only: a right-censored row with a finite `tr` is
        # a genuine data contradiction that the handler warns about, so it
        # does not belong in a sweep of *valid* inputs.
        return dict(
            x=base,
            c=np.zeros(N, dtype=int),
            tl=rng.uniform(0.0, 0.9, N),
            tr=base + rng.uniform(1.0, 4.0, N),
        )
    if kind == "interval_left_truncated":
        return dict(
            x=np.column_stack([base, upper]),
            c=np.full(N, 2),
            tl=rng.uniform(0.0, 0.9, N),
        )
    if kind == "all_censoring_types_truncated":
        c = rng.choice([0, 1, -1, 2], size=N)
        c[0] = 0
        x = np.column_stack([base, np.where(c == 2, upper, base)])
        return dict(x=x, c=c, tl=rng.uniform(0.0, 0.9, N))
    raise ValueError(kind)


KINDS = [
    "observed",
    "observed_right",
    "observed_left",
    "interval",
    "all_censoring_types",
    "left_truncated",
    "right_truncated",
    "both_truncated",
    "interval_left_truncated",
    # Left censoring together with two or more distinct entry times is
    # not identifiable: the likelihood has a flat direction with its
    # supremum on the boundary, so the NPMLE is not attained and the EM
    # cannot settle. The fit now warns and reports the share of mass
    # resting there (#308), but the estimate it returns is still the
    # boundary one, so this sweep's assertions still fail. Deciding the
    # case exactly, rather than by a mass threshold, is #327. Marked
    # xfail rather than dropped so the sweep reports the day it changes.
    pytest.param(
        "all_censoring_types_truncated",
        marks=pytest.mark.xfail(
            reason="left censoring + distinct entry times is not "
            "identifiable; see #327",
            strict=True,
        ),
    ),
]


@pytest.mark.parametrize("estimator", ESTIMATORS)
@pytest.mark.parametrize("kind", KINDS)
def test_fit_produces_a_valid_survival_curve(kind, estimator):
    model = Turnbull.fit(
        **_case(kind), turnbull_estimator=estimator, max_iter=MAX_ITER
    )
    assert model.converged, f"{kind}/{estimator} did not converge"
    sf = np.ravel(model.sf(GRID)).astype(float)
    finite = sf[np.isfinite(sf)]

    assert finite.size, f"{kind}/{estimator} produced no finite survival"
    assert (
        (finite >= -1e-12) & (finite <= 1.0 + 1e-12)
    ).all(), f"{kind}/{estimator} survival outside [0, 1]: {finite}"
    assert (np.diff(finite) <= 1e-12).all(), (
        f"{kind}/{estimator} survival is not monotone non-increasing: "
        f"{finite}"
    )
    # The risk-set ladder must stay coherent: non-negative counts, no more
    # events than units at risk.
    r = np.asarray(model.r, dtype=float)
    d = np.asarray(model.d, dtype=float)
    assert (r >= -1e-9).all(), f"{kind}/{estimator} negative at-risk counts"
    assert (d >= -1e-9).all(), f"{kind}/{estimator} negative event counts"
    assert (d <= r + 1e-9).all(), f"{kind}/{estimator} more events than risk"


@pytest.mark.parametrize("kind", KINDS)
def test_estimator_choice_is_honoured(kind):
    # The three hazard estimators must not collapse onto one another --
    # a silently ignored `turnbull_estimator` would show up here.
    kwargs = _case(kind)
    curves = {}
    for est in ESTIMATORS:
        model = Turnbull.fit(
            **kwargs, turnbull_estimator=est, max_iter=MAX_ITER
        )
        # Comparing estimators on a fit that never converged says nothing
        # about the estimator switch, so require convergence first.
        assert model.converged, f"{kind}/{est} did not converge"
        curves[est] = np.ravel(model.sf(GRID)).astype(float)
    na, km = curves["Nelson-Aalen"], curves["Kaplan-Meier"]
    assert not np.array_equal(
        na, km
    ), f"{kind}: Nelson-Aalen and Kaplan-Meier gave identical curves"


def test_vacuous_left_truncation_does_not_change_a_left_censored_fit():
    # Every entry time is 0 and every observation is above 0, so no unit is
    # excluded and the truncated fit must equal the untruncated one. It
    # currently raises instead: a left-censored row lives on the (-inf, x]
    # bound, and the support-window intersection drops that bound as soon
    # as an entry time sits above its lower edge, even though the event
    # interval (0, x] is non-empty (#308).
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    c = np.array([-1, 0, 0, -1, 0, 0])
    grid = [2.5, 4.5, 6.5]

    untruncated = np.ravel(
        Turnbull.fit(x=x, c=c, turnbull_estimator="Kaplan-Meier").sf(grid)
    )
    truncated = np.ravel(
        Turnbull.fit(
            x=x,
            c=c,
            tl=np.zeros(6),
            turnbull_estimator="Kaplan-Meier",
            max_iter=MAX_ITER,
        ).sf(grid)
    )
    np.testing.assert_allclose(truncated, untruncated, rtol=1e-9, atol=1e-9)


def test_fleming_harrington_matches_nelson_aalen_only_without_ties():
    # Fleming-Harrington differs from Nelson-Aalen *only* in the tied-event
    # correction, so the two must agree when every event time is distinct
    # and separate once times are tied. Pinning both directions keeps the
    # estimator switch honest.
    def sf_for(x, estimator):
        return np.ravel(
            Turnbull.fit(
                x=x, turnbull_estimator=estimator, max_iter=MAX_ITER
            ).sf([2.0, 4.0, 6.0])
        ).astype(float)

    distinct = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    tied = np.array([2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 6.0, 7.0, 9.0])

    np.testing.assert_allclose(
        sf_for(distinct, "Fleming-Harrington"),
        sf_for(distinct, "Nelson-Aalen"),
        rtol=1e-9,
        atol=1e-9,
    )
    assert not np.allclose(
        sf_for(tied, "Fleming-Harrington"),
        sf_for(tied, "Nelson-Aalen"),
        rtol=1e-6,
        atol=1e-6,
    )


def test_left_censored_row_keeps_the_interval_starting_at_its_entry_time():
    """The support index at the entry time is admissible, not excluded.

    A support index ``j`` stands for ``(bounds[j], bounds[j+1]]``, so an
    event placed there is already strictly after ``bounds[j]``. The first
    index a row entering at ``tl`` may use is therefore the *last* one
    whose bound equals ``tl`` -- that interval is ``(tl, next]``.

    A left-censored event lies in ``(-inf, xr]``, which under an entry at
    ``tl`` is the single interval ``(tl, xr]``. Excluding it left such a
    row with an empty support, and the fit raised (#308).

    A *common* entry time below every observation excludes nobody and
    introduces no differential risk, so the estimate must equal the
    untruncated one exactly. Distinct entry times below every observation
    also exclude nobody, but do not yet round-trip -- see the reproducer
    below.
    """
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    c = np.array([-1, 0, 0, -1, 0, 0])
    grid = [1.5, 2.5, 4.5, 6.5, 8.0]

    untruncated = np.ravel(
        Turnbull.fit(x=x, c=c, turnbull_estimator="Kaplan-Meier").sf(grid)
    )
    for entry in (np.zeros(6), np.full(6, 0.5), np.full(6, 1.9)):
        truncated = np.ravel(
            Turnbull.fit(
                x=x, c=c, tl=entry, turnbull_estimator="Kaplan-Meier"
            ).sf(grid)
        )
        assert np.allclose(truncated, untruncated, atol=1e-9), (
            f"entry times {entry} exclude nobody, so the fit must be "
            f"unchanged; got {truncated} against {untruncated}"
        )


@pytest.mark.xfail(
    reason="left censoring + distinct entry times is not identifiable; the "
    "fit now warns, but the estimate itself is still the boundary one. "
    "See #327",
    strict=True,
)
def test_distinct_entry_times_with_left_censoring_round_trip():
    """Minimal reproducer for the half of #308 that is still open.

    Six observations, two of them left censored, and six *distinct* entry
    times all below the earliest observation. Nobody is excluded, so the
    estimate should again be the untruncated one; instead the survival
    curve collapses to the order of 1e-3.

    A common entry time (the test above) round-trips exactly, so the
    trigger is entry times that *differ*. That is what activates the
    ghost step: a row entering later than another has unseen deaths
    imputed below its own window, and the expected event counts come back
    inflated -- the sweep case sees ~3.1 expected events at every
    observation time for 30 observations, which exhausts the risk set
    within a few steps of the estimator ladder.

    Six points rather than the thirty in the issue, and the failure is
    the same, so this is the cheaper thing to debug against.
    """
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    c = np.array([-1, 0, 0, -1, 0, 0])
    grid = [1.5, 2.5, 4.5, 6.5, 8.0]

    untruncated = np.ravel(
        Turnbull.fit(x=x, c=c, turnbull_estimator="Kaplan-Meier").sf(grid)
    )
    truncated = np.ravel(
        Turnbull.fit(
            x=x,
            c=c,
            tl=np.linspace(0.1, 1.0, 6),
            turnbull_estimator="Kaplan-Meier",
            max_iter=MAX_ITER,
        ).sf(grid)
    )
    assert np.allclose(truncated, untruncated, atol=1e-9)


def test_entry_exactly_at_an_event_time_stays_strict():
    """Widening the entry window must not readmit the event at the entry.

    The fix above must not reach the zero-width interval ``(tl, tl]`` that
    a duplicated exact event time creates: a subject entering exactly at
    an event time is not at risk for that event (#260). Taking
    ``side="left"`` rather than ``side="right" - 1`` would have readmitted
    it.
    """
    x = np.array([2.0, 3.0, 3.0, 4.0, 5.0, 6.0])
    tl = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])

    model = Turnbull.fit(x=x, tl=tl, turnbull_estimator="Kaplan-Meier")
    # Two subjects enter at 2.0 and so are not at risk for the event at
    # 2.0: four at risk, one event, 1 - 1/4.
    assert float(np.ravel(model.sf(2.0))[0]) == pytest.approx(0.75, abs=1e-6)


def test_non_identifiable_entry_windows_are_reported():
    """A fit resting on the flat direction must say so.

    Left censoring together with two or more distinct entry times creates
    intervals that one observation could have failed in but that precede
    another's entry. Mass there raises the first's likelihood and costs
    the others nothing, because their contributions are conditional on
    their own entry. The likelihood has no interior maximum, so the
    estimate is not identifiable and the EM cannot settle (#308).

    Rejecting the data would be wrong -- over 240 simulated samples that
    all met the structural condition, most fitted perfectly well -- so
    the fit is returned with a warning and a reported share.
    """
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    c = np.array([-1, 0, 0, -1, 0, 0])

    with pytest.warns(UserWarning, match="not identifiable"):
        model = Turnbull.fit(
            x=x,
            c=c,
            tl=np.linspace(0.1, 1.0, 6),
            turnbull_estimator="Kaplan-Meier",
            max_iter=MAX_ITER,
        )
    assert model.exploitable_mass > 0.9


@pytest.mark.parametrize(
    "tl",
    [
        pytest.param(np.zeros(6), id="common_entry_at_zero"),
        pytest.param(np.full(6, 0.5), id="common_entry_below_data"),
        pytest.param(None, id="no_truncation"),
    ],
)
def test_identifiable_fits_are_not_warned_about(tl):
    """The warning must not fire on data that is perfectly estimable.

    One common entry time gives every observation the same window, so no
    interval is inside one and outside another and the flat direction
    does not exist.
    """
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    c = np.array([-1, 0, 0, -1, 0, 0])
    kwargs = {} if tl is None else {"tl": tl}

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = Turnbull.fit(
            x=x, c=c, turnbull_estimator="Kaplan-Meier", **kwargs
        )
    assert model.exploitable_mass == 0.0


def test_distinct_entry_times_alone_are_identifiable():
    """Distinct entry times are not themselves the problem.

    Without a left-censored row nothing reaches down into the entry
    region, so six distinct entry times estimate cleanly.
    """
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    c = np.zeros(6, dtype=int)

    untruncated = np.ravel(
        Turnbull.fit(x=x, c=c, turnbull_estimator="Kaplan-Meier").sf(
            [2.5, 4.5]
        )
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = Turnbull.fit(
            x=x,
            c=c,
            tl=np.linspace(0.1, 1.0, 6),
            turnbull_estimator="Kaplan-Meier",
        )
    assert model.exploitable_mass == 0.0
    np.testing.assert_allclose(
        np.ravel(model.sf([2.5, 4.5])), untruncated, rtol=1e-9, atol=1e-9
    )
