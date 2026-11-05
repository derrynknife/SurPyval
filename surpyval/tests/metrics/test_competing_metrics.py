"""Properties of the Brier score, integrated Brier score and time-dependent
AUC of one cause with competing risks (``e=`` and ``cause=``); the R
references are in surpyval/tests/reference/test_competing_metrics.py.

* With one cause they are the ordinary metrics of ``S = 1 - F``, standard
  errors included, and the two definitions of the AUC's controls agree.
* The labels do not matter: scoring a cause called ``"death"`` is scoring
  it called 2, and it need not be the first label.
* Without censoring every weight is 1: the Brier score is the mean squared
  error against the indicator of the cause by ``t``, and the AUC the
  Mann-Whitney statistic of the cases against the controls of the chosen
  definition, both worked by brute force.
* ``c`` may be omitted: the rows without an event type are the censored
  ones.
* The 95% intervals cover the population Brier score and AUC at about the
  nominal rate (a simulation, seed and size fixed before the first run).
* The refusals name the argument and say what to pass.
* The result records the cause scored and the AUC's controls, and keeps
  them through ``to_dict`` / ``from_dict`` (a tuple label included).
"""

import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.metrics import (
    PredictionScores,
    auc_td,
    brier_score,
    integrated_brier_score,
    score_predictions,
)

TIMES = np.array([2.0, 5.0, 8.0])


def _cif(z, t, cause):
    """The true cumulative incidences of the simulation: constant
    cause-specific hazards 0.1 exp(0.8 z) (cause 1) and 0.08 (cause 2)."""
    l1 = 0.1 * np.exp(0.8 * np.asarray(z))[:, None]
    l2 = 0.08
    share = l1 / (l1 + l2) if cause == 1 else l2 / (l1 + l2)
    return share * (1 - np.exp(-(l1 + l2) * np.asarray(t)[None, :]))


def _simulate(rng, n, censor=12.0):
    z = rng.normal(size=n)
    l1 = 0.1 * np.exp(0.8 * z)
    t = rng.exponential(1 / (l1 + 0.08))
    cause = np.where(rng.uniform(size=n) < l1 / (l1 + 0.08), 1, 2)
    if censor is None:
        return z, t, list(cause)
    u = rng.uniform(0, censor, n)
    x = np.minimum(t, u)
    e = [None if u_i < t_i else int(k) for u_i, t_i, k in zip(u, t, cause)]
    return z, x, e


def _data(n=80, seed=22910, censor=12.0):
    z, x, e = _simulate(np.random.default_rng(seed), n, censor)
    # A deliberately imperfect model: the right form, an attenuated effect.
    F = _cif(0.6 * z, TIMES, 1)
    return x, e, F


def test_one_cause_is_the_ordinary_score():
    x, e, F = _data()
    c = np.array([1 if k is None else 0 for k in e])
    one = ["event" if k is not None else None for k in e]
    S = 1 - F
    assert_allclose(
        brier_score(x, None, F, TIMES, e=one, cause="event")[1],
        brier_score(x, c, S, TIMES)[1],
        rtol=1e-12,
    )
    assert_allclose(
        integrated_brier_score(x, None, F, TIMES, e=one, cause="event"),
        integrated_brier_score(x, c, S, TIMES),
        rtol=1e-12,
    )
    for controls in ("cause_free", "event_free"):
        assert_allclose(
            auc_td(x, None, F, TIMES, e=one, cause="event", controls=controls)[
                1
            ],
            auc_td(x, c, F, TIMES)[1],
            rtol=1e-12,
        )
    got = score_predictions(x, None, F, TIMES, e=one, cause="event")
    plain = score_predictions(x, c, S, TIMES)
    assert_allclose(got.scores.estimate, plain.scores.estimate, rtol=1e-12)
    assert_allclose(got.scores.se, plain.scores.se, rtol=1e-10)


def test_labels_do_not_matter():
    x, e, F = _data()
    names = {1: "relapse", 2: "death"}
    named = [None if k is None else names[k] for k in e]
    swapped = [None if k is None else 3 - k for k in e]
    by_int = score_predictions(x, None, F, TIMES, e=e, cause=1)
    by_name = score_predictions(x, None, F, TIMES, e=named, cause="relapse")
    by_second = score_predictions(x, None, F, TIMES, e=swapped, cause=2)
    for other in (by_name, by_second):
        assert_allclose(other.scores.estimate, by_int.scores.estimate)
        assert_allclose(other.scores.se, by_int.scores.se, rtol=1e-12)


def test_c_may_be_omitted_or_given():
    x, e, F = _data()
    c = [1 if k is None else 0 for k in e]
    assert_allclose(
        brier_score(x, None, F, TIMES, e=e, cause=1)[1],
        brier_score(x, c, F, TIMES, e=e, cause=1)[1],
    )


def test_without_censoring_the_metrics_are_plain_averages():
    x, e, F = _data(n=40, censor=None)
    e = np.array(e)
    for k, t in enumerate(TIMES):
        y = ((x <= t) & (e == 1)).astype(float)
        _, bs = brier_score(x, None, F[:, [k]], [t], e=list(e), cause=1)
        assert_allclose(bs, np.mean((y - F[:, k]) ** 2), rtol=1e-12)
        cases = y == 1
        for controls, ctrl in (
            ("cause_free", ~cases),
            ("event_free", x > t),
        ):
            pairs = [
                (F[i, k] > F[j, k]) + 0.5 * (F[i, k] == F[j, k])
                for i in np.flatnonzero(cases)
                for j in np.flatnonzero(ctrl)
            ]
            _, auc = auc_td(
                x, None, F[:, k], [t], e=list(e), cause=1, controls=controls
            )
            assert_allclose(auc, np.mean(pairs), rtol=1e-12)


def test_a_model_compared_with_itself_differs_by_zero():
    x, e, F = _data()
    result = score_predictions(x, None, {"a": F, "b": F}, TIMES, e=e, cause=2)
    assert_allclose(result.comparisons.difference, 0.0, atol=1e-15)
    assert_allclose(result.comparisons.se, 0.0, atol=1e-15)


def test_intervals_cover_at_the_nominal_rate():
    """400 data sets of 200 subjects (seed 22911, fixed before the first
    run), scored with the true incidences of cause 1; the population
    Brier score and AUC (both definitions) from 400,000 uncensored
    subjects. Each 95% interval should cover about 380 times; the band
    allows for the binomial error of 400 draws (sd about 1.1%)."""
    rng = np.random.default_rng(22911)
    times = np.array([2.0, 5.0])
    z, t, cause = _simulate(rng, 400_000, censor=None)
    cause = np.array(cause)
    F = _cif(z, times, 1)
    truth = {}
    for k, h in enumerate(times):
        y = (t <= h) & (cause == 1)
        truth["brier", k] = np.mean((y - F[:, k]) ** 2)
        # The AUC as the probability a case outranks a control: F is
        # increasing in z, so rank by z (the Mann-Whitney statistic).
        for controls, ctrl in (("cause_free", ~y), ("event_free", t > h)):
            ranks = np.argsort(np.argsort(z[y | ctrl])) + 1.0
            is_case = y[y | ctrl]
            n1, n0 = is_case.sum(), (~is_case).sum()
            u = ranks[is_case].sum() - n1 * (n1 + 1) / 2
            truth[controls, k] = u / (n1 * n0)
    reps = 400
    covered = dict.fromkeys(truth, 0)
    for _ in range(reps):
        z, x, e = _simulate(rng, 200)
        F = _cif(z, times, 1)
        for controls in ("cause_free", "event_free"):
            result = score_predictions(
                x,
                None,
                F,
                times,
                metrics=("brier", "auc"),
                e=e,
                cause=1,
                controls=controls,
            )
            table = result.scores
            for k in range(times.size):
                for metric, key in (("brier", "brier"), ("auc", controls)):
                    if metric == "brier" and controls == "event_free":
                        continue
                    row = table[table.metric == metric].iloc[k]
                    covered[key, k] += row.lower <= truth[key, k] <= row.upper
    for key, count in covered.items():
        assert 0.92 <= count / reps <= 0.98, (key, count / reps)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(cause=1), "needs 'e'"),
        (dict(e="labels"), "pass cause="),
        (dict(e="labels", cause=3), "Unknown cause 3"),
        (dict(e="labels", cause=1, c="missing"), "'c' is required"),
    ],
)
def test_refusals_name_the_argument(kwargs, match):
    x, e, F = _data()
    c = [1 if k is None else 0 for k in e]
    kwargs = dict(kwargs)
    if kwargs.get("e") == "labels":
        kwargs["e"] = e
    if kwargs.pop("c", None) == "missing":
        kwargs.pop("e")
        kwargs.pop("cause")
        c = None
    for f in (brier_score, auc_td, score_predictions):
        with pytest.raises(ValueError, match=match):
            f(x, c, F, TIMES, **kwargs)


def test_refuses_a_failure_without_an_event_type():
    x, e, F = _data()
    c = [0] * len(x)  # says every row failed, but some have no type
    with pytest.raises(ValueError, match="no event type"):
        brier_score(x, c, F, TIMES, e=e, cause=1)


def test_refuses_incidences_outside_the_unit_interval():
    x, e, F = _data()
    with pytest.raises(ValueError, match="cumulative incidences"):
        brier_score(x, None, F * 5, TIMES, e=e, cause=1)


def test_refuses_an_unknown_controls_definition():
    x, e, F = _data()
    with pytest.raises(ValueError, match="controls"):
        auc_td(x, None, F, TIMES, e=e, cause=1, controls="all")
    with pytest.raises(ValueError, match="controls"):
        score_predictions(x, None, F, TIMES, e=e, cause=1, controls="all")


def test_refuses_event_types_of_another_length():
    x, e, F = _data()
    with pytest.raises(ValueError, match="same length"):
        brier_score(x, None, F, TIMES, e=e[:-1], cause=1)


def test_the_result_records_the_cause_and_round_trips():
    x, e, F = _data()
    labels = {1: ("relapse", 1), 2: ("death", 2)}
    e = [None if k is None else labels[k] for k in e]
    result = score_predictions(x, None, F, TIMES, e=e, cause=("relapse", 1))
    assert result.cause == ("relapse", 1)
    assert result.controls == "cause_free"
    back = PredictionScores.from_dict(json.loads(json.dumps(result.to_dict())))
    assert back.cause == ("relapse", 1)
    assert back.controls == "cause_free"
    assert_allclose(back.scores.estimate, result.scores.estimate)
    plain = score_predictions(
        x, [1 if k is None else 0 for k in e], 1 - F, TIMES
    )
    assert plain.cause is None and plain.controls is None
