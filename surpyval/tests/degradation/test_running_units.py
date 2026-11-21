"""Units still short of the threshold at their last reading.

By default (the pseudo-failure-time method) a unit whose fitted path
crosses after its last reading fails at the extrapolated crossing, and
the model marks it ``extrapolated``; with ``running="censor"``, or when
listed in ``censored_units``, it is right censored at its last reading.

The paths here are exact straight lines with slopes of 1/4, 1/5, 1/8 and
1/10 from 0, so each crossing of the threshold 10 is a whole number
(40, 50, 80, 100). The life fit is checked against a fit of the units'
times and flags, which are checked against those numbers: comparing the
life fit with a fit of the hand numbers themselves differs by up to 5e-7
(relative), the optimiser's stopping point moved by the last bits of the
least-squares crossings, not a difference in the data.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.degradation import DegradationAnalysis, DegradationModel

THRESHOLD = 10.0
# unit: (slope, last reading time)
UNITS = {
    0: (0.25, 60.0),  # crosses at 40, within its readings
    1: (0.2, 60.0),  # crosses at 50, within
    2: (0.125, 60.0),  # crosses at 80, past its last reading at 60
    3: (0.1, 70.0),  # crosses at 100, past its last reading at 70
    4: (0.25, 45.0),  # crosses at 40, within
    5: (0.2, 55.0),  # crosses at 50, within
}
CROSSING = {0: 40.0, 1: 50.0, 2: 80.0, 3: 100.0, 4: 40.0, 5: 50.0}


def _data(units=UNITS):
    x, y, i = [], [], []
    for unit, (slope, last) in units.items():
        t = np.arange(5.0, last + 1, 5.0)
        x.append(t)
        y.append(slope * t)
        i.append(np.full(t.size, unit))
    return np.concatenate(x), np.concatenate(y), np.concatenate(i)


def _same_life_fit(model):
    """The life model is the Weibull fit of the units' times, with their
    censoring flags."""
    reference = sp.Weibull.fit(model.pseudo_failure_times, c=model.c)
    np.testing.assert_allclose(
        model.life_model.params, reference.params, rtol=1e-8
    )
    np.testing.assert_allclose(
        model.sf([45.0, 65.0]), reference.sf([45.0, 65.0]), rtol=1e-8
    )


def test_default_extrapolates_and_marks_the_units():
    x, y, i = _data()
    model = DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD)
    np.testing.assert_allclose(
        model.pseudo_failure_times,
        [CROSSING[u] for u in UNITS],
        rtol=1e-12,
    )
    assert (model.c == 0).all()
    assert model.extrapolated.tolist() == [
        False,
        False,
        True,
        True,
        False,
        False,
    ]
    assert model.running == "extrapolate"
    assert "Censored Units      : 0" in repr(model)
    assert "Extrapolated Units  : 2" in repr(model)
    _same_life_fit(model)


def test_censor_takes_the_last_reading_for_units_still_running():
    x, y, i = _data()
    model = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, running="censor"
    )
    times = [40.0, 50.0, 60.0, 70.0, 40.0, 50.0]
    flags = [0, 0, 1, 1, 0, 0]
    np.testing.assert_allclose(model.pseudo_failure_times, times, rtol=1e-12)
    assert model.c.tolist() == flags
    assert not model.extrapolated.any()
    assert "Censored Units      : 2" in repr(model)
    assert "Extrapolated Units  : 0" in repr(model)
    _same_life_fit(model)


def test_censored_units_marks_units_whatever_running_says():
    x, y, i = _data()
    model = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, censored_units=[3]
    )
    assert model.c.tolist() == [0, 0, 0, 1, 0, 0]
    assert model.pseudo_failure_times[3] == 70.0
    assert model.extrapolated.tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert model.censored_units == [3]
    np.testing.assert_allclose(
        model.pseudo_failure_times,
        [40.0, 50.0, 80.0, 70.0, 40.0, 50.0],
        rtol=1e-12,
    )
    _same_life_fit(model)


def test_a_unit_that_failed_another_way_is_censored_when_it_stopped():
    # unit 6 broke at 30 with its reading at 0.25 * 30 = 7.5, short of 10;
    # its path would have crossed at 40, after it stopped
    units = dict(UNITS)
    units[6] = (0.25, 30.0)
    x, y, i = _data(units)
    model = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, censored_units=[6]
    )
    assert model.c[-1] == 1
    assert model.pseudo_failure_times[-1] == 30.0


def test_a_unit_seen_to_cross_is_refused_as_censored():
    x, y, i = _data()
    with pytest.raises(ValueError, match=r"censored_units.*\[0\].*40"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units=[0]
        )


def test_an_unknown_label_is_refused():
    x, y, i = _data()
    with pytest.raises(ValueError, match="censored_units lists \\[42\\]"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units=[42]
        )


def test_running_is_checked():
    x, y, i = _data()
    with pytest.raises(ValueError, match="running"):
        DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD, running="yes")


def test_nothing_left_to_fit_is_refused_saying_what_to_do():
    units = {2: UNITS[2], 3: UNITS[3]}
    x, y, i = _data(units)
    with pytest.raises(ValueError, match="running='extrapolate'"):
        DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD, running="censor")


def test_censor_changes_nothing_where_every_unit_crossed():
    units = {u: UNITS[u] for u in (0, 1, 4, 5)}
    x, y, i = _data(units)
    plain = DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD)
    censor = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, running="censor"
    )
    np.testing.assert_array_equal(
        plain.life_model.params, censor.life_model.params
    )
    assert (plain.c == censor.c).all()


def test_a_falling_path_is_read_from_its_side():
    # degradation falls from 20 towards a threshold of 10
    x, y, i = _data()
    model = DegradationAnalysis.fit(
        x, 20.0 - y, i, threshold=THRESHOLD, running="censor"
    )
    assert model.c.tolist() == [0, 0, 1, 1, 0, 0]


def test_the_choice_is_saved_and_restored():
    x, y, i = _data()
    model = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, running="censor", censored_units=[3]
    )
    back = DegradationModel.from_json(model.to_json())
    assert back.running == "censor"
    assert back.censored_units == [3]
    assert back.extrapolated.tolist() == model.extrapolated.tolist()
    assert repr(back) == repr(model)
    plain = DegradationAnalysis.fit(x, y, i, threshold=THRESHOLD)
    old = plain.to_dict()
    del old["extrapolated"]
    assert "running" not in old and "censored_units" not in old
    restored = DegradationModel.from_dict(old)
    assert restored.extrapolated.tolist() == plain.extrapolated.tolist()


def test_bootstrap_refits_keep_the_censoring():
    # Noisy straight lines; the bootstrap reruns the fit on resampled
    # units with the model's own running and censored_units.
    rng = np.random.default_rng(20261121)
    x, y, i = [], [], []
    for unit in range(10):
        slope = rng.normal(0.2, 0.03)
        last = 60.0 if unit % 2 else 40.0
        t = np.arange(5.0, last + 1, 5.0)
        x.append(t)
        y.append(slope * t + rng.normal(0, 0.1, t.size))
        i.append(np.full(t.size, unit))
    x, y, i = (np.concatenate(v) for v in (x, y, i))
    model = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, running="censor"
    )
    assert (model.c == 1).any()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bounds = model.cb(
            [40.0], method="bootstrap", n_boot=20, random_state=1
        )
    assert np.isfinite(bounds).all()
    assert bounds[0, 0] <= model.sf(40.0) <= bounds[0, 1]
