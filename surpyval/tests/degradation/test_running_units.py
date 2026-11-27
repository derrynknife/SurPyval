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


# A unit known to be still running at a time after its last reading:
# ``censored_units={unit: time}`` censors it there; ``"data"`` reads the
# times from a SurpyvalData's right-censored units.


def _life_fit_of(model, times, flags):
    """The model's unit times and flags are ``times`` and ``flags``, and
    its life fit is the Weibull fit of them."""
    np.testing.assert_allclose(model.pseudo_failure_times, times, rtol=1e-12)
    assert model.c.tolist() == flags
    _same_life_fit(model)


def test_a_unit_running_after_its_last_reading_is_censored_then():
    # unit 3: last reading at 70, still running at 90, path crosses at 100
    x, y, i = _data()
    model = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, censored_units={3: 90.0}
    )
    _life_fit_of(
        model, [40.0, 50.0, 80.0, 90.0, 40.0, 50.0], [0, 0, 0, 1, 0, 0]
    )
    assert model.censored_units == [3]
    assert model.censored_at == {3: 90.0}
    assert model.extrapolated.tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    # the later time moves the life fit: censored at 70, the fit differs
    at_last = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, censored_units=[3]
    )
    assert not np.allclose(model.life_model.params, at_last.life_model.params)


def test_a_time_of_none_or_the_last_reading_is_the_list_form():
    x, y, i = _data()
    listed = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, censored_units=[2, 3]
    )
    for given in ({2: None, 3: None}, {2: 60.0, 3: np.float64(70.0)}):
        model = DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units=given
        )
        np.testing.assert_array_equal(
            model.pseudo_failure_times, listed.pseudo_failure_times
        )
        np.testing.assert_array_equal(model.c, listed.c)
        np.testing.assert_array_equal(
            model.life_model.params, listed.life_model.params
        )
    assert listed.censored_at is None


@pytest.mark.parametrize("past", [0.0, 5.0])
def test_a_time_at_or_past_the_crossing_is_refused(past):
    # unit 2's path crosses at 80, after its last reading at 60: running
    # at its crossing or later, by its path it failed unseen. "At" is the
    # crossing the fit computes (80 to within the least squares' last
    # bits, which differ by platform: just above 80 on x86-64 Linux), not the
    # literal 80.
    x, y, i = _data()
    crossing = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD
    ).pseudo_failure_times[2]
    assert crossing == pytest.approx(CROSSING[2], rel=1e-12)
    with pytest.raises(ValueError, match=r"censored_units.*\[2\].*failed"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units={2: crossing + past}
        )


def test_a_time_before_the_last_reading_is_refused():
    x, y, i = _data()
    with pytest.raises(ValueError, match=r"before their last readings"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units={3: 65.0}
        )


@pytest.mark.parametrize("time", [np.nan, np.inf, "soon", True])
def test_a_time_that_is_not_a_finite_number_is_refused(time):
    x, y, i = _data()
    with pytest.raises(ValueError, match="finite number"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units={3: time}
        )


def test_a_unit_seen_to_cross_is_refused_with_a_time_too():
    x, y, i = _data()
    with pytest.raises(ValueError, match=r"by their last reading"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units={0: 70.0}
        )


def test_a_path_that_never_crosses_is_censored_at_the_time_given():
    # unit 6 falls away from the threshold: it never reaches it
    units = dict(UNITS)
    units[6] = (-0.01, 60.0)
    x, y, i = _data(units)
    with pytest.warns(UserWarning, match="never reach"):
        model = DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units={6: 120.0}
        )
    assert model.c[-1] == 1
    assert model.pseudo_failure_times[-1] == 120.0


def test_a_time_is_refused_on_the_stress_clock():
    x, y, i = _data()
    Z = np.where(x > 30, 2.0, 1.0)
    with pytest.raises(ValueError, match="acceleration='clock'"):
        DegradationAnalysis.fit(
            x,
            y,
            i,
            threshold=THRESHOLD,
            Z=Z,
            acceleration="clock",
            censored_units={3: 90.0},
        )


def test_the_running_at_times_are_saved_and_restored():
    x, y, i = _data()
    model = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, censored_units={3: 90.0, 2: None}
    )
    assert model.censored_units == [2, 3]
    assert model.censored_at == {3: 90.0}
    back = DegradationModel.from_json(model.to_json())
    assert back.censored_units == [2, 3]
    assert back.censored_at == {3: 90.0}
    assert repr(back) == repr(model)
    assert (
        "censored_at"
        not in DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units=[3]
        ).to_dict()
    )


def test_bootstrap_refits_keep_the_running_at_times():
    rng = np.random.default_rng(20261125)
    x, y, i = [], [], []
    for unit in range(10):
        slope = rng.normal(0.2, 0.03)
        last = 60.0 if unit % 2 else 30.0
        t = np.arange(5.0, last + 1, 5.0)
        x.append(t)
        y.append(slope * t + rng.normal(0, 0.1, t.size))
        i.append(np.full(t.size, unit))
    x, y, i = (np.concatenate(v) for v in (x, y, i))
    model = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, censored_units={0: 35.0, 2: 34.0}
    )
    assert model.pseudo_failure_times[[0, 2]].tolist() == [35.0, 34.0]
    seen = []
    original = DegradationAnalysis.fit

    def spy(*args, **kwargs):
        seen.append(kwargs.get("censored_units"))
        return original(*args, **kwargs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(DegradationAnalysis, "fit", spy)
            bounds = model.cb(
                [40.0], method="bootstrap", n_boot=20, random_state=1
            )
    assert np.isfinite(bounds).all()
    times = {t for given in seen if given for t in given.values()}
    assert times <= {35.0, 34.0} and times


def _readings_data(c, x_life):
    from surpyval import Measurements

    x, y, i = _data()
    store = {}
    for t, v, u in zip(x, y, i):
        store.setdefault(int(u), {})[float(t)] = float(v)
    readings = Measurements({"wear": store})
    return sp.SurpyvalData(x_life, c=c, i=list(UNITS), m=readings)


def test_data_reads_the_right_censored_units_when_asked():
    life = _readings_data(
        c=[0, 0, 0, 1, 0, 0], x_life=[40.0, 50.0, 80.0, 90.0, 40.0, 50.0]
    )
    model = DegradationAnalysis.fit(
        life, threshold=THRESHOLD, censored_units="data"
    )
    x, y, i = _data()
    by_hand = DegradationAnalysis.fit(
        x, y, i, threshold=THRESHOLD, censored_units={3: 90.0}
    )
    np.testing.assert_array_equal(
        model.pseudo_failure_times, by_hand.pseudo_failure_times
    )
    np.testing.assert_array_equal(model.c, by_hand.c)
    np.testing.assert_allclose(
        model.life_model.params, by_hand.life_model.params, rtol=1e-12
    )
    assert model.censored_at == {3: 90.0}
    # without the opt-in the data's censoring is not read, as before
    plain = DegradationAnalysis.fit(life, threshold=THRESHOLD)
    assert (plain.c == 0).all() and plain.censored_units is None


def test_data_with_no_right_censored_units_censors_none():
    life = _readings_data(c=[0] * 6, x_life=[40.0] * 6)
    model = DegradationAnalysis.fit(
        life, threshold=THRESHOLD, censored_units="data"
    )
    assert model.censored_units is None and (model.c == 0).all()


def test_data_running_past_the_crossing_is_refused():
    # unit 2 recorded as running at 85, but its path crosses at 80
    life = _readings_data(c=[0, 0, 1, 0, 0, 0], x_life=[85.0] * 6)
    with pytest.raises(ValueError, match=r"\[2\].*failed unseen"):
        DegradationAnalysis.fit(
            life, threshold=THRESHOLD, censored_units="data"
        )


def test_data_is_refused_without_a_surpyval_data():
    x, y, i = _data()
    with pytest.raises(ValueError, match="SurpyvalData"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units="data"
        )
    life = _readings_data(c=[1] * 6, x_life=[100.0] * 6)
    with pytest.raises(ValueError, match="Measurements has none"):
        DegradationAnalysis.fit(
            life.m, threshold=THRESHOLD, censored_units="data"
        )
    with pytest.raises(ValueError, match="list of units"):
        DegradationAnalysis.fit(
            x, y, i, threshold=THRESHOLD, censored_units="all"
        )
