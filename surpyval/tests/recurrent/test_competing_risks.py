import numpy as np

from surpyval import RecurrentEventData
from surpyval.recurrent.competing_risks import CauseSpecificMCF


def _example_data():
    # Two items, two failure modes ('a', 'b'), each ending with a
    # right-censored end-of-observation row (mark None).
    x = [3, 5, 9, 12, 4, 8, 12]
    i = [1, 1, 1, 1, 2, 2, 2]
    c = [0, 0, 0, 1, 0, 0, 1]
    e = ["a", "b", "a", None, "b", "a", None]
    return x, i, c, e


def test_recurrent_event_data_marks_backward_compatible():
    # No marks -> e is None, event_types empty, behaviour unchanged.
    data = RecurrentEventData(
        np.array([1, 2, 3, 1, 2, 3]),
        np.array([1, 1, 1, 2, 2, 2]),
        np.array([0, 0, 1, 0, 0, 1]),
        np.array([1, 1, 1, 1, 1, 1]),
    )
    assert data.e is None
    assert data.event_types == []
    # slicing preserves the (absent) marks
    assert data[0:2].e is None


def test_recurrent_event_data_with_marks():
    x, i, c, e = _example_data()
    data = RecurrentEventData(x, i, c, np.ones_like(x), e)
    assert data.event_types == ["a", "b"]
    # slicing keeps marks aligned
    assert list(data[0:2].e) == ["a", "b"]


def test_cause_specific_xrd_shares_risk_set():
    x, i, c, e = _example_data()
    data = RecurrentEventData(x, i, c, np.ones_like(x), e)

    _, r_all, _ = data.to_xrd()
    xa, ra, da = data.to_cause_specific_xrd("a")
    _, rb, db = data.to_cause_specific_xrd("b")

    # Risk set is shared across causes
    np.testing.assert_array_equal(ra, r_all)
    np.testing.assert_array_equal(rb, r_all)
    # Cause counts sum to total observed-event counts
    _, _, d_all = data.to_xrd()
    np.testing.assert_array_equal(da + db, d_all)


def test_cause_specific_mcf_fit_and_eval():
    x, i, c, e = _example_data()
    model = CauseSpecificMCF.fit(x, i, c, e=e)

    assert model.event_types == ["a", "b"]
    # MCF is the expected number of events per item (events / shared risk
    # set). 'a': events at 3, 8, 9 over a risk set of 2 -> 1.5; 'b': events
    # at 4, 5 -> 1.0. Evaluated at the last observation time (12).
    assert model.mcf(12, "a")[0] == 1.5
    assert model.mcf(12, "b")[0] == 1.0
    # confidence bounds return a finite interval inside observation
    cb = model.mcf_cb(9, "a")
    assert np.isfinite(cb).all()


def test_cause_specific_mcf_requires_marks():
    x, i, c, _ = _example_data()
    try:
        CauseSpecificMCF.fit(x, i, c)
    except ValueError as err:
        assert "event types" in str(err).lower()
    else:
        raise AssertionError("expected ValueError when marks are missing")
