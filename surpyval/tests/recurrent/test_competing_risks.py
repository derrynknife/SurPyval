import numpy as np
import pytest

from surpyval import RecurrentEventData
from surpyval.recurrent import NonParametricCounting
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
    assert model.mcf(12, "a") == 1.5
    assert model.mcf(12, "b") == 1.0
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


# ---------------------------------------------------------------------------
# HPP as a cause-specific baseline; the cause-specific MCF plot
# draws its bounds on request.
# ---------------------------------------------------------------------------


def test_hpp_works_as_a_cause_specific_baseline_and_from_params():
    from surpyval.recurrent import HPP, CauseSpecificNHPP

    x = [1.0, 2.0, 4.0, 5.0, 3.0, 6.0, 8.0, 9.0, 10.0, 10.0]
    i = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2]
    c = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    e = ["a", "b", "a", "a", "b", "a", "b", "b", None, None]
    model = CauseSpecificNHPP.fit(x, i=i, c=c, e=e, dist=HPP)
    # each cause's rate is its events over the total exposure (20)
    assert model.models["a"].params[0] == pytest.approx(4 / 20)
    assert model.models["b"].params[0] == pytest.approx(4 / 20)

    given = HPP.from_params([0.5])
    assert given.cif(np.array([2.0]))[0] == pytest.approx(1.0)
    assert "given parameters" in repr(given)
    fitted = HPP.fit([1.0, 2.0, 3.0, 4.0], c=[0, 0, 0, 1])
    assert "Fitted by           : MLE" in repr(fitted)


def test_cause_specific_mcf_plot_draws_bounds_on_request():
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    from surpyval.recurrent import CauseSpecificMCF

    x = [1.0, 2.0, 4.0, 5.0, 3.0, 6.0, 8.0, 9.0, 10.0, 10.0]
    i = [1, 1, 1, 1, 2, 2, 2, 2, 1, 2]
    c = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1]
    e = ["a", "b", "a", "a", "b", "a", "b", "b", None, None]
    model = CauseSpecificMCF.fit(x, i=i, c=c, e=e)
    _, ax = plt.subplots()
    model.plot(ax=ax, plot_bounds=False)
    assert len(ax.lines) == 2
    _, ax = plt.subplots()
    model.plot(ax=ax, alpha_ci=0.2)
    # an MCF plus a [lower, upper] pair per cause
    assert len(ax.lines) == 6
    upper = model.models["a"].mcf_cb(model.models["a"].x, alpha_ci=0.2)
    np.testing.assert_allclose(ax.lines[2].get_ydata(), upper[:, 1])
    plt.close("all")


# ---------------------------------------------------------------------------
# The cause-specific MCF carries the robust (Lawless-Nadeau)
# variance per cause.
# ---------------------------------------------------------------------------


# Three items; item 1 has far more events than the others, so the robust
# (Lawless-Nadeau) variance is well above the per-step one.
X = [1, 2, 3, 4, 5, 6, 7, 3, 6, 9, 2, 9]
I = [1] * 7 + [2] * 3 + [3] * 2
C = [0] * 6 + [1] + [0, 0, 1] + [0, 1]


def _lawless_nadeau_by_hand(x, i, c, e, cause):
    """Brute-force Lawless-Nadeau variance of one cause's MCF: each item's
    deviations n_k - d/r, weighted by 1/r while at risk, summed over time
    and then squared."""
    x, i, c, e = map(np.asarray, (x, i, c, e))
    grid = np.unique(x)
    items = np.unique(i)
    exit_ = {k: x[i == k].max() for k in items}
    r = np.array([sum(t <= exit_[k] for k in items) for t in grid])
    counts = {
        k: np.array(
            [
                np.sum((x == t) & (i == k) & (c == 0) & (e == cause))
                for t in grid
            ]
        )
        for k in items
    }
    d = sum(counts.values())
    total = np.zeros(len(grid))
    for k in items:
        at_risk = grid <= exit_[k]
        total += np.cumsum(at_risk * (counts[k] - d / r) / r) ** 2
    return total


def test_cause_specific_mcf_single_cause_matches_overall_mcf():
    # With every event of one cause, the cause-specific MCF is the overall
    # MCF, variance included. It used to carry the per-step variance
    # (0.556 at the end, against the robust 1.556).
    e = ["A" if ci == 0 else None for ci in C]
    overall = NonParametricCounting.fit(X, I, C)
    cs = CauseSpecificMCF.fit(X, I, C, e=e).models["A"]
    np.testing.assert_allclose(cs.mcf_hat, overall.mcf_hat)
    np.testing.assert_allclose(cs.var, overall.var)
    assert cs.var[-1] == pytest.approx(14 / 9)


def test_cause_specific_mcf_robust_variance_per_cause():
    # Other causes' events are non-events for a cause; the risk set is
    # shared.
    e = ["A", "B", "A", "A", "B", "A", None, "B", "A", None, "A", None]
    model = CauseSpecificMCF.fit(X, I, C, e=e)
    for cause in ("A", "B"):
        np.testing.assert_allclose(
            model.models[cause].var,
            _lawless_nadeau_by_hand(X, I, C, e, cause),
        )
