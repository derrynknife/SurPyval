"""Direct tests for the previously-untested NonParametric entry points:
``from_xrd``, the ``set_lower_limit`` fit option, and ``plot`` with the
non-step interpolation schemes."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import surpyval  # noqa: E402
import surpyval as sp  # noqa: E402
from surpyval import NonParametric  # noqa: E402

# --- from_xrd -------------------------------------------------------------


def test_from_xrd_nelson_aalen_matches_formula():
    x = [1, 2, 3, 4, 5, 6]
    r = [10, 8, 6, 4, 3, 2]
    d = [2, 1, 1, 1, 1, 1]
    model = surpyval.NelsonAalen.from_xrd(x, r, d)
    expected = np.exp(-np.cumsum(np.array(d) / np.array(r)))
    assert np.allclose(model.x, x)
    assert np.allclose(model.R, expected)


def test_from_xrd_kaplan_meier_matches_formula():
    x = [1, 2, 3, 4, 5, 6]
    r = [10, 8, 6, 4, 3, 2]
    d = [2, 1, 1, 1, 1, 1]
    model = surpyval.KaplanMeier.from_xrd(x, r, d)
    expected = np.cumprod(1 - np.array(d) / np.array(r))
    assert np.allclose(model.R, expected)


def test_from_xrd_supports_confidence_bounds_but_not_bootstrap():
    # from_xrd carries r and d, so the Greenwood variance (and cb) is
    # available, but it stores no raw data so the bootstrap cannot run.
    model = surpyval.NelsonAalen.from_xrd(
        [1, 2, 3, 4, 5], [10, 8, 6, 4, 2], [2, 2, 2, 2, 2]
    )
    assert model.greenwood is not None
    assert model.cb([2.0, 3.0]).shape == (2, 2)
    with pytest.raises(ValueError, match="Bootstrap requires the data"):
        model.bootstrap_cb([2.0, 3.0])


def test_from_xrd_rejected_for_turnbull():
    with pytest.raises(ValueError, match="from_xrd with Turnbull"):
        surpyval.Turnbull.from_xrd([1, 2, 3], [3, 2, 1], [1, 1, 1])


# --- set_lower_limit ------------------------------------------------------


def test_set_lower_limit_anchors_curve():
    x = np.array([2.0, 3.0, 4.0, 5.0, 6.0])
    model = surpyval.KaplanMeier.fit(x, set_lower_limit=0.0)
    baseline = surpyval.KaplanMeier.fit(x)
    # A leading point is inserted at the lower limit with no deaths, so the
    # survival there is 1 and the rest of the ladder is unchanged.
    assert model.x[0] == 0.0
    assert model.d[0] == 0
    assert model.R[0] == 1.0
    assert model.x.size == baseline.x.size + 1
    assert np.allclose(model.R[1:], baseline.R)


def test_set_lower_limit_shifts_survival_evaluation():
    x = np.array([2.0, 3.0, 4.0, 5.0])
    model = surpyval.KaplanMeier.fit(x, set_lower_limit=0.5)
    # Between the lower limit and the first event the survival is 1.
    assert np.allclose(model.sf([0.6, 1.0, 1.9]), 1.0)


# --- plot with non-step interpolation -------------------------------------


@pytest.mark.parametrize("interp", ["linear", "cubic"])
def test_plot_with_non_step_interp(interp):
    model = surpyval.KaplanMeier.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    ax = plt.figure().gca()
    returned = model.plot(ax=ax, interp=interp)
    # A curve is drawn (as a line, not a step) and the confidence band as a
    # filled region.
    assert returned is ax
    assert len(ax.lines) >= 1
    assert len(ax.collections) >= 1
    plt.close("all")


def test_plot_non_step_interp_without_bounds():
    model = surpyval.KaplanMeier.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    ax = plt.figure().gca()
    model.plot(ax=ax, interp="linear", plot_bounds=False)
    assert len(ax.collections) == 0
    plt.close("all")


# ---------------------------------------------------------------------------
# ``set_lower_limit``, ``fit_from_ecdf`` and ``success_run``
# argument checks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 2, np.nan])
def test_set_lower_limit_must_be_below_the_data(limit):
    with pytest.raises(ValueError, match="set_lower_limit"):
        sp.KaplanMeier.fit([1, 2, 3], set_lower_limit=limit)


def test_set_lower_limit_below_the_data_still_works():
    model = sp.KaplanMeier.fit([1, 2, 3], set_lower_limit=0)
    np.testing.assert_array_equal(model.x, [0, 1, 2, 3])


@pytest.mark.parametrize(
    "x, R",
    [
        ([2, 1, 3], [0.8, 0.5, 0.1]),
        ([1, 2, 3], [0.5, 0.8, 0.1]),
        ([1, 2, 3], [1.2, 0.5, 0.1]),
        ([1, 2, 3], [0.8, 0.5]),
        ([1, 2, 3], [0.8, np.nan, 0.1]),
    ],
)
def test_fit_from_ecdf_rejects_invalid_curves(x, R):
    with pytest.raises(ValueError):
        NonParametric.fit_from_ecdf(x, R)


def test_fit_from_ecdf_can_plot_without_bounds():
    model = NonParametric.fit_from_ecdf([1, 2, 3], [0.8, 0.5, 0.1])
    fig, ax = plt.subplots()
    model.plot(ax=ax, plot_bounds=False)
    plt.close(fig)
    assert model.get_plot_data(plot_bounds=False)["cbs"] is None
    with pytest.raises(ValueError, match="variance"):
        model.plot(plot_bounds=True)


def test_success_run_needs_a_positive_run():
    with pytest.raises(ValueError, match="'n'"):
        sp.success_run(0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        sp.success_run(5, alpha_ci=1.5)


def test_580_success_run_takes_alpha_ci():
    # #580: success_run was the one bound without alpha_ci (TypeError);
    # confidence= and alpha= still work in v0.23, with a warning.
    assert sp.success_run(1200, alpha_ci=0.1) == pytest.approx(
        0.1 ** (1 / 1200), rel=1e-14
    )
    assert sp.success_run(59) == sp.success_run(59, alpha_ci=0.05)
    with pytest.warns(DeprecationWarning, match="alpha_ci = 1 - confidence"):
        assert sp.success_run(59, 0.95) == pytest.approx(
            sp.success_run(59, alpha_ci=0.05), rel=1e-14
        )
    with pytest.warns(DeprecationWarning, match="use 'alpha_ci'") as w:
        assert sp.success_run(22, alpha=0.1) == sp.success_run(
            22, alpha_ci=0.1
        )
    assert w[0].filename == __file__
    with pytest.raises(ValueError, match="only one of"):
        sp.success_run(5, alpha=0.1, alpha_ci=0.1)
