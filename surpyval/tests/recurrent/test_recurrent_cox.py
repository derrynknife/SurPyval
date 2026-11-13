"""The semi-parametric recurrent-event regressions: ``AndersenGill``,
``PrenticeWilliamsPeterson`` and ``WeiLinWeissfeld``.

Hand-checked cases (a closed form, and each layout written out), recovery
of known coefficients from simulated data, the entry points and the
refusals. The comparison with R is in
``tests/reference/test_recurrent_cox.py``.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

import surpyval
from surpyval import CoxPH
from surpyval.recurrent import (
    AndersenGill,
    PrenticeWilliamsPeterson,
    WeiLinWeissfeld,
)
from surpyval.recurrent.regression import RecurrentCoxModel
from surpyval.recurrent.regression.recurrent_cox import (
    PWP_GAP,
    PWP_TOTAL,
    WLW,
    _layout,
    rows_from_event_times,
)

# -- a closed form ----------------------------------------------------------


def _four_items(counts):
    """Four items watched over (0, 100], items 0 and 1 with Z = 0 and
    items 2 and 3 with Z = 1, with ``counts`` events each at distinct
    times."""
    x, i, c, Z = [], [], [], []
    time = 0.0
    for item, count in enumerate(counts):
        for _ in range(count):
            time += 1.0
            x.append(time + item / 10)
            i.append(item)
            c.append(0)
            Z.append(item // 2)
        x.append(100.0)
        i.append(item)
        c.append(1)
        Z.append(item // 2)
    return np.array(x), np.array(Z, dtype=float), np.array(i), np.array(c)


@pytest.mark.parametrize("counts", [(2, 1, 3, 3), (2, 1, 4, 2), (1, 3, 5, 3)])
def test_andersen_gill_closed_form(counts):
    """With two items at each level of one 0/1 covariate, all at risk
    throughout and no ties, every event's risk set is the same, so with
    theta = exp(beta) the partial likelihood is
    (theta / (2 theta + 2))^a (1 / (2 theta + 2))^b for a events at
    Z = 1 and b at Z = 0:

    - theta = a / b;
    - the information is N p (1 - p), p = theta / (1 + theta), N = a + b;
    - an item's score residual is (1 - p)(d - N p / 2) at Z = 1 and
      -p (d - N (1 - p) / 2) at Z = 0, d its events, and the robust
      variance their sum of squares over the information squared;
    - the Breslow baseline rises by 1 / (2 theta + 2) at each event.
    """
    x, Z, i, c = _four_items(counts)
    model = AndersenGill.fit(x, Z, i=i, c=c)
    d = np.array(counts, dtype=float)
    b, a = d[:2].sum(), d[2:].sum()
    total = a + b
    theta = a / b
    p = theta / (1 + theta)
    information = total * p * (1 - p)
    residual = np.r_[
        -p * (d[:2] - total * (1 - p) / 2), (1 - p) * (d[2:] - total * p / 2)
    ]
    assert model.params == pytest.approx([np.log(theta)], rel=1e-9)
    assert model.naive_covariance() == pytest.approx(
        np.array([[1 / information]]), rel=1e-8
    )
    assert model.covariance() == pytest.approx(
        np.array([[(residual**2).sum() / information**2]]), rel=1e-7
    )
    events = np.sort(x[c == 0])
    assert model.cif(events, [0.0]) == pytest.approx(
        np.arange(1, total + 1) / (2 * theta + 2), rel=1e-9
    )
    assert model.cif(events, [1.0]) == pytest.approx(
        theta * np.arange(1, total + 1) / (2 * theta + 2), rel=1e-9
    )
    assert model.n_items == 4 and model.n_events == total


# -- the layouts, written out -----------------------------------------------

# Item "a": events at 4 and 9, watched to 12. Item "b": an event at 6,
# watched to 10. Item "c": events at 2, 5 and 7 (its last row).
LOG = {
    "x": [4, 9, 12, 6, 10, 2, 5, 7],
    "i": ["a", "a", "a", "b", "b", "c", "c", "c"],
    "c": [0, 0, 1, 0, 1, 0, 0, 0],
    "Z": [[1.0], [1.0], [1.0], [0.0], [0.0], [2.0], [2.0], [2.0]],
}


def _log_rows():
    return rows_from_event_times(LOG["x"], LOG["Z"], LOG["i"], LOG["c"])


def test_event_log_becomes_start_stop_rows():
    rows = _log_rows()
    assert rows["xl"].tolist() == [0, 4, 9, 0, 6, 0, 2, 5]
    assert rows["xr"].tolist() == [4, 9, 12, 6, 10, 2, 5, 7]
    assert rows["c"].tolist() == LOG["c"]


def test_total_time_layout_strata_are_event_numbers():
    layout = _layout(PWP_TOTAL, _log_rows(), None)
    assert layout["strata"].tolist() == [1, 2, 3, 1, 2, 1, 2, 3]
    assert layout["x"].tolist() == [4, 9, 12, 6, 10, 2, 5, 7]
    assert layout["tl"].tolist() == [0, 4, 9, 0, 6, 0, 2, 5]


def test_gap_time_layout_restarts_the_clock_at_each_event():
    layout = _layout(PWP_GAP, _log_rows(), None)
    assert layout["strata"].tolist() == [1, 2, 3, 1, 2, 1, 2, 3]
    assert layout["x"].tolist() == [4, 5, 3, 6, 4, 2, 3, 2]
    assert layout["tl"] is None


def test_gap_time_layout_of_rows_split_between_events():
    # Item "a" with its second spell cut at 6 (a covariate changed), and
    # entering at 1: the clock runs from the entry, then from the event.
    rows = {
        "i": np.array(["a"] * 4),
        "xl": np.array([1.0, 4.0, 6.0, 9.0]),
        "xr": np.array([4.0, 6.0, 9.0, 12.0]),
        "c": np.array([0, 1, 0, 1]),
        "Z": np.array([[1.0], [1.0], [3.0], [3.0]]),
    }
    layout = _layout(PWP_GAP, rows, None)
    assert layout["strata"].tolist() == [1, 2, 2, 3]
    assert layout["tl"].tolist() == [0, 0, 2, 0]
    assert layout["x"].tolist() == [3, 2, 5, 3]


def test_max_events_leaves_out_what_follows():
    layout = _layout(PWP_TOTAL, _log_rows(), 2)
    assert layout["strata"].tolist() == [1, 2, 1, 2, 1, 2]
    assert layout["x"].tolist() == [4, 9, 6, 10, 2, 5]


def test_marginal_layout_has_a_row_per_item_and_event_number():
    layout = _layout(WLW, _log_rows(), None)
    # a: 4, 9, then censored at 12; b: 6, then censored at 10 twice;
    # c: 2, 5, 7.
    assert layout["x"].tolist() == [4, 9, 12, 6, 10, 10, 2, 5, 7]
    assert layout["c"].tolist() == [0, 0, 1, 0, 1, 1, 0, 0, 0]
    assert layout["strata"].tolist() == [1, 2, 3] * 3
    assert layout["cluster"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert layout["Z"].ravel().tolist() == [1, 1, 1, 0, 0, 0, 2, 2, 2]
    assert layout["tl"] is None
    two = _layout(WLW, _log_rows(), 2)
    assert two["x"].tolist() == [4, 9, 6, 10, 2, 5]
    assert two["c"].tolist() == [0, 0, 0, 1, 0, 0]


# -- simulated data ---------------------------------------------------------


def _poisson_items(rng, size, beta, frailty_variance, end=10.0):
    """Items whose events are a Poisson process of intensity
    0.3 t^0.5 w exp(beta'Z) over (0, end], w a gamma frailty of mean 1."""
    Z = np.column_stack(
        [rng.integers(0, 2, size).astype(float), rng.normal(size=size)]
    )
    w = np.ones(size)
    if frailty_variance:
        w = rng.gamma(1 / frailty_variance, frailty_variance, size)
    x, i, c, rows = [], [], [], []
    for item in range(size):
        mean = 0.2 * end**1.5 * w[item] * np.exp(Z[item] @ beta)
        count = rng.poisson(mean)
        # Given the count, the times are draws from cif / cif(end).
        times = np.sort(end * rng.uniform(size=count) ** (1 / 1.5))
        x += [*times, end]
        i += [item] * (count + 1)
        c += [0] * count + [1]
        rows += [Z[item]] * (count + 1)
    return np.array(x), np.array(rows), np.array(i), np.array(c)


def test_andersen_gill_recovers_the_coefficients():
    beta = np.array([0.5, -0.3])
    rng = np.random.default_rng(20261111)
    x, Z, i, c = _poisson_items(rng, 400, beta, 0.0)
    model = AndersenGill.fit(x, Z, i=i, c=c)
    error = np.abs(model.params - beta) / model.standard_errors()
    assert (error < 3).all(), (model.params, model.standard_errors())
    # Poisson events: the two covariances estimate the same thing.
    ratio = model.standard_errors() / np.sqrt(
        np.diag(model.naive_covariance())
    )
    assert ((ratio > 0.8) & (ratio < 1.25)).all(), ratio
    # The baseline: 0.2 t^1.5 at Z = 0.
    assert model.cif(9.0, [0.0, 0.0]) == pytest.approx(0.2 * 9.0**1.5, rel=0.1)


def test_robust_errors_widen_with_unexplained_differences_between_items():
    beta = np.array([0.5, -0.3])
    rng = np.random.default_rng(20261112)
    x, Z, i, c = _poisson_items(rng, 400, beta, 1.0)
    model = AndersenGill.fit(x, Z, i=i, c=c)
    naive = np.sqrt(np.diag(model.naive_covariance()))
    assert (model.standard_errors() > 1.5 * naive).all()
    error = np.abs(model.params - beta) / model.standard_errors()
    assert (error < 3).all(), (model.params, model.standard_errors())


def _renewal_items(rng, size, beta, end=8.0):
    """Items whose k-th gap is Weibull with a shape that depends on k
    (1.5 for the first, 0.8 after) and a scale divided by
    exp(beta Z / shape): proportional hazards in the gap, a different
    baseline for each event number."""
    z = rng.integers(0, 2, size).astype(float)
    x, i, c, rows = [], [], [], []
    for item in range(size):
        age, number = 0.0, 1
        while True:
            shape = 1.5 if number == 1 else 0.8
            scale = 3.0 * np.exp(-beta * z[item] / shape)
            age += scale * rng.weibull(shape)
            if age >= end:
                break
            x.append(age)
            i.append(item)
            c.append(0)
            rows.append(z[item])
            number += 1
        x.append(end)
        i.append(item)
        c.append(1)
        rows.append(z[item])
    return np.array(x), np.array(rows), np.array(i), np.array(c)


def test_gap_time_model_recovers_the_coefficient_of_a_renewal_process():
    rng = np.random.default_rng(20261113)
    x, Z, i, c = _renewal_items(rng, 500, 0.6)
    model = PrenticeWilliamsPeterson.fit(
        x, Z, i=i, c=c, time_scale="gap", max_events=4
    )
    error = abs(model.params[0] - 0.6) / model.standard_errors()[0]
    assert error < 3, (model.params, model.standard_errors())
    assert model.strata == [1, 2, 3, 4]
    # The first gap's survival at Z = 0: exp(-(t / 3)^1.5).
    assert model.sf(2.0, [0.0], stratum=1) == pytest.approx(
        np.exp(-((2.0 / 3.0) ** 1.5)), abs=0.06
    )


def test_marginal_model_of_the_first_event_is_the_cox_model_of_it():
    rng = np.random.default_rng(20261114)
    x, Z, i, c = _renewal_items(rng, 200, 0.6)
    model = WeiLinWeissfeld.fit(x, Z, i=i, c=c, max_events=1)
    first = np.r_[True, i[1:] != i[:-1]]
    cox = CoxPH.fit(x[first], Z[first], c=c[first])
    assert model.params == pytest.approx(cox.params, rel=1e-9)
    assert model.naive_covariance() == pytest.approx(
        cox.covariance(), rel=1e-9
    )
    assert model.covariance() == pytest.approx(
        cox.robust_covariance(), rel=1e-8
    )
    error = abs(model.params[0] - 0.6) / model.standard_errors()[0]
    assert error < 3, (model.params, model.standard_errors())


# -- entry points -----------------------------------------------------------


def _small():
    rng = np.random.default_rng(7)
    return _poisson_items(rng, 40, np.array([0.5, -0.3]), 0.5)


FITTERS = {
    "ag": (AndersenGill, {}),
    "pwp_total": (PrenticeWilliamsPeterson, {"max_events": 3}),
    "pwp_gap": (
        PrenticeWilliamsPeterson,
        {"max_events": 3, "time_scale": "gap"},
    ),
    "wlw": (WeiLinWeissfeld, {"max_events": 3}),
}


@pytest.mark.parametrize("name", list(FITTERS))
def test_row_order_does_not_matter(name):
    fitter, options = FITTERS[name]
    x, Z, i, c = _small()
    model = fitter.fit(x, Z, i=i, c=c, **options)
    order = np.random.default_rng(1).permutation(x.shape[0])
    shuffled = fitter.fit(
        x[order], Z[order], i=i[order], c=c[order], **options
    )
    assert shuffled.params == pytest.approx(model.params, rel=1e-9)
    assert shuffled.covariance() == pytest.approx(model.covariance(), rel=1e-8)


@pytest.mark.parametrize("name", list(FITTERS))
def test_round_trips(name):
    fitter, options = FITTERS[name]
    x, Z, i, c = _small()
    model = fitter.fit(x, Z, i=i, c=c, unit="months", **options)
    stratum = None if name == "ag" else 2
    saved = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    for restored in (
        surpyval.from_dict(saved),
        RecurrentCoxModel.from_dict(saved),
        pickle.loads(pickle.dumps(model)),
    ):
        assert restored.kind == model.kind
        assert restored.unit == model.unit == "months"
        assert restored.strata == model.strata
        assert restored.max_events == model.max_events
        assert (restored.n_items, restored.n_events) == (
            model.n_items,
            model.n_events,
        )
        np.testing.assert_array_equal(restored.params, model.params)
        np.testing.assert_array_equal(
            restored.covariance(), model.covariance()
        )
        np.testing.assert_array_equal(
            restored.naive_covariance(), model.naive_covariance()
        )
        np.testing.assert_array_equal(
            restored.cif([1.0, 5.0], [1.0, 0.2], stratum=stratum),
            model.cif([1.0, 5.0], [1.0, 0.2], stratum=stratum),
        )
        pd.testing.assert_frame_equal(restored.summary(), model.summary())
        assert repr(restored) == repr(model)
    # The diagnostics need the layout's rows, which are not saved: the
    # restored model says so and names its own fitter, not CoxPH.
    with pytest.raises(ValueError, match="data") as err:
        surpyval.from_dict(saved).diagnose()
    assert type(fitter).__name__ in str(err.value)
    assert "CoxPH" not in str(err.value)


def test_start_stop_rows_are_the_cox_fit_of_them():
    # Delayed entry, a gap in observation and a covariate that changes.
    i = [1, 1, 1, 2, 2, 3, 3, 3, 4, 4]
    xl = [2, 5, 9, 0, 6, 0, 3, 8, 1, 4]
    xr = [5, 8, 12, 6, 11, 3, 7, 12, 4, 10]
    c = [0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
    Z = [0.2, 0.9, 0.9, 1.5, 1.1, 0.4, 0.1, 0.8, 1.9, 1.3]
    model = AndersenGill.fit_tvc(i, xl, xr, c, Z)
    cox = CoxPH.fit(xr, Z, c=c, tl=xl)
    assert model.params == pytest.approx(cox.params, rel=1e-12)
    assert model.covariance() == pytest.approx(
        cox.robust_covariance(i), rel=1e-12
    )
    frame = pd.DataFrame({"i": i, "xl": xl, "xr": xr, "c": c, "z": Z})
    from_df = AndersenGill.fit_tvc_from_df(
        frame, i_col="i", xl_col="xl", xr_col="xr", c_col="c", Z_cols="z"
    )
    assert from_df.params == pytest.approx(model.params, rel=1e-12)
    assert from_df.parameter_names == ["z"]


def test_covariates_may_change_between_an_items_rows():
    # Z on a row is in force over the interval ending at the row.
    x = [3, 7, 10, 4, 10, 2, 6, 9, 10]
    i = [1, 1, 1, 2, 2, 3, 3, 3, 3]
    c = [0, 0, 1, 0, 1, 0, 0, 0, 1]
    Z = [0, 1, 1, 1, 0, 0, 0, 1, 1]
    model = AndersenGill.fit(x, Z, i=i, c=c)
    xl = [0, 3, 7, 0, 4, 0, 2, 6, 9]
    assert model.params == pytest.approx(
        AndersenGill.fit_tvc(i, xl, x, c, Z).params, rel=1e-12
    )


def test_names_come_from_a_data_frame():
    x, Z, i, c = _small()
    frame = pd.DataFrame(Z, columns=["coated", "load"])
    model = AndersenGill.fit(x, frame, i=i, c=c)
    assert model.parameter_names == ["coated", "load"]
    assert model.summary().index.tolist() == ["coated", "load"]
    assert "coated" in repr(model)


def test_summary_columns():
    x, Z, i, c = _small()
    table = AndersenGill.fit(x, Z, i=i, c=c).summary(alpha_ci=0.1)
    assert table.columns.tolist() == [
        "coef",
        "exp(coef)",
        "se(coef)",
        "naive se(coef)",
        "coef lower 90%",
        "coef upper 90%",
        "exp(coef) lower 90%",
        "exp(coef) upper 90%",
        "z",
        "p",
    ]
    assert (table["coef lower 90%"] < table["coef"]).all()


def test_an_end_of_observation_at_the_last_event_adds_nothing():
    x, Z, i, c = _four_items((2, 1, 3, 3))
    plain = AndersenGill.fit(x, Z, i=i, c=c)
    # Item 0 also "censored" at the time of its last row
    again = AndersenGill.fit(
        np.r_[x, 100.0], np.r_[Z, 0.0], i=np.r_[i, 0], c=np.r_[c, 1]
    )
    assert again.params == pytest.approx(plain.params, rel=1e-12)


def test_plot_draws_a_line_per_baseline():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, Z, i, c = _small()
    fig, ax = plt.subplots()
    AndersenGill.fit(x, Z, i=i, c=c).plot(ax=ax)
    assert len(ax.lines) == 1
    fig, ax = plt.subplots()
    WeiLinWeissfeld.fit(x, Z, i=i, c=c, max_events=3).plot(ax=ax)
    assert len(ax.lines) == 3
    plt.close("all")


def test_diagnose_is_the_cox_fits():
    x, Z, i, c = _small()
    model = AndersenGill.fit(x, Z, i=i, c=c)
    assert type(model.diagnose()).__name__ == "Diagnosis"


# -- refusals ---------------------------------------------------------------


def _refused(match, call):
    with pytest.raises(ValueError, match=match):
        call()


def test_refusals_name_the_argument_and_the_fix():
    x, Z, i, c = (
        [3, 7, 10, 4, 10],
        [0, 0, 0, 1, 1],
        [1, 1, 1, 2, 2],
        [0, 0, 1, 0, 1],
    )
    fit = AndersenGill.fit
    _refused(
        "`c` must be 0 .* or 1", lambda: fit(x, Z, i=i, c=[0, 2, 1, 0, 1])
    )
    _refused("`n` must be 1", lambda: fit(x, Z, i=i, c=c, n=[1, 2, 1, 1, 1]))
    _refused("previous row", lambda: fit([3, 3, 10, 4, 10], Z, i=i, c=c))
    _refused("`Z` has 1 row", lambda: fit(x, [0, np.nan, 0, 1, 1], i=i, c=c))
    _refused(
        "`x` has 1 missing", lambda: fit([3, np.nan, 10, 4, 10], Z, i=i, c=c)
    )
    _refused("`i` has 1 missing", lambda: fit(x, Z, i=[1, 1, None, 2, 2], c=c))
    _refused(
        "`i` must have one value per row", lambda: fit(x, Z, i=[1, 2], c=c)
    )
    _refused(
        "`x` has 1 negative", lambda: fit([-3, 7, 10, 4, 10], Z, i=i, c=c)
    )
    _refused("no events", lambda: fit(x, Z, i=i, c=[1, 1, 1, 1, 1]))
    _refused(
        "`time_scale` must be 'total'",
        lambda: PrenticeWilliamsPeterson.fit(
            x, Z, i=i, c=c, time_scale="gaps"
        ),
    )
    _refused(
        "`max_events` must be a whole number",
        lambda: WeiLinWeissfeld.fit(x, Z, i=i, c=c, max_events=0),
    )
    _refused(
        "constant within an item",
        lambda: WeiLinWeissfeld.fit(x, [0, 1, 1, 1, 1], i=i, c=c),
    )
    tvc = AndersenGill.fit_tvc
    _refused("`xr <= xl`", lambda: tvc([1, 1], [0, 5], [5, 5], [0, 1], [0, 1]))
    _refused("overlap", lambda: tvc([1, 1], [0, 4], [5, 8], [0, 1], [0, 1]))


def test_prediction_refusals():
    x, Z, i, c = _small()
    ag = AndersenGill.fit(x, Z, i=i, c=c)
    wlw = WeiLinWeissfeld.fit(x, Z, i=i, c=c, max_events=2)
    total = PrenticeWilliamsPeterson.fit(x, Z, i=i, c=c, max_events=2)
    _refused("Leave it out", lambda: ag.cif(1.0, [0, 0], stratum=1))
    _refused(r"one of \[1, 2\]", lambda: wlw.cif(1.0, [0, 0]))
    _refused(r"one of \[1, 2\]", lambda: wlw.cif(1.0, [0, 0], stratum=3))
    _refused(
        "sf is not defined for the Andersen-Gill", lambda: ag.sf(1.0, [0, 0])
    )
    _refused(
        "sf is not defined for the Prentice",
        lambda: total.sf(1.0, [0, 0], stratum=1),
    )
    with pytest.raises(NotImplementedError, match="summary\\(\\)"):
        ag.cif_cb([1.0], [0, 0])


def test_no_finite_maximum_is_said_and_has_no_errors():
    # The covariate is 1 exactly where there are no events.
    x, Z, i, c = _four_items((2, 1, 0, 0))
    with pytest.warns(UserWarning, match="No finite maximum"):
        model = AndersenGill.fit(x, Z, i=i, c=c)
    assert model.maximum == "no finite maximum"
    assert np.isnan(model.standard_errors()).all()
