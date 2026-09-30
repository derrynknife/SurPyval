"""A fitted model's printout says what data it was fitted to (#508).

A censoring flag read backwards (a "1 = failed" column passed as ``c``)
fits without complaint and looks like a plausible wear-out fit; the
printout's "Data" line shows the inversion at once. The counts are units,
weighted by ``n``.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.utils.data_summary import data_summary


def _issue_data():
    rng = np.random.default_rng(1)
    t = rng.weibull(1.8, 60) * 1000
    end = rng.uniform(800, 2500, 60)
    x = np.minimum(t, end)
    failed = (t <= end).astype(int)  # 51 of 60 failed
    return x, failed


def test_inverted_flag_is_visible_in_the_printout():
    x, failed = _issue_data()
    wrong = repr(sp.Weibull.fit(x, c=failed))
    right = repr(sp.Weibull.fit(x, c=1 - failed))
    assert (
        "Data                : 60 units: 9 events at 9 unique times, "
        "51 right censored"
    ) in (wrong)
    assert (
        "Data                : 60 units: 51 events at 51 unique times, "
        "9 right censored" in right
    )


def test_counts_are_units_weighted_by_n():
    model = sp.Weibull.fit([1, 2, 3, 4], c=[0, 0, 1, 0], n=[3, 1, 10, 2])
    assert (
        "Data                : 16 units: 6 events at 3 unique times, "
        "10 right censored"
    ) in (repr(model))


def test_every_kind_of_censoring_and_truncation_is_counted():
    model = sp.Weibull.fit(
        x=[[1, 1], [2, 2], [3, 3], [4, 5], [6, 6]],
        c=[0, 1, -1, 2, 0],
        tl=[0, 0, 0.5, 0, 1],
        tr=[np.inf, np.inf, np.inf, np.inf, 20],
    )
    assert (
        "Data                : 5 units: 2 events at 2 unique times, "
        "1 right censored, "
        "1 left censored, 1 interval censored; 2 left truncated, "
        "1 right truncated"
    ) in repr(model)


def test_a_truncation_at_the_support_edge_truncates_nothing():
    # tl = 0 for a distribution on (0, inf) is not a truncation.
    model = sp.Weibull.fit([1, 2, 3, 4], tl=0)
    assert repr(model).count("truncated") == 0


def test_a_model_from_parameters_has_no_data_line():
    assert "Data" not in repr(sp.Weibull.from_params([10, 2]))


def test_non_parametric_printout_has_the_data_line():
    x, failed = _issue_data()
    km = sp.KaplanMeier.fit(x, c=failed)
    assert (
        "Data             : 60 units: 9 events at 9 unique times, "
        "51 right censored"
    ) in (repr(km))
    tb = sp.Turnbull.fit(xl=[1, 2, 3], xr=[2, 4, 5])
    assert "Data             : 3 units: 0 events, 3 interval censored" in (
        repr(tb)
    )
    na = sp.NelsonAalen.fit([1, 2, 3, 4], tl=[0, 0, 1, 1])
    assert (
        "Data             : 4 units: 4 events at 4 unique times; "
        "2 left truncated" in repr(na)
    )


@pytest.fixture(scope="module")
def rossi():
    from surpyval.datasets import load_rossi_static

    df = load_rossi_static()
    Z = df[["fin", "age", "prio"]].to_numpy(dtype=float)
    return df["week"].to_numpy(dtype=float), Z, 1 - df["arrest"].to_numpy()


LINE = (
    "Data                : 432 units: 114 events at 49 unique times, "
    "318 right censored"
)


@pytest.mark.parametrize(
    "fitter", [sp.CoxPH, sp.WeibullPH, sp.WeibullAFT, sp.BuckleyJames]
)
def test_regression_printout_has_the_data_line(rossi, fitter):
    x, Z, c = rossi
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = fitter.fit(x, Z, c)
    assert LINE in repr(model)


def test_cox_start_stop_counts_units_and_intervals():
    model = sp.CoxPH.fit_tvc(
        [0, 0, 1, 2, 2, 3, 4, 4, 5, 6],
        [0, 2, 0, 0, 1, 0, 0, 3, 0, 0],
        [2, 5, 3, 1, 4, 6, 3, 7, 2, 8],
        [1, 0, 0, 1, 0, 1, 1, 0, 0, 1],
        [0, 1, 0, 0, 1, 0, 0, 1, 1, 0],
    )
    assert (
        "Data                : 7 units in 10 start-stop intervals: " "5 events"
    ) in repr(model)


def test_mixture_printout_has_the_data_line():
    x = [1, 2, 3, 4, 5, 6, 6, 7, 8, 10, 13, 15, 16, 17, 17, 18, 19]
    model = sp.MixtureModel.fit(x, dist=sp.Weibull, m=2)
    assert (
        "Data                : 17 units: 17 events at 15 unique times"
        in repr(model)
    )


def test_data_summary_singular_and_plural():
    assert data_summary([0]) == "1 unit: 1 event"
    assert data_summary([1, 1]) == "2 units: 0 events, 2 right censored"


def test_a_restored_model_prints_the_same_data_line(rossi):
    # Principle 20: a model saved without its data still prints the line.
    x, Z, c = rossi
    models = [
        sp.Weibull.fit(x, c),
        sp.KaplanMeier.fit(x, c),
        sp.CoxPH.fit(x, Z, c),
        sp.WeibullPH.fit(x, Z, c),
    ]
    for model in models:
        back = sp.from_dict(model.to_dict())
        assert (
            "432 units: 114 events at 49 unique times, " "318 right censored"
        ) in repr(back)
        assert repr(back).count("Data  ") == 1
