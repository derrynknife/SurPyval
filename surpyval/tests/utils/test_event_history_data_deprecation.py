"""Building an EventHistoryData yourself is deprecated in v0.30 and
removed in v0.31: a SurpyvalData with each row's unit ``i`` is the one
container for several rows per unit. Each way of building one warns once,
at the caller's line, naming the SurpyvalData call that does the same.
The recurrent fits and simulations, which still build one inside, warn
nobody."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import EventHistoryData, ItemHistory, SurpyvalData
from surpyval.recurrent import GeneralizedRenewal
from surpyval.utils.deprecation import REMOVED_IN
from surpyval.utils.notation import Items

X = [2.0, 5.0, 8.0, 3.0, 9.0]
I = [1, 1, 1, 2, 2]  # noqa: E741 - the package's name for the unit ids
C = [0, 0, 1, 0, 1]


def _built():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return EventHistoryData.from_histories(
            observed={1: [2, 5], 2: [3]}, right_censored={1: 8, 2: 9}
        )


def _renewal_simulation():
    truth = GeneralizedRenewal.from_params([10.0, 2.0], 0.5, kijima="i")
    return truth.time_terminated_simulation_data(T=100, items=3)


ENTRIES = {
    "Building an EventHistoryData": (
        lambda: EventHistoryData(X, I, C, 1),
        "SurpyvalData(x, c, n, i=i",
    ),
    "EventHistoryData.from_histories": (
        lambda: EventHistoryData.from_histories(
            observed={1: [2, 5], 2: [3]}, right_censored={1: 8, 2: 9}
        ),
        "SurpyvalData.from_histories",
    ),
    "EventHistoryData.from_items": (
        lambda: EventHistoryData.from_items(
            [ItemHistory("A", observed=[5, 9], right_censored=12)]
        ),
        "SurpyvalData.from_items",
    ),
    "EventHistoryData.from_dict": (
        lambda: EventHistoryData.from_dict(_built().to_dict()),
        "surpyval.from_dict",
    ),
    "EventHistoryData.from_json": (
        lambda: EventHistoryData.from_json(_built().to_json()),
        "surpyval.from_json",
    ),
    "surpyval.handle_xicn": (
        lambda: sp.handle_xicn(X, i=I, c=C),
        "SurpyvalData(x, c, n, i=i",
    ),
    "Items.event_history_data()": (
        lambda: Items.parse("A: 5, 9, 12+").event_history_data(),
        "Items.surpyval_data()",
    ),
}


@pytest.mark.parametrize("what", list(ENTRIES))
def test_each_way_of_building_one_warns_once_naming_the_replacement(what):
    build, instead = ENTRIES[what]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build()
    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(future) == 1, [str(w.message) for w in future]
    message = str(future[0].message)
    assert message.startswith(f"{what} is deprecated")
    assert f"removed in v{REMOVED_IN}" in message
    assert instead in message
    # Attributed to the caller's line, here, not to the package's code.
    assert future[0].filename == __file__


def test_from_dataframe_warns_naming_the_replacement():
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame({"unit": I, "age": X, "c": C})
    with pytest.warns(FutureWarning) as caught:
        EventHistoryData.from_dataframe(
            df, i_col="unit", x_col="age", c_col="c"
        )
    future = [w for w in caught if issubclass(w.category, FutureWarning)]
    assert len(future) == 1
    assert "SurpyvalData.from_dataframe" in str(future[0].message)


@pytest.mark.parametrize(
    "run",
    [
        lambda: sp.CrowAMSAA.fit(X, i=I, c=C),
        lambda: sp.HPP.fit(X, i=I, c=C),
        lambda: sp.NonParametricCounting.fit(X, i=I, c=C),
        lambda: sp.GeneralizedRenewal.fit(X, i=I, c=C),
        lambda: sp.CrowAMSAA.fit(SurpyvalData(X, C, i=I)),
        lambda: sp.CrowAMSAA.fit(
            [3.0, 7.0, 4.0],
            i=[1, 1, 2],
            windows={1: [(0, 5), (6, 10)], 2: [(0, 8)]},
        ),
        lambda: _renewal_simulation(),
        lambda: sp.CrowAMSAA.from_params(
            [2.0, 1.5]
        ).count_terminated_simulation_data(events=5, items=2),
    ],
)
def test_the_fits_and_simulations_warn_nobody(run):
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        run()


def test_the_surpyval_data_route_gives_the_same_fit():
    # The deprecated route and the one container give the same numbers,
    # to the last digit.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        old = sp.CrowAMSAA.fit(sp.handle_xicn(X, i=I, c=C))
    new = sp.CrowAMSAA.fit(SurpyvalData(X, C, i=I))
    assert np.array_equal(old.params, new.params)


# -- attributes that change meaning when model.data becomes a SurpyvalData ---


def _fitted():
    from surpyval.recurrent import CrowAMSAA

    return CrowAMSAA.fit(
        [1, 3, 5, 2, 4, 6],
        i=[1, 1, 1, 2, 2, 2],
        c=[0, 0, 1, 0, 0, 1],
        tl=[0.5, 0.5, 0.5, 1, 1, 1],
    )


@pytest.mark.parametrize(
    "name, says", [("ids", "data.item_ids"), ("tl", "data.to_items()")]
)
def test_reading_an_attribute_that_changes_meaning_warns(name, says):
    model = _fitted()
    with pytest.warns(FutureWarning, match=f"EventHistoryData.{name}") as rec:
        value = getattr(model.data, name)
    assert says in str(rec[0].message)
    assert f"v{REMOVED_IN}" in str(rec[0].message)
    assert rec[0].filename == __file__  # at the reader's line
    expected = [1, 2] if name == "ids" else [0.5, 0.5, 0.5, 1, 1, 1]
    np.testing.assert_array_equal(value, expected)


def test_item_ids_means_the_same_on_both_containers():
    model = _fitted()
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        held = model.data.item_ids
        rows = model.data.to_surpyval_data()
        assert held == [1, 2] == rows.item_ids
        assert rows.ids == [1, 1, 1, 2, 2, 2]  # one per row there
        assert model.data.n_items == 2
    assert SurpyvalData([1, 2], [0, 1], [1, 1]).item_ids is None
    data = SurpyvalData([4, 2, 9], [0, 0, 1], [1, 1, 1], i=["b", "a", "b"])
    assert data.item_ids == ["b", "a"]


def test_the_package_reads_them_silently():
    # A fit, its bounds and a simulation read ids and tl inside.
    from surpyval.recurrent import CrowAMSAA

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        model = _fitted()
        model.cif_cb([2.0, 4.0])
        model.residuals("martingale")
        CrowAMSAA.fit(model.data.to_surpyval_data())
        _renewal_simulation()
