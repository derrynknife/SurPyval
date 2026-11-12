"""State in the lifetime data model: ``SurpyvalData(..., s=..., i=...)``.

``s`` is the state a row's unit was in during the row, ``e`` the state it
moved into at ``x`` and ``c`` whether it moved (0) or was still in ``s``
(1); ``i`` is the unit (``ids`` is its other name). Without ``s`` every
row starts from the one initial state and nothing changes: the first
tests pin the reprs, dictionaries, warnings and fits of data without
``s`` to the values they had before ``s`` existed.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval import SurpyvalData

# --- Data without s: what it gave before states existed -----------------


def _plain():
    return SurpyvalData(
        x=[2.0, 3.0, 3.0, 5.0, 7.0, 8.0, 11.0, 13.0, [4.0, 6.0]],
        c=[0, 0, 0, 1, 0, 1, 0, 1, 2],
        n=[1, 2, 2, 1, 3, 1, 1, 2, 1],
        tl=[0.5, 0.5, 0.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )


_X10 = [2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 11.0, 13.0, 14.0, 16.0]
_C10 = [0, 0, 1, 0, 0, 1, 0, 0, 1, 0]
_Z10 = [
    [0.1, 1],
    [0.4, 0],
    [0.2, 1],
    [0.9, 0],
    [0.5, 1],
    [0.3, 0],
    [0.8, 1],
    [0.6, 0],
    [0.7, 1],
    [0.05, 0],
]


def _covariates(**extra):
    return SurpyvalData(
        _X10,
        _C10,
        Z=_Z10,
        covariate_names=["load", "arm"],
        ids=list("ABCDEFGHIJ"),
        **extra,
    )


_X12 = _X10 + [17.0, 19.0]
_C12 = _C10 + [0, 1]


def _typed_with_masked_and_suspension():
    e = ["wear", "crack", "wear", "wear", None, None, "crack", "wear"]
    e += [None, "crack", "wear", None]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        data = SurpyvalData(_X12, _C12, e=e)
    return data, [str(w.message) for w in caught]


_E12 = ["wear", "crack", None, "wear", "crack", None, "crack", "wear"]
_E12 += [None, "crack", "wear", None]


def _typed(**extra):
    return SurpyvalData(_X12, _C12, e=_E12, **extra)


def test_without_s_repr_is_unchanged():
    assert repr(_plain()) == (
        "SurpyvalData: 14 items, 8 rows\n"
        "             x  c  n   tl\n"
        "    [2.0, 2.0]  0  1  0.5\n"
        "    [3.0, 3.0]  0  4  0.5\n"
        "    [4.0, 6.0]  2  1  0.0\n"
        "    [5.0, 5.0]  1  1  1.0\n"
        "    [7.0, 7.0]  0  3  1.0\n"
        "    [8.0, 8.0]  1  1  0.0\n"
        "  [11.0, 11.0]  0  1  0.0\n"
        "  [13.0, 13.0]  1  2  0.0"
    )
    data, _ = _typed_with_masked_and_suspension()
    assert repr(data) == (
        "SurpyvalData: 12 items, 12 rows\n"
        "     x  c  n        e\n"
        "   2.0  0  1   'wear'\n"
        "   3.0  0  1  'crack'\n"
        "   4.0  1  1     None\n"
        "   5.0  0  1   'wear'\n"
        "   7.0  0  1     None\n"
        "   8.0  1  1     None\n"
        "  11.0  0  1  'crack'\n"
        "  13.0  0  1   'wear'\n"
        "  14.0  1  1     None\n"
        "  16.0  0  1  'crack'\n"
        "  ... 2 more rows; summary() describes it all"
    )
    assert data._repr_html_().startswith(
        "<div><p><b>SurpyvalData: 12 items, 12 rows</b></p><table><thead>"
        "<tr><th>x</th><th>c</th><th>n</th><th>e</th></tr>"
    )
    assert repr(_covariates()).splitlines()[:3] == [
        "SurpyvalData: 10 items, 10 rows",
        "   id     x  c  n  load  arm",
        "  'A'   2.0  0  1   0.1  1.0",
    ]


def test_without_s_summary_and_warnings_are_unchanged():
    assert _plain().summary() == (
        "SurpyvalData: 14 items, 8 rows\n"
        "  outcomes: 9 observed, 4 right censored, 1 interval censored\n"
        "  truncation: 14 left"
    )
    data, caught = _typed_with_masked_and_suspension()
    assert data.summary() == (
        "SurpyvalData: 12 items, 12 rows\n"
        "  outcomes: 8 observed, 4 right censored\n"
        "  event types: crack (3), wear (4), not recorded (1)"
    )
    assert caught == [
        "1 suspension (right censored: rows 2) had an event type, which "
        "was dropped: a suspension has not failed, so it has none.",
        "1 failure has no recorded event type (masked causes): rows 4. "
        "Fits of every failure count them; a fit of one type treats them "
        "as for_event_type's masked= says (not of that type, by default); "
        "competing-risks fits refuse them. relabel_event_types({None: "
        "'unknown'}) makes them a type of their own.",
    ]
    assert data.s is None


def test_without_s_to_dict_is_unchanged():
    assert _plain().to_dict() == {
        "class": "SurpyvalData",
        "x": [
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 6.0],
            [5.0, 5.0],
            [7.0, 7.0],
            [8.0, 8.0],
            [11.0, 11.0],
            [13.0, 13.0],
        ],
        "c": [0, 0, 2, 1, 0, 1, 0, 1],
        "n": [1, 4, 1, 1, 3, 1, 1, 2],
        "t": [[b, None] for b in (0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0)],
        "non_finite": {"inf": [f"/t/{k}/1" for k in range(8)]},
    }
    out = _covariates().to_dict()
    assert list(out) == [
        "class",
        "x",
        "c",
        "n",
        "t",
        "Z",
        "covariate_names",
        "ids",
        "non_finite",
    ]
    assert "schema" not in out and "s" not in out
    data, _ = _typed_with_masked_and_suspension()
    assert data.to_dict()["e"] == [
        "wear",
        "crack",
        None,
        "wear",
        None,
        None,
        "crack",
        "wear",
        None,
        "crack",
        "wear",
        None,
    ]


def test_without_s_arrays_are_unchanged():
    data = _plain()
    assert data.x.tolist() == [
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 6.0],
        [5.0, 5.0],
        [7.0, 7.0],
        [8.0, 8.0],
        [11.0, 11.0],
        [13.0, 13.0],
    ]
    assert data.c.tolist() == [0, 0, 2, 1, 0, 1, 0, 1]
    assert data.n.tolist() == [1, 4, 1, 1, 3, 1, 1, 2]
    assert data.tl.tolist() == [0.5, 0.5, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    assert np.isinf(data.tr).all()


# The fits of the data above before states existed (arm64, numpy and
# scipy of the time); the tolerance allows only for another platform's
# last bits.
_FITS = {
    "weibull": [8.50097477427776, 1.4381608270005815],
    "km_R": [
        0.8333333333333334,
        0.6666666666666667,
        0.44444444444444453,
        0.44444444444444453,
        0.0,
    ],
    "cox": [0.7202225768659077, 0.03818161838516938],
    "cr_cif": {
        "wear": [0.08333333333333333, 0.1759259259259259, 0.2839506172839506],
        "crack": [
            0.08333333333333333,
            0.1759259259259259,
            0.28395061728395055,
        ],
    },
    "pcr": {
        "wear": [0.04601147505958924, 0.176530016807665, 0.33999614259961486],
        "crack": [
            0.031936221444673826,
            0.15332297065053974,
            0.3340986463620359,
        ],
    },
}
_AT = np.array([3.0, 8.0, 15.0])


def _km_data(**extra):
    return SurpyvalData(
        [2, 3, 3, 5, 7, 8], [0, 0, 1, 0, 1, 0], tl=[0, 0, 1, 1, 0, 0], **extra
    )


def test_without_s_fits_are_unchanged():
    close = dict(rtol=1e-10, atol=0)
    np.testing.assert_allclose(
        sp.Weibull.fit(_plain()).params, _FITS["weibull"], **close
    )
    np.testing.assert_allclose(
        sp.KaplanMeier.fit(_km_data()).R, _FITS["km_R"], **close
    )
    np.testing.assert_allclose(
        sp.CoxPH.fit(_covariates()).params, _FITS["cox"], **close
    )
    cr = sp.CompetingRisks.fit(_typed())
    pcr = sp.ParametricCompetingRisks.fit(_typed())
    for event in ("wear", "crack"):
        np.testing.assert_allclose(
            cr.cif(_AT, event), _FITS["cr_cif"][event], **close
        )
        np.testing.assert_allclose(
            pcr.cif(_AT, event), _FITS["pcr"][event], **close
        )


def test_one_state_fits_as_no_state():
    """Data whose rows are all in one state is the data without states:
    every fit gives the same numbers, bit for bit."""
    one = ["start"] * 10
    np.testing.assert_array_equal(
        sp.CoxPH.fit(_covariates(s=one)).params,
        sp.CoxPH.fit(_covariates()).params,
    )
    np.testing.assert_array_equal(
        sp.KaplanMeier.fit(_km_data(s=[0] * 6)).R,
        sp.KaplanMeier.fit(_km_data()).R,
    )
    data = SurpyvalData(
        [1.0, 2.0, 4.0, 5.0, 9.0], [0, 0, 1, 0, 1], s=["a"] * 5
    )
    plain = SurpyvalData([1.0, 2.0, 4.0, 5.0, 9.0], [0, 0, 1, 0, 1])
    np.testing.assert_array_equal(
        sp.Weibull.fit(data).params, sp.Weibull.fit(plain).params
    )
    states = ["start"] * 12
    for model in (sp.CompetingRisks, sp.ParametricCompetingRisks):
        with_s = model.fit(_typed(s=states))
        without = model.fit(_typed())
        for event in ("wear", "crack"):
            np.testing.assert_array_equal(
                with_s.cif(_AT, event), without.cif(_AT, event)
            )
    assert data.to_xrd()[1].tolist() == plain.to_xrd()[1].tolist()


# --- Data with states ---------------------------------------------------


def _illness_death(**extra):
    """Three patients: A well then ill then dead; B well, still well at 7;
    C well then dead."""
    kwargs = dict(
        x=[3, 5, 7, 4],
        tl=[0, 3, 0, 0],
        s=["well", "ill", "well", "well"],
        e=["ill", "dead", None, "dead"],
        i=["A", "A", "B", "C"],
    )
    kwargs.update(extra)
    return SurpyvalData(**kwargs)


def test_states_are_kept_as_given_and_c_is_read_from_e():
    data = _illness_death()
    assert data.c.tolist() == [0, 0, 1, 0]
    assert data.s.dtype == object and data.s.ndim == 1
    assert data.s.tolist() == ["well", "ill", "well", "well"]
    assert data.e.tolist() == ["ill", "dead", None, "dead"]
    assert data.x.tolist() == [3.0, 5.0, 7.0, 4.0]  # not grouped or sorted
    numbers = SurpyvalData(
        x=[3, 5, 7], s=np.array([1, 2, 1]), e=[2, 3, None], i=[1, 1, 2]
    )
    assert [type(v) for v in numbers.s] == [int, int, int]
    assert [type(v) for v in numbers.e[:2]] == [int, int]
    assert numbers.c.tolist() == [0, 0, 1]
    # A blank is None, NaN or ""
    blanks = SurpyvalData(x=[1, 2, 3], s=["a"] * 3, e=["b", np.nan, ""])
    assert blanks.c.tolist() == [0, 1, 1]


def test_n_items_counts_units():
    data = _illness_death()
    assert len(data) == 4
    assert data.n_items == 3
    assert repr(data).splitlines()[0] == "SurpyvalData: 3 items, 4 rows"
    # Without ids each row is its own unit
    assert _illness_death(i=None).n_items == 4


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            dict(c=[1, 0, 1, 0]),
            r"Row 0 has c = 1 \(still in the state s at x\) and a state "
            r"entered in e",
        ),
        (
            dict(c=[0, 0, 0, 0]),
            r"Row 2 has no state entered in e, but c says the unit moved",
        ),
        (
            dict(e=["ill", None, None, "dead"], c=[0, 0, 0, 0]),
            r"Rows 1, 2 have no state entered in e",
        ),
        (
            dict(s=["well", None, "well", "well"]),
            r"'s' must give the state of every row .*; row 1 has none",
        ),
        (dict(s=["well", "ill"]), r"'s' must hold one state per row \(4\)"),
        (
            dict(s="well"),
            r"'s' must hold one state per row, not a single value",
        ),
        (dict(e="ill"), r"'e' must hold one state per row, not a single"),
        (dict(e=["ill"]), r"'e' must hold one state per row \(4\); got 1"),
        (dict(e=None), r"Data with states s needs e"),
    ],
)
def test_refusals_name_the_rows_and_the_fix(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _illness_death(**kwargs)


def test_i_and_ids_are_one_argument_and_one_list():
    by_i = SurpyvalData([1, 2, 3], i=["a", "b", "c"])
    by_ids = SurpyvalData([1, 2, 3], ids=["a", "b", "c"])
    assert by_i.i == by_i.ids == by_ids.i == by_ids.ids == ["a", "b", "c"]
    assert by_i.i is by_i.ids
    assert by_i.to_dict() == by_ids.to_dict()
    with pytest.raises(TypeError, match="same argument"):
        SurpyvalData([1, 2, 3], i=["a", "b", "c"], ids=["a", "b", "c"])
    # Setting either sets both
    data = SurpyvalData([1, 2, 3], c=[0, 1, 0], group_and_sort=False)
    data.i = ["x", "y", "z"]
    assert data.ids == ["x", "y", "z"]
    data.ids = ["p", "q", "r"]
    assert data.i == ["p", "q", "r"]
    assert {"i", "ids", "s"} <= set(dir(data))
    with pytest.raises(ValueError, match="'i' must hold one id per entry"):
        SurpyvalData([1, 2, 3], i=["a", "b"])


def test_repeated_ids_only_with_states():
    with pytest.raises(ValueError, match="given more than once: 'A'"):
        SurpyvalData([3, 5, 7], i=["A", "A", "B"])
    assert _illness_death().i == ["A", "A", "B", "C"]
    with pytest.raises(ValueError, match="all be of one kind"):
        _illness_death(i=["A", "A", 1, "C"])


def test_subsets_and_reshapes_carry_states_and_units():
    data = _illness_death()
    part = data[1:]
    assert part.s.tolist() == ["ill", "well", "well"]
    assert part.i == ["A", "B", "C"]
    well = data[data.s == "well"]
    assert well.i == ["A", "B", "C"] and set(well.s) == {"well"}
    unit = data.select(ids="A")
    assert unit.i == ["A", "A"] and unit.s.tolist() == ["well", "ill"]
    assert data.drop("A").s.tolist() == ["well", "well"]
    assert data.select(where=lambda r: r.value > 3.5).s.tolist() == [
        "ill",
        "well",
        "well",
    ]
    cut = data.censor_at(4)
    assert cut.s.tolist() == data.s.tolist() and cut.i == data.i
    assert cut.c.tolist() == [0, 1, 1, 0]
    assert cut.e.tolist() == ["ill", None, None, "dead"]
    by_unit = data.censor_at({"A": 4})
    assert by_unit.i == data.i and by_unit.c.tolist() == [0, 1, 1, 0]
    entered = data.enter_at(lambda r: 0.5)
    assert entered.tl.tolist() == [0.5, 3.0, 0.5, 0.5]
    assert entered.s.tolist() == data.s.tolist()
    extended = data.extend_running(9)
    assert extended.x.tolist() == [3.0, 5.0, 9.0, 4.0]
    assert extended.s.tolist() == data.s.tolist() and extended.i == data.i
    assert data.without("event_types").s.tolist() == data.s.tolist()
    assert data.without("states").s is None
    assert data.without("states").e.tolist() == data.e.tolist()
    dead = data.for_event_type("dead")
    assert dead.s.tolist() == data.s.tolist() and dead.e is None
    assert dead.c.tolist() == [1, 0, 1, 0]
    renamed = data.relabel_event_types({"well": "healthy", "dead": "died"})
    assert renamed.s.tolist() == ["healthy", "ill", "healthy", "healthy"]
    assert renamed.e.tolist() == ["ill", "died", None, "died"]
    with pytest.raises(ValueError, match="a unit has a row for each stay"):
        data.update([sp.ItemLifetime("D", observed=2)])


def test_records_are_refused_with_states():
    data = _illness_death()
    for call in (data.to_items, lambda: list(data.records())):
        with pytest.raises(ValueError, match="no place for the state s"):
            call()
    with pytest.raises(ValueError, match=r"item\('A'\) gives ItemLifetime"):
        data.item("A")
    with pytest.raises(ValueError, match="from_observations has no place"):
        data.to_observations()


def test_to_xrd_refuses_several_states_and_passes_one():
    with pytest.raises(ValueError, match=r"rows are in 2 states .*multi"):
        _illness_death().to_xrd()
    one = SurpyvalData([1, 2, 4, 5], [0, 0, 1, 0], s=["a"] * 4)
    plain = SurpyvalData([1, 2, 4, 5], [0, 0, 1, 0])
    for got, want in zip(one.to_xrd(), plain.to_xrd()):
        np.testing.assert_array_equal(got, want)


def test_dataframe_round_trip():
    data = _illness_death()
    df = data.to_dataframe()
    assert list(df.columns) == ["id", "x", "c", "n", "tl", "s", "e"]
    back = SurpyvalData.from_dataframe(
        df,
        x_col="x",
        c_col="c",
        n_col="n",
        tl_col="tl",
        i_col="id",
        s_col="s",
        e_col="e",
    )
    assert back.to_dict() == data.to_dict()
    # Whole-number states stay whole numbers, though pandas stores the
    # blank among them as a float
    frame = pd.DataFrame(
        {"u": [1, 1, 2], "x": [1, 2, 3], "s": [1, 2, 1], "e": [2, 3, None]}
    )
    read = SurpyvalData.from_dataframe(
        frame, x_col="x", s_col="s", e_col="e", i_col="u"
    )
    assert read.s.tolist() == [1, 2, 1] and read.e.tolist() == [2, 3, None]
    assert [type(v) for v in read.e[:2]] == [int, int]
    assert read.c.tolist() == [0, 0, 1] and read.i == [1, 1, 2]


def test_dict_and_json_round_trip_keep_the_states_types():
    data = SurpyvalData(
        x=[1, 2, 3], s=["1", 1, "1"], e=[None, "1", 2], i=[7, 7, 8]
    )
    out = data.to_dict()
    assert out["s"] == ["1", 1, "1"] and out["ids"] == [7, 7, 8]
    # A reader without states (schema 3 at most) refuses it
    assert out["schema"] == 4
    again = SurpyvalData.from_json(data.to_json())
    assert again.s.tolist() == ["1", 1, "1"]
    assert [type(v) for v in again.s] == [str, int, str]
    assert again.e.tolist() == [None, "1", 2]
    assert again.i == [7, 7, 8] and again.c.tolist() == [1, 0, 0]
    assert sp.from_dict(json.loads(json.dumps(out))).s.tolist() == [
        "1",
        1,
        "1",
    ]


def test_old_dict_loads_as_before():
    old = {
        "class": "SurpyvalData",
        "x": [1.0, 2.0],
        "c": [0, 1],
        "n": [1, 1],
        "t": [[None, None], [None, None]],
        "ids": ["a", "b"],
        "non_finite": {"-inf": ["/t/0/0", "/t/1/0"], "inf": ["/t/0/1"]},
    }
    old["non_finite"]["inf"].append("/t/1/1")
    data = SurpyvalData.from_dict(old)
    assert data.s is None and data.i == ["a", "b"]
    assert data.to_dict() == old


def test_repr_and_summary_with_states():
    data = _illness_death()
    assert repr(data) == (
        "SurpyvalData: 3 items, 4 rows\n"
        "   id    x  c  n   tl       s       e\n"
        "  'A'  3.0  0  1  0.0  'well'   'ill'\n"
        "  'A'  5.0  0  1  3.0   'ill'  'dead'\n"
        "  'B'  7.0  1  1  0.0  'well'    None\n"
        "  'C'  4.0  0  1  0.0  'well'  'dead'"
    )
    assert "<th>s</th><th>e</th>" in data._repr_html_()
    assert data.summary() == (
        "SurpyvalData: 3 items, 4 rows (with item ids)\n"
        "  outcomes: 3 observed, 1 right censored\n"
        "  truncation: 4 left\n"
        "  states: well, ill, dead\n"
        "  moves: well to ill (1), ill to dead (1), well to dead (1)"
    )


def test_fits_that_are_not_multi_state_refuse_several_states():
    data = _illness_death()
    for fit in (
        sp.Weibull.fit,
        sp.KaplanMeier.fit,
        sp.CompetingRisks.fit,
        sp.Weibull.fit_from_surpyval_data,
    ):
        with pytest.raises(ValueError, match=r"more than one state \(s\)"):
            fit(data)
    with pytest.raises(ValueError, match="data.s == state"):
        sp.Weibull.fit(data.without("event_types"))
    with pytest.raises(ValueError, match="life_table takes data with one"):
        sp.life_table(data, intervals=[0, 5, 10])
    # The stays in one state fit as competing risks
    well = data[data.s == "well"]
    model = sp.CompetingRisks.fit(well)
    assert sorted(model.event_idx_map) == ["dead", "ill"]


def test_a_fit_with_a_state_argument_is_given_the_states():
    """A fit whose signature has ``state`` (or ``s``) and ``i`` takes them
    from the data, and its censoring from ``e``."""
    from surpyval.utils.surpyval_data import accepts_surpyval_data

    seen = {}

    class Fitter:
        @accepts_surpyval_data
        def fit(self, x, e, state, tl=None, i=None, n=None):
            seen.update(x=x, e=e, state=state, tl=tl, i=i, n=n)
            return None

    data = _illness_death()
    Fitter().fit(data)
    assert seen["state"].tolist() == data.s.tolist()
    assert seen["i"] == data.i and seen["tl"].tolist() == [0, 3, 0, 0]
    assert seen["e"].tolist() == data.e.tolist()
    with pytest.raises(TypeError, match="leave out 'state'"):
        Fitter().fit(data, state=data.s)
    with pytest.raises(TypeError, match="This fit needs states, state"):
        Fitter().fit(SurpyvalData([1, 2], e=["a", "b"]))
