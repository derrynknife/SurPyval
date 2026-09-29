"""The strategies generate what they promise (#379).

Every generated univariate data set is accepted by ``xcnt_handler`` (so a
failure elsewhere is the fitter's, not the generator's), and the
generator does reach the corners the properties are meant to search:
ties between an event and a censoring, a single row, all rows censored,
one distinct time, every censoring kind and both truncations.
"""

import numpy as np
from hypothesis import find, given, settings

import surpyval as sp
from surpyval.tests.properties import strategies as gen


@given(data=gen.xcnt())
def test_xcnt_is_valid(data):
    sp.xcnt_handler(**data)


@given(data=gen.xicn())
def test_xicn_is_valid(data):
    sp.handle_xicn(**data)


@given(case=gen.invalid_xcnt())
def test_invalid_xcnt_is_refused(case):
    _, data = case
    try:
        sp.xcnt_handler(**data)
    except ValueError:
        return
    # The one defect the handler lets through: the fitters check it.
    x, c = np.asarray(data["x"]), np.asarray(data["c"])
    assert np.any(np.isinf(x) & (c == gen.EXACT)), case


_FIND = settings(max_examples=2000, database=None, derandomize=True)


def _reaches(strategy, condition):
    # ``find`` raises NoSuchExample if nothing generated meets condition.
    find(strategy, condition, settings=_FIND)


def _tie_event_censored(d):
    x, c = np.asarray(d["x"]), np.asarray(d["c"])
    return x.ndim == 1 and bool(np.intersect1d(x[c == 0], x[c == 1]).size)


def test_corners_are_reached():
    xcnt = gen.xcnt()
    _reaches(xcnt, _tie_event_censored)
    _reaches(xcnt, lambda d: len(d["c"]) == 1)
    _reaches(xcnt, lambda d: len(d["c"]) > 2 and np.all(d["c"] == 1))
    _reaches(
        xcnt,
        lambda d: len(d["c"]) > 2 and np.unique(np.ravel(d["x"])).size == 1,
    )
    for flag in gen.ALL_CENSORING:
        _reaches(xcnt, lambda d, f=flag: bool(np.any(d["c"] == f)))
    _reaches(xcnt, lambda d: "tl" in d)
    _reaches(xcnt, lambda d: "tr" in d)
    _reaches(gen.competing_risks(), lambda d: len(set(d["e"])) == 3)
