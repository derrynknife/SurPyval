"""The exposure (person-years) table, ``person_years``, against R's
``survival::pyears`` and ``poisson.test``.

The values are in ``data/r_pyears.json``, written by
``scripts/reference/reference_r_pyears.R`` (R 4.6.2, survival 3.8.12), on
``survival::mgus2`` (one row per patient, every time in months, with a
late entry for every fourth id; fixture ``mgus2_pyears``).

What is compared:

- ``pyears(Surv(futime, death) ~ tcut(age) + tcut(year) + sex,
  ratetable = rt)``: the months, patients and deaths in every cell of
  attained age by calendar period by sex, the off-table time, and the
  expected deaths under a rate table whose cut points are the bands'.
  The months are sums of whole numbers, so they agree to rounding
  (1e-12 relative);
- ``pyears(Surv(tl, futime, death) ~ tcut(age) + tcut(-tl) + sex)``:
  late entry, time since entry as a time scale, and ages below the
  first band off the table;
- ``poisson.test``: the exact (Garwood) interval of a rate, of the
  observed to expected ratio, and the exact conditional interval and
  p-value of a rate ratio (the men's death rate over the women's).
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

TOL = 1e-12


@pytest.fixture(scope="module")
def mgus():
    return fixture("mgus2_pyears")


def _cells(table, keys, ref, ref_keys):
    """The table's rows in the order of R's cells."""
    r = pd.DataFrame({k: ref[rk] for k, rk in zip(keys, ref_keys)})
    t = table.copy()
    for k in keys:
        if isinstance(t[k].dtype, pd.CategoricalDtype):
            t[k] = t[k].astype(str)
    out = r.merge(t, on=keys, how="left", validate="one_to_one")
    assert len(out) == len(t)
    return out


def _sex(mgus):
    return np.where(mgus["male"] == 1, "M", "F")


def _first_table(mgus, rates=None):
    ref = values("r_pyears", "pyears_age_period_sex")
    ages = 12 * np.array([20, 60, 70, 80, 110.0])
    years = 12 * np.array([1960, 1980, 1990, 2010.0])
    return ref, sp.person_years(
        mgus["futime"],
        1 - mgus["death"],
        factors={"sex": _sex(mgus)},
        timescales={
            "age": (12 * mgus["age"], ages),
            "year": (12 * mgus["dxyr"], years),
        },
        rates=rates,
    )


def test_age_period_sex_cells_match_pyears(mgus):
    ref, py = _first_table(mgus)
    got = _cells(py.table, ["age", "year", "sex"], ref, ["age", "year", "sex"])
    assert_allclose(got["exposure"], ref["pyears"], rtol=TOL)
    assert_allclose(got["events"], ref["event"], rtol=0)
    assert_allclose(got["units"], ref["n"], rtol=0)
    assert py.offtable["exposure"] == ref["offtable"] == 0
    assert py.dimensions == ["sex", "age", "year"]


def test_expected_deaths_match_pyears_with_a_rate_table(mgus):
    ref = values("r_pyears", "pyears_age_period_sex")
    rates = pd.DataFrame(
        {
            "age": ref["age"],
            "year": ref["year"],
            "sex": ref["sex"],
            "rate": ref["rate"],
        }
    )
    _, py = _first_table(mgus, rates=rates)
    got = _cells(py.table, ["age", "year", "sex"], ref, ["age", "year", "sex"])
    assert_allclose(got["expected"], ref["expected"], rtol=1e-12)
    total = py.margins().iloc[0]
    ratios = values("r_pyears", "ratios")
    assert total["events"] == ratios["observed"]
    assert_allclose(total["expected"], ratios["expected"], rtol=1e-12)
    assert_allclose(total["oe_ratio"], ratios["oe"], rtol=1e-12)
    assert_allclose(total["oe_lower"], ratios["oe_lower"], rtol=1e-10)
    assert_allclose(total["oe_upper"], ratios["oe_upper"], rtol=1e-10)


def test_rate_ratio_matches_poisson_test(mgus):
    _, py = _first_table(mgus)
    ratios = values("r_pyears", "ratios")
    rr = py.rate_ratio({"sex": "M"}, {"sex": "F"})
    assert_allclose(rr["ratio"], ratios["ratio"], rtol=1e-12)
    assert_allclose(rr["lower"], ratios["ratio_lower"], rtol=1e-9)
    assert_allclose(rr["upper"], ratios["ratio_upper"], rtol=1e-9)
    assert_allclose(rr["p_value"], ratios["p_value"], rtol=1e-9)


def test_late_entry_and_time_since_entry_match_pyears(mgus):
    ref = values("r_pyears", "pyears_late_entry")
    with pytest.warns(UserWarning, match="outside the bands"):
        py = sp.person_years(
            mgus["futime"],
            1 - mgus["death"],
            tl=mgus["tl"],
            factors={"sex": _sex(mgus)},
            timescales={
                "age": (12 * mgus["age"], 12 * np.array([50, 65, 80, 120])),
                "since": (-mgus["tl"], [0, 24, 60, 500]),
            },
        )
    got = _cells(
        py.table, ["age", "since", "sex"], ref, ["age", "since", "sex"]
    )
    assert_allclose(got["exposure"], ref["pyears"], rtol=TOL)
    assert_allclose(got["events"], ref["event"], rtol=0)
    assert_allclose(got["units"], ref["n"], rtol=0)
    assert_allclose(py.offtable["exposure"], ref["offtable"], rtol=TOL)


def test_garwood_bounds_match_poisson_test():
    ref = values("r_pyears", "garwood")
    lower, upper = sp.poisson_rate_bounds(ref["events"], ref["exposure"])
    assert_allclose(lower, ref["lower"], rtol=1e-10)
    assert_allclose(upper, ref["upper"], rtol=1e-10)
    lo90, hi90 = sp.poisson_rate_bounds(7, 3120, alpha_ci=0.1)
    assert_allclose([lo90, hi90], [ref["lower_90"], ref["upper_90"]], 1e-10)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.poisson_rate_bounds(0, 1.0)
