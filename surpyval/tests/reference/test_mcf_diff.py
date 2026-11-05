"""The two-sample MCF comparison against R's ``reda``.

The data are the seeded recurrent-event fixtures ``recurrent_groups`` and
``recurrent_groups_ties`` (``scripts/reference/make_fixtures.py``); the
values are ``reda::mcf``, ``reda::mcfDiff`` and ``reda::mcfDiff.test``
(reda 0.5.6, R 4.6.2), written by
``scripts/reference/reference_r_reda.R``. Everything is a closed form, so
the tolerance is 1e-6 (absolute for the MCFs, their difference and the
standard errors; relative for the test statistics, their variances and
p-values); they agree to about 1e-14.

reda carries the MCF of the group followed for less time at its last
value past its end; SurPyval does not estimate the difference there, so
the difference is compared on reda's times up to the end of the shorter
follow-up.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.recurrent import NonParametricCounting, mcf_diff

from ._data import fixture, reference, values

SOURCE = "r_reda"
CASES = ("recurrent_groups", "recurrent_groups_ties")
TOL = 1e-6


def _models(name):
    d = fixture(name)
    out = []
    for group in (0, 1):
        rows = d["group"] == group
        out.append(
            NonParametricCounting.fit(
                d["time"][rows],
                i=d["id"][rows].astype(int),
                c=1 - d["event"][rows].astype(int),
            )
        )
    return out


@pytest.mark.parametrize("name", CASES)
def test_each_mcf_and_its_robust_se_match_reda(name):
    v = values(SOURCE, name)
    for group, model in enumerate(_models(name)):
        times = v[f"time_{group}"]
        assert_allclose(model.mcf(times), v[f"mcf_{group}"], atol=TOL)
        idx = np.searchsorted(model.x, times)
        assert_allclose(np.sqrt(model.var[idx]), v[f"se_{group}"], atol=TOL)


@pytest.mark.parametrize("name", CASES)
def test_the_difference_and_its_se_match_mcfdiff(name):
    v = values(SOURCE, name)
    result = mcf_diff(*_models(name))
    times = v["time"]
    shared = times <= result.x.max()
    # The same times, up to the end of the shorter follow-up.
    assert_allclose(result.x, times[shared])
    assert_allclose(result.diff(times[shared]), v["diff"][shared], atol=TOL)
    assert_allclose(result.se, v["diff_se"][shared], atol=TOL)
    # Past it the difference is not estimated.
    assert np.isnan(result.diff(times[~shared])).all()
    assert (~shared).any()


@pytest.mark.parametrize("variance", ["robust", "poisson"])
@pytest.mark.parametrize("name", CASES)
def test_the_pseudo_score_tests_match_mcfdiff_test(name, variance):
    expected = reference(SOURCE, name)["values"][variance]
    result = mcf_diff(*_models(name), test_variance=variance)
    for row, weight in enumerate(("constant", "linear")):
        test = result.tests[weight]
        assert_allclose(test["score"], expected["statistic"][row], rtol=TOL)
        assert_allclose(test["score_var"], expected["variance"][row], rtol=TOL)
        assert_allclose(test["statistic"], expected["chisq"][row], rtol=TOL)
        assert_allclose(test["p_value"], expected["p_value"][row], rtol=TOL)
    assert result.statistic == result.tests["constant"]["statistic"]
    assert result.p_value == result.tests["constant"]["p_value"]
