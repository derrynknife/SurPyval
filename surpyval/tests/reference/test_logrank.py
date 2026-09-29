"""SurPyval's log-rank tests against stored R ``survdiff`` results (#379).

The statistic is a closed-form quadratic form in observed-minus-expected
counts in both programs, so it must agree to rounding (``rtol=1e-9``).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

from ._data import fixture, values

EXACT = dict(rtol=1e-9, atol=1e-12)


def _cases():
    aml = fixture("aml")
    lung = fixture("lung")
    ties = fixture("ties")
    # survdiff drops the one patient with ph.ecog missing (na.omit).
    cc = ~np.isnan(lung["ph_ecog"])
    return {
        "logrank_aml": (
            aml["time"],
            aml["maintained"],
            1 - aml["status"],
            {},
        ),
        "logrank_lung_sex": (lung["time"], lung["sex"], lung["c"], {}),
        # survdiff's rho is the G-rho family, S(t-)^rho with the pooled
        # Kaplan-Meier: SurPyval's Fleming-Harrington weighting, gamma 0.
        "logrank_lung_sex_rho1": (
            lung["time"],
            lung["sex"],
            lung["c"],
            {"weighting": "fleming-harrington", "rho": 1},
        ),
        "logrank_lung_ecog": (
            lung["time"][cc],
            lung["ph_ecog"][cc],
            lung["c"][cc],
            {},
        ),
        "logrank_lung_sex_strata": (
            lung["time"][cc],
            lung["sex"][cc],
            lung["c"][cc],
            {"strata": np.minimum(lung["ph_ecog"][cc], 2)},
        ),
        "logrank_ties": (ties["x"], ties["z1"], ties["c"], {}),
    }


@pytest.mark.parametrize("ref_id", sorted(_cases()))
def test_logrank_matches_survdiff(ref_id):
    x, Z, c, kwargs = _cases()[ref_id]
    ref = values("r_survival", ref_id)
    res = sp.logrank(x, Z, c=c, **kwargs)
    assert_allclose(res.statistic, ref["chisq"], **EXACT)
    assert res.dof == ref["df"]
    assert_allclose(res.p_value, ref["p"], **EXACT)
