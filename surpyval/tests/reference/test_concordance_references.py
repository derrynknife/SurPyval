"""Harrell's and Uno's concordance indices, their jackknife variances and
the covariance of two scores' indices against stored R results
(``scripts/reference/reference_r_concordance.R``).

``survival::concordance(..., reverse = TRUE, timewt = "n" or "n/G2",
ymax = tau)`` on R's ``lung`` and ``veteran`` with two fixed linear
predictors each (one with many tied scores; both data sets have tied
times), and ``survC1::Est.Cval`` for Uno's C on two fixtures without tied
times. The indices are ratios of sums of the same pairs, so they agree
with ``survival`` to 1e-10 and the variances to a relative 1e-8;
``survC1`` rounds its inputs and its sums are not exact to double
precision, so it agrees with both to about 3e-8 (tested at 1e-7).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.metrics import compare_concordance, concordance_estimate

from ._data import fixture, reference, values

SOURCE = "r_concordance"
C_TOL = dict(rtol=0, atol=1e-10)
VAR_TOL = dict(rtol=1e-8, atol=0)

SURVIVAL_IDS = [
    f"{data}_{w}_{tau}"
    for data in ("lung", "veteran")
    for w in ("harrell", "uno")
    for tau in (("all", "tau500") if data == "lung" else ("all", "tau100"))
]


def _outcomes(name):
    if name == "lung":
        d = fixture("lung")
        keep = ~np.isnan(d["ph_ecog"])
        return d["time"][keep], d["c"][keep]
    d = fixture("veteran")
    return d["time"], 1 - d["status"]


@pytest.mark.parametrize("ref_id", SURVIVAL_IDS)
def test_concordance_and_its_variance_match_survival(ref_id):
    ref = reference(SOURCE, ref_id)
    v = values(SOURCE, ref_id)
    x, c = _outcomes(ref["fixture"])
    weighting = "harrell" if ref["settings"]["timewt"] == "n" else "uno"
    tau = ref["settings"]["ymax"]
    for k, lp in enumerate((v["lp_a"], v["lp_b"])):
        res = concordance_estimate(x, c, lp, weighting=weighting, tau=tau)
        assert_allclose(res.estimate, v["concordance"][k], **C_TOL)
        assert_allclose(res.se**2, v["var_single"][k], **VAR_TOL)
        assert_allclose(res.se**2, v["var"][k, k], **VAR_TOL)
        concordant, discordant, tied = v["count"][k]
        assert_allclose(res.concordant, concordant + tied / 2, rtol=1e-10)
        assert_allclose(res.pairs, concordant + discordant + tied, rtol=1e-10)


@pytest.mark.parametrize("ref_id", SURVIVAL_IDS)
def test_two_scores_covariance_matches_survival(ref_id):
    ref = reference(SOURCE, ref_id)
    v = values(SOURCE, ref_id)
    x, c = _outcomes(ref["fixture"])
    weighting = "harrell" if ref["settings"]["timewt"] == "n" else "uno"
    tau = ref["settings"]["ymax"]
    res = compare_concordance(
        x, c, v["lp_a"], v["lp_b"], weighting=weighting, tau=tau
    )
    assert_allclose(res.estimates, v["concordance"], **C_TOL)
    assert_allclose(res.cov, v["var"], **VAR_TOL)
    # R's contrast c(1, -1) of the two indices
    contrast = np.array([1.0, -1.0])
    var = contrast @ v["var"] @ contrast
    assert_allclose(res.se, np.sqrt(var), rtol=1e-8)
    assert_allclose(
        res.difference, v["concordance"][0] - v["concordance"][1], **C_TOL
    )


@pytest.mark.parametrize(
    "ref_id",
    [
        "prediction_continuous_uno_survc1_tau4.5",
        "prediction_continuous_uno_survc1_tau10.5",
        "additive_uno_survc1_tau5.5",
        "additive_uno_survc1_tau15.5",
    ],
)
def test_uno_matches_survc1_and_survival(ref_id):
    ref = reference(SOURCE, ref_id)
    v = values(SOURCE, ref_id)
    d = fixture(ref["fixture"])
    score = d[ref["settings"]["score"]]
    res = concordance_estimate(
        d["x"], d["c"], score, weighting="uno", tau=ref["settings"]["tau"]
    )
    assert_allclose(res.estimate, v["Dhat"], rtol=0, atol=1e-7)
    assert_allclose(res.estimate, v["survival_concordance"], **C_TOL)
    assert_allclose(res.se**2, v["survival_var"], **VAR_TOL)
