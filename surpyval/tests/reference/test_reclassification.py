"""The IDI and NRI of a new risk model over an old one against stored R
results (``scripts/reference/reference_r_reclassification.R``): the Mayo
PBC trial at 5 years, Cox models of age and sex against age, sex and log
bilirubin, with survIDINRI's own predicted risks.

- ``nricens`` (1.6): the continuous and the categorical NRI (cut-offs 0.2
  and 0.4), its event and non-event parts and the four shares, by IPCW
  and by Kaplan-Meier, agree to 1e-12. Its bootstrap bounds are random:
  each of ours (1000 resamples) must be within a quarter of the
  interval's half-width of R's (Monte Carlo error of a 2.5% quantile of
  1000 resamples is about 0.06 of the half-width per side).
- ``survIDINRI`` (1.1-2): its estimator ``IDI.FUN`` given the same
  Kaplan-Meier censoring weights averages over a grid of about 2000
  points on [-1, 1] by the trapezoid rule, which is within half a grid
  step, 5e-4, of each exact mean; the IDI and its parts agree to 1e-3.
  Its M2 (``P(up | event) - P(up | non-event)``) agrees to 1e-5 (it is
  read from survIDINRI's compiled empirical distribution, which differs
  from the exact share in the seventh digit). ``IDI.INF`` itself weights
  by the Fleming-Harrington estimate of the censoring; it agrees to 1e-3
  too.
- The perturbation bounds with the risks held fixed (survIDINRI's
  perturbation of the censoring weights and the empirical distributions,
  4000 draws, without refitting the Cox models) against ours (4000
  draws): within a quarter of the half-width. ``IDI.INF``'s own bounds
  also perturb the Cox fits, so they are wider; ours, which hold the
  risks fixed, must lie inside them.

The tolerances were set before the comparison was run.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from surpyval.metrics import idi_nri

from ._data import fixture, values

SOURCE = "r_reclassification"
T0 = 5.0
CUTS = [0.2, 0.4]


@pytest.fixture(scope="module")
def data():
    d = fixture("pbc")
    risks = values(SOURCE, "idi_inf")
    x = d["years"]
    c = (d["cause"] != 2).astype(int)
    return x, c, risks["p_old"], risks["p_new"]


def _names(prefix, suffix):
    return [
        prefix,
        prefix + "_events",
        prefix + "_nonevents",
        "p_up_events" + suffix,
        "p_down_events" + suffix,
        "p_down_nonevents" + suffix,
        "p_up_nonevents" + suffix,
    ]


@pytest.mark.parametrize("method", ["ipw", "km"])
def test_nri_matches_nricens(data, method):
    x, c, old, new = data
    res = idi_nri(
        x,
        c,
        old,
        new,
        T0,
        cutoffs=CUTS,
        censoring="ipcw" if method == "ipw" else "km",
        bounds="bootstrap",
        n_resamples=1000,
        random_state=20261113,
    )
    est = res.estimates.set_index("measure")
    for updown, prefix, suffix in (
        ("diff", "nri", ""),
        ("category", "nri_categorical", "_categorical"),
    ):
        ref = values(SOURCE, f"nricens_{updown}_{method}")
        names = _names(prefix, suffix)
        assert_allclose(
            est.loc[names, "estimate"], ref["estimate"], rtol=0, atol=1e-12
        )
        half = (ref["upper"] - ref["lower"]) / 2
        assert np.all(
            np.abs(est.loc[names, "lower"] - ref["lower"]) <= half / 4
        )
        assert np.all(
            np.abs(est.loc[names, "upper"] - ref["upper"]) <= half / 4
        )


def test_idi_matches_survidinri_estimator(data):
    x, c, old, new = data
    res = idi_nri(x, c, old, new, T0, bounds="none")
    est = res.estimates.set_index("measure")["estimate"]
    ref = values(SOURCE, "idi_fun_km")
    assert_allclose(est["idi"], ref["idi"], rtol=0, atol=1e-3)
    assert_allclose(est["idi_events"], ref["idi_d1"], rtol=0, atol=1e-3)
    # survIDINRI's non-event part is the mean rise; ours the mean fall
    assert_allclose(est["idi_nonevents"], -ref["idi_d0"], rtol=0, atol=1e-3)
    m2 = est["p_up_events"] - est["p_up_nonevents"]
    assert_allclose(m2, ref["txi"], rtol=0, atol=1e-5)
    assert_allclose(est["p_up_events"], ref["txi_d1"], rtol=0, atol=1e-5)
    assert_allclose(est["p_up_nonevents"], ref["txi_d0"], rtol=0, atol=1e-5)
    inf = values(SOURCE, "idi_inf")
    assert_allclose(est["idi"], inf["m1_est"][0], rtol=0, atol=1e-3)
    assert_allclose(m2, inf["m2_est"][0], rtol=0, atol=1e-5)


def test_perturbation_bounds_match_fixed_risk_perturbation(data):
    x, c, old, new = data
    res = idi_nri(x, c, old, new, T0, n_resamples=4000, random_state=20261113)
    est = res.estimates.set_index("measure")
    ref = values(SOURCE, "idi_fixed_risk_perturbation")
    lo, hi = ref["idi_bounds"]
    half = (hi - lo) / 2
    assert abs(est.loc["idi", "lower"] - lo) <= half / 4
    assert abs(est.loc["idi", "upper"] - hi) <= half / 4
    # survIDINRI's own bounds also carry the variation of the Cox fits
    inf = values(SOURCE, "idi_inf")["m1"]
    assert inf[1] <= est.loc["idi", "lower"] <= est.loc["idi", "upper"]
    assert est.loc["idi", "upper"] <= inf[2]
