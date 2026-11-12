"""Group-sequential designs against R's gsDesign and rpact.

``data/group_sequential_r.json`` is written by
``scripts/reference/reference_r_group_sequential.R`` (R 4.6.2, gsDesign
3.11.0, rpact 4.4.0). Both packages integrate the same multivariate
normal by the same recursion (gsDesign on a coarser grid, r = 18, and
with a boundary tolerance of about 1e-6), so the boundaries, inflation
factors, crossing probabilities and expected information agree to about
1e-6 (tolerance 2e-6), with these exceptions, each investigated:

* gsDesign's designs with a *binding* futility boundary differ by about
  2e-5 in the last efficacy boundary. Its level is then 0.0250009, by an
  independent multivariate normal integral (scipy, Genz), and
  SurPyval's 0.0250000 (``test_planning/test_group_sequential.py``
  checks SurPyval's), so the difference is gsDesign's; they are held to
  5e-5.
* rpact writes the last alpha spent as ``alpha - 1e-8``, its own
  tolerance; held to 2e-8.
* gsDesign spends, at a last look before the planned information, only
  the level its fraction allows; SurPyval spends all that is left. The
  re-computed designs are compared with the last look past the plan,
  where the two agree.
* gsDesign and rpact work with the unrounded maximum events; SurPyval's
  ``max_events`` is rounded up, so the comparisons of events use the
  unrounded value, ``fixed_events * inflation_factor``.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from surpyval.test_planning import (
    GroupSequentialDesign,
    boundary_crossing,
    group_sequential_design,
    logrank_group_sequential,
)

DATA = Path(__file__).parent / "data" / "group_sequential_r.json"
TOL = 2e-6
ONE = {"alternative": "one-sided", "alpha_ci": 0.025}


@pytest.fixture(scope="module")
def refs():
    with open(DATA) as f:
        return json.load(f)["references"]


# gsDesign's one-sided alpha is SurPyval's alpha_ci for a one-sided
# design; test.type = 2 is the two-sided design at twice it.
GSDESIGN = {
    "ldof_one_sided": dict(looks=3, power=0.9, **ONE),
    "ldpocock_two_sided": dict(
        looks=4, power=0.8, efficacy="lan-demets-pocock"
    ),
    "ldof_two_sided": dict(looks=3, power=0.8),
    "hsd_unequal": dict(
        looks=[0.2, 0.45, 0.7, 0.85, 1.0],
        power=0.9,
        efficacy="hwang-shih-decani",
        efficacy_parameter=-4,
        **ONE,
    ),
    "nonbinding_hsd": dict(
        looks=3,
        power=0.9,
        futility="hwang-shih-decani",
        futility_parameter=-2,
        **ONE,
    ),
    "binding_hsd": dict(
        looks=3,
        power=0.9,
        futility="hwang-shih-decani",
        futility_parameter=-2,
        binding=True,
        **ONE,
    ),
    "power_nonbinding": dict(
        looks=[0.3, 0.5, 0.8, 1.0],
        power=0.8,
        alpha_ci=0.05,
        alternative="one-sided",
        efficacy="kim-demets",
        efficacy_parameter=3,
        futility="kim-demets",
        futility_parameter=2,
    ),
    "ldpocock_binding": dict(
        looks=4,
        power=0.8,
        efficacy="lan-demets-pocock",
        futility="lan-demets-obrien-fleming",
        binding=True,
        **ONE,
    ),
    "classical_of": dict(looks=4, power=0.8, efficacy="obrien-fleming"),
    "classical_pocock": dict(looks=5, power=0.9, efficacy="pocock"),
    "classical_wt": dict(
        looks=3,
        power=0.9,
        efficacy="wang-tsiatis",
        efficacy_parameter=0.25,
        **ONE,
    ),
}


@pytest.mark.parametrize("ref_id", sorted(GSDESIGN))
def test_against_gsdesign(refs, ref_id):
    v = refs[ref_id]["values"]
    design = group_sequential_design(**GSDESIGN[ref_id])
    tol = 5e-5 if "binding" in ref_id and "non" not in ref_id else TOL
    np.testing.assert_allclose(design.timing, v["timing"], atol=1e-12)
    np.testing.assert_allclose(design.efficacy_z, v["upper"], atol=tol)
    if v["lower"] is not None:
        np.testing.assert_allclose(design.futility_z, v["lower"], atol=tol)
    assert design.inflation_factor == pytest.approx(
        v["inflation_factor"], abs=tol
    )
    np.testing.assert_allclose(
        design.timing * design.inflation_factor, v["information"], atol=tol
    )
    assert design.drift == pytest.approx(
        v["delta"] * np.sqrt(v["inflation_factor"]), abs=tol
    )
    # crossing probabilities: gsDesign's "upper" is the upper side only
    lower, upper = design._bounds()
    up0, low0 = boundary_crossing(design.timing, lower, upper, 0.0)
    up1, low1 = boundary_crossing(design.timing, lower, upper, design.drift)
    np.testing.assert_allclose(up0, v["upper_prob_null"], atol=tol)
    np.testing.assert_allclose(up1, v["upper_prob_alt"], atol=tol)
    if v["lower"] is not None:
        np.testing.assert_allclose(low1, v["lower_prob_alt"], atol=tol)
        np.testing.assert_allclose(low0, v["lower_prob_null"], atol=tol)
    # expected information, relative to one analysis
    expected = design.expected_information
    assert expected["null"] * design.inflation_factor == pytest.approx(
        v["expected_null"], abs=tol
    )
    assert expected["alternative"] * design.inflation_factor == (
        pytest.approx(v["expected_alt"], abs=tol)
    )


RPACT = {
    "rpact_asof_two_sided": dict(looks=3, power=0.8),
    "rpact_hp_two_sided": dict(looks=4, power=0.8, efficacy="haybittle-peto"),
    "rpact_wt_one_sided": dict(
        looks=[0.25, 0.5, 0.8, 1.0],
        power=0.9,
        efficacy="wang-tsiatis",
        efficacy_parameter=0.25,
        **ONE,
    ),
    "rpact_ksd_nonbinding": dict(
        looks=3,
        power=0.9,
        efficacy="kim-demets",
        efficacy_parameter=3,
        futility="kim-demets",
        futility_parameter=2,
        **ONE,
    ),
}


@pytest.mark.parametrize("ref_id", sorted(RPACT))
def test_against_rpact(refs, ref_id):
    v = refs[ref_id]["values"]
    design = group_sequential_design(**RPACT[ref_id])
    np.testing.assert_allclose(design.efficacy_z, v["critical"], atol=TOL)
    np.testing.assert_allclose(design.alpha_spent, v["alpha_spent"], atol=2e-8)
    if v["futility"] is not None:
        # rpact lists the interim futility boundaries only
        np.testing.assert_allclose(
            design.futility_z[:-1], v["futility"], atol=TOL
        )
    assert design.inflation_factor == pytest.approx(
        v["inflation_factor"], abs=TOL
    )
    expected = design.expected_information
    assert expected["null"] * design.inflation_factor == pytest.approx(
        v["expected_null"], abs=TOL
    )
    assert expected["alternative"] * design.inflation_factor == (
        pytest.approx(v["expected_alt"], abs=TOL)
    )


def test_logrank_events(refs):
    design = logrank_group_sequential(0.7, 3, 0.9, **ONE)
    exact = design.fixed_events * design.inflation_factor
    for ref_id in ("logrank_events", "rpact_logrank_events"):
        v = refs[ref_id]["values"]
        np.testing.assert_allclose(
            design.timing * exact, v["events"], rtol=1e-6
        )
        expected = design.expected_information
        assert expected["null"] * exact == pytest.approx(
            v["expected_null"], rel=1e-6
        )
        assert expected["alternative"] * exact == pytest.approx(
            v["expected_alt"], rel=1e-6
        )
    v = refs["logrank_events"]["values"]
    assert design.fixed_events == pytest.approx(v["fixed_events"], rel=1e-12)
    assert design.max_events == int(np.ceil(v["events"][-1]))


@pytest.mark.parametrize(
    "ref_id, kwargs, events",
    [
        ("logrank_update", {}, [130, 240, 337]),
        (
            "logrank_update_futility",
            {"futility": "hwang-shih-decani", "futility_parameter": -2},
            [120, 250, 360],
        ),
    ],
)
def test_update_against_gsdesign(refs, ref_id, kwargs, events):
    v = refs[ref_id]["values"]
    design = logrank_group_sequential(0.7, 3, 0.9, **ONE, **kwargs)
    exact = design.fixed_events * design.inflation_factor
    assert exact == pytest.approx(v["max_events"], rel=1e-6)
    # the fractions of the unrounded plan, as gsDesign's maxn.IPlan
    actual = GroupSequentialDesign.update(design, np.array(events) / exact)
    np.testing.assert_allclose(actual.efficacy_z, v["upper"], atol=TOL)
    if "lower" in v:
        np.testing.assert_allclose(actual.futility_z, v["lower"], atol=TOL)
    assert actual.power == pytest.approx(v["power"], abs=TOL)


def test_analysis_against_rpact(refs):
    """The stage-wise p-value, median-unbiased estimate and interval and
    the repeated intervals of a study stopped at its second look, on
    the hazard-ratio scale (log-rank information d / 4)."""
    v = refs["rpact_analysis"]["values"]
    design = group_sequential_design(3, **ONE)
    scale = np.sqrt(v["max_events"] / 4.0)
    assert design.p_value(2, v["z"][1]) == pytest.approx(
        v["p_value"], abs=1e-8
    )
    est = design.estimate(2, v["z"][1])
    for key in ("estimate", "lower", "upper"):
        assert np.exp(est[key] / scale) == pytest.approx(v[key], rel=1e-6)
    for k in range(2):
        lo, hi = design.repeated_interval(k + 1, v["z"][k])
        assert np.exp(lo / scale) == pytest.approx(
            v["repeated_lower"][k], rel=1e-6
        )
        assert np.exp(hi / scale) == pytest.approx(
            v["repeated_upper"][k], rel=1e-6
        )
