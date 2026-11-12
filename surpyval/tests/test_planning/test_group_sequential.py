"""Group-sequential designs: the numerical integration against an
independent multivariate normal integral, the published constants of
Jennison and Turnbull (2000), the designs' own consistency (levels,
power, spending, the analysis at a stop), serialisation and the
refusals. The comparisons with R's gsDesign and rpact are in
``tests/reference/test_group_sequential.py``."""

import json
from typing import Any

import numpy as np
import pytest
from scipy import stats

from surpyval.test_planning import (
    GroupSequentialDesign,
    LogrankGroupSequentialDesign,
    boundary_crossing,
    group_sequential_design,
    logrank_events,
    logrank_group_sequential,
    logrank_sample_size,
)

ONE_SIDED: dict[str, Any] = {"alternative": "one-sided", "alpha_ci": 0.025}


def _mvn_crossing(timing, lower, upper, drift=0.0):
    """The same chances as boundary_crossing, from scipy's multivariate
    normal integral (Genz's quasi-Monte Carlo method, seeded): the chance
    of staying between the boundaries to look k - 1 and leaving above at
    look k. In two and three dimensions it agrees with the recursive
    integration on a fine grid (r = 128) to about 1e-10; in four and
    five its own error reaches about 4e-7, whatever the tolerance asked
    for (raising its number of points enough to do better takes
    minutes). The default grid (r = 32) is within about 1e-8 of the fine
    one. The tolerances below are set from these."""
    t = np.asarray(timing, dtype=float)
    cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
    mean = drift * np.sqrt(t)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    def inside(k, top):
        """P(lower_j < Z_j < upper_j for j < k, Z_k < top)."""
        dist = stats.multivariate_normal(
            mean=mean[: k + 1],
            cov=cov[: k + 1, : k + 1],
            abseps=1e-11,
            releps=1e-11,
            seed=20261113,
        )
        lo = np.r_[lower[:k], -np.inf]
        hi = np.r_[upper[:k], top]
        return dist.cdf(hi, lower_limit=lo)

    up = []
    for k in range(t.size):
        reach = inside(k, np.inf)
        up.append(reach - inside(k, upper[k]))
    return np.array(up)


# --- the numerical integration -------------------------------------------


@pytest.mark.parametrize(
    "timing, lower, upper, drift",
    [
        ([0.5, 1.0], [-2.796, -1.977], [2.796, 1.977], 0.0),
        ([1 / 3, 2 / 3, 1.0], [-np.inf] * 3, [3.71, 2.511, 1.993], 3.3),
        (
            [0.2, 0.45, 0.7, 1.0],
            [-0.5, 0.3, 1.0, 2.0],
            [3.5, 2.8, 2.3, 2.0],
            2.5,
        ),
        ([0.3, 0.6, 1.0], [-1.0, -0.2, -np.inf], [2.5, 2.4, 2.3], -1.0),
    ],
)
def test_integration_against_multivariate_normal(timing, lower, upper, drift):
    up, _ = boundary_crossing(timing, lower, upper, drift)
    np.testing.assert_allclose(
        up, _mvn_crossing(timing, lower, upper, drift), rtol=0, atol=5e-8
    )
    # and the integration has converged: a grid four times finer agrees
    fine, _ = boundary_crossing(timing, lower, upper, drift, r=128)
    np.testing.assert_allclose(up, fine, rtol=0, atol=2e-8)


def test_integration_lower_crossings_mirror_upper():
    t = [0.25, 0.5, 0.75, 1.0]
    upper = [3.0, 2.6, 2.3, 2.1]
    lower = [-1.0, -0.5, 0.0, 0.5]
    up, low = boundary_crossing(t, lower, upper, 1.2)
    mirrored_up, mirrored_low = boundary_crossing(
        t, -np.asarray(upper), -np.asarray(lower), -1.2
    )
    np.testing.assert_allclose(low, mirrored_up, atol=1e-12)
    np.testing.assert_allclose(up, mirrored_low, atol=1e-12)
    # every path stops by the last look, if the boundaries meet there
    up, low = boundary_crossing(t, lower[:-1] + [2.1], upper, 1.2)
    assert up.sum() + low.sum() == pytest.approx(1.0, abs=1e-8)


def test_integration_one_look_is_the_normal():
    up, low = boundary_crossing([1.0], [-1.0], [1.5], 0.7)
    assert up[0] == pytest.approx(stats.norm.sf(1.5 - 0.7), abs=1e-14)
    assert low[0] == pytest.approx(stats.norm.cdf(-1.0 - 0.7), abs=1e-14)


def test_integration_finer_grid_agrees():
    args = (
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [-np.inf] * 5,
        [4.2, 3.3, 2.7, 2.3, 2.0],
    )
    coarse, _ = boundary_crossing(*args, 2.0, r=16)
    fine, _ = boundary_crossing(*args, 2.0, r=64)
    np.testing.assert_allclose(coarse, fine, atol=1e-7)


# --- published constants -------------------------------------------------


@pytest.mark.parametrize(
    "looks, pocock, obrien_fleming",
    # Jennison and Turnbull (2000), tables 2.1 and 2.3: the constants
    # C_P(K, 0.05) and C_B(K, 0.05) of the two-sided 5% tests with K
    # equally spaced looks (to the three decimals printed; gsDesign gives
    # the same, tests/reference/test_group_sequential.py).
    [
        (2, 2.178, 1.977),
        (3, 2.289, 2.004),
        (4, 2.361, 2.024),
        (5, 2.413, 2.040),
        (10, 2.555, 2.087),
    ],
)
def test_jennison_turnbull_constants(looks, pocock, obrien_fleming):
    p = group_sequential_design(looks, efficacy="pocock")
    np.testing.assert_allclose(p.efficacy_z, pocock, atol=5e-4)
    of = group_sequential_design(looks, efficacy="obrien-fleming")
    k = np.arange(1, looks + 1)
    np.testing.assert_allclose(
        of.efficacy_z, obrien_fleming * np.sqrt(looks / k), atol=5e-4 * 3.2
    )
    assert of.efficacy_z[-1] == pytest.approx(obrien_fleming, abs=5e-4)


@pytest.mark.parametrize(
    "looks, pocock, obrien_fleming",
    # Jennison and Turnbull (2000), tables 2.2 and 2.4: the maximum
    # sample size over the fixed test's, R_P(K, 0.05, 0.2) and
    # R_B(K, 0.05, 0.2), two-sided 5%, 80% power.
    [
        (2, 1.110, 1.008),
        (3, 1.166, 1.017),
        (4, 1.202, 1.024),
        (5, 1.229, 1.028),
    ],
)
def test_jennison_turnbull_inflation(looks, pocock, obrien_fleming):
    p = group_sequential_design(looks, efficacy="pocock")
    assert p.inflation_factor == pytest.approx(pocock, abs=5e-4)
    of = group_sequential_design(looks, efficacy="obrien-fleming")
    assert of.inflation_factor == pytest.approx(obrien_fleming, abs=5e-4)


# --- the designs keep what they promise ----------------------------------


DESIGNS = {
    "ldof two-sided": {},
    "pocock spending": {"efficacy": "lan-demets-pocock", "looks": 4},
    "hsd one-sided": {
        **ONE_SIDED,
        "efficacy": "hwang-shih-decani",
        "efficacy_parameter": -4,
        "looks": [0.2, 0.45, 0.7, 0.85, 1.0],
        "power": 0.9,
    },
    "kim-demets": {**ONE_SIDED, "efficacy": "kim-demets"},
    "wang-tsiatis": {"efficacy": "wang-tsiatis", "efficacy_parameter": 0.25},
    "haybittle-peto": {"efficacy": "haybittle-peto", "looks": 4},
    "non-binding": {
        **ONE_SIDED,
        "futility": "hwang-shih-decani",
        "futility_parameter": -2,
        "power": 0.9,
    },
    "binding": {
        **ONE_SIDED,
        "futility": "lan-demets-obrien-fleming",
        "binding": True,
        "efficacy": "lan-demets-pocock",
        "looks": 4,
    },
    "non-binding classical": {
        **ONE_SIDED,
        "efficacy": "obrien-fleming",
        "futility": "kim-demets",
        "futility_parameter": 2.0,
    },
}


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_level_and_power(name):
    kwargs = dict(DESIGNS[name])
    design = group_sequential_design(**kwargs)
    t = design.timing
    level = kwargs.get("alpha_ci", 0.05)
    # the level, from the independent integral: two-sided both sides, a
    # non-binding futility boundary ignored
    if design.alternative == "two-sided":
        lower = -design.efficacy_z
    elif design.futility_z is not None and design.binding:
        lower = design.futility_z
    else:
        lower = np.full(t.size, -np.inf)
    two = 2.0 if design.alternative == "two-sided" else 1.0
    if design.looks <= 3:  # Genz's integral is slow and coarse beyond
        up = _mvn_crossing(t, lower, design.efficacy_z)
        assert two * up.sum() == pytest.approx(level, abs=5e-8)
    fine, _ = boundary_crossing(t, lower, design.efficacy_z, r=128)
    assert two * fine.sum() == pytest.approx(level, abs=2e-8)
    assert design.alpha_spent[-1] == pytest.approx(level, abs=1e-8)
    # the power, at the drift
    assert design.power == pytest.approx(kwargs.get("power", 0.8), abs=1e-8)
    if design.futility_z is not None:
        assert design.beta_spent[-1] == pytest.approx(
            1.0 - design.power, abs=1e-8
        )
        assert design.futility_z[-1] == design.efficacy_z[-1]
        assert np.all(design.futility_z <= design.efficacy_z)
    # a design no smaller than one analysis, and stopping on average
    # before the end
    assert design.inflation_factor >= 1.0
    assert 0.0 < design.expected_information["alternative"] < 1.0
    for stops in (design.stop_null, design.stop_alternative):
        total = stops["efficacy"].sum() + stops["futility"].sum()
        assert total == pytest.approx(1.0, abs=1e-12)


def test_spending_functions_are_spent():
    """Each look spends exactly its share of the level."""
    t = np.array([0.2, 0.45, 0.7, 0.85, 1.0])
    alpha = 0.025
    cases = {
        "lan-demets-obrien-fleming": (
            None,
            2 * stats.norm.sf(stats.norm.isf(alpha / 2) / np.sqrt(t)),
        ),
        "lan-demets-pocock": (None, alpha * np.log1p((np.e - 1) * t)),
        "hwang-shih-decani": (
            -4.0,
            alpha * np.expm1(4.0 * t) / np.expm1(4.0),
        ),
        "kim-demets": (3.0, alpha * t**3),
    }
    for name, (param, spent) in cases.items():
        design = group_sequential_design(
            t, efficacy=name, efficacy_parameter=param, **ONE_SIDED
        )
        np.testing.assert_allclose(design.alpha_spent, spent, atol=1e-9)
    hsd0 = group_sequential_design(
        t, efficacy="hwang-shih-decani", efficacy_parameter=0.0, **ONE_SIDED
    )
    np.testing.assert_allclose(hsd0.alpha_spent, alpha * t, atol=1e-9)


def test_two_sided_spends_half_on_each_side():
    two = group_sequential_design(3)
    one = group_sequential_design(3, **ONE_SIDED)
    np.testing.assert_allclose(two.efficacy_z, one.efficacy_z, atol=1e-7)
    np.testing.assert_allclose(two.alpha_spent, 2 * one.alpha_spent, atol=1e-9)
    np.testing.assert_allclose(two.efficacy_p, 2 * one.efficacy_p, rtol=1e-6)


def test_wang_tsiatis_spans_obrien_fleming_and_pocock():
    for delta, name in ((0.0, "obrien-fleming"), (0.5, "pocock")):
        wt = group_sequential_design(
            4, efficacy="wang-tsiatis", efficacy_parameter=delta
        )
        np.testing.assert_allclose(
            wt.efficacy_z,
            group_sequential_design(4, efficacy=name).efficacy_z,
            atol=1e-9,
        )


def test_haybittle_peto():
    design = group_sequential_design(4, efficacy="haybittle-peto")
    np.testing.assert_array_equal(design.efficacy_z[:-1], 3.0)
    assert 1.96 < design.efficacy_z[-1] < 2.0
    looser = group_sequential_design(
        4, efficacy="haybittle-peto", efficacy_parameter=2.5
    )
    assert looser.efficacy_z[-1] > design.efficacy_z[-1]


def test_one_look_is_the_fixed_test():
    design = group_sequential_design(1)
    assert design.efficacy_z[0] == pytest.approx(stats.norm.isf(0.025))
    assert design.inflation_factor == pytest.approx(1.0, abs=1e-9)


def test_binding_lowers_the_efficacy_boundary():
    kwargs = {**ONE_SIDED, "futility": "hwang-shih-decani"}
    free = group_sequential_design(3, futility_parameter=-2, **kwargs)
    bound = group_sequential_design(
        3, futility_parameter=-2, binding=True, **kwargs
    )
    assert bound.efficacy_z[-1] < free.efficacy_z[-1]
    assert bound.inflation_factor < free.inflation_factor
    # the first look's efficacy boundary has nothing earlier to differ by
    assert bound.efficacy_z[0] == pytest.approx(free.efficacy_z[0])


def test_crossing_probabilities_and_max_information():
    design = group_sequential_design(3)
    null = design.crossing_probabilities(0.0)
    np.testing.assert_allclose(
        np.cumsum(null["efficacy"]), design.alpha_spent, atol=1e-12
    )
    at = design.crossing_probabilities(design.drift)
    # the power, and the (tiny) chance of stopping on the wrong side
    assert design.power == pytest.approx(0.8, abs=1e-9)
    wrong = at["efficacy"].sum() - design.power
    assert 0.0 < wrong < 1e-5
    assert design.max_information(0.25) == pytest.approx(
        (design.drift / 0.25) ** 2
    )


# --- using a design ------------------------------------------------------


def test_decide():
    design = group_sequential_design(
        3, futility="hwang-shih-decani", futility_parameter=-2, **ONE_SIDED
    )
    b, a = design.efficacy_z, design.futility_z
    assert design.decide(1, b[0]) == "efficacy"
    assert design.decide(1, b[0] - 1e-9) == "continue"
    assert design.decide(2, a[1]) == "futility"
    assert design.decide(2, a[1] + 1e-9) == "continue"
    assert design.decide(3, b[2] - 1e-9) == "futility"
    two = group_sequential_design(3)
    assert two.decide(2, -two.efficacy_z[1]) == "efficacy"
    assert two.decide(2, 0.0) == "continue"


def test_p_value_agrees_with_the_boundaries():
    for kwargs in ({}, ONE_SIDED, DESIGNS["binding"]):
        design = group_sequential_design(**{"looks": 3, **kwargs})
        level = design.alpha_ci
        for k in range(design.looks):
            b = design.efficacy_z[k]
            # on the boundary, the level spent so far
            assert design.p_value(k + 1, b) == pytest.approx(
                design.alpha_spent[k], abs=1e-9
            )
            assert design.p_value(k + 1, b + 0.01) < level
        last = design.looks
        assert design.p_value(last, design.efficacy_z[-1] - 0.01) > level
    # at the first look, the plain normal p-value
    design = group_sequential_design(3, **ONE_SIDED)
    assert design.p_value(1, 4.0) == pytest.approx(stats.norm.sf(4.0))


def test_estimate_is_median_unbiased_and_covers():
    design = group_sequential_design(3, **ONE_SIDED)
    est = design.estimate(2, 2.6)
    # by its definition: at the estimate the exceedance is one half
    assert design._exceedance(1, 2.6, est["estimate"]) == pytest.approx(0.5)
    assert design._exceedance(1, 2.6, est["lower"]) == pytest.approx(0.025)
    assert est["lower"] < est["estimate"] < est["upper"]
    # it pulls the naive estimate z / sqrt(t) back towards 0
    assert est["estimate"] < 2.6 / np.sqrt(design.timing[1])
    # at a single look it is the naive estimate and the Wald interval
    one = group_sequential_design(1, **ONE_SIDED).estimate(1, 2.5)
    assert one["estimate"] == pytest.approx(2.5, abs=1e-8)
    assert one["lower"] == pytest.approx(2.5 - 1.959964, abs=1e-6)
    # a two-sided design, stopped on the lower side, mirrors the upper
    two = group_sequential_design(3)
    up, down = two.estimate(2, 2.7), two.estimate(2, -2.7)
    assert down["estimate"] == pytest.approx(-up["estimate"])
    assert down["lower"] == pytest.approx(-up["upper"])


def test_repeated_interval():
    design = group_sequential_design(3)
    lo, hi = design.repeated_interval(2, 1.0)
    t, b = design.timing[1], design.efficacy_z[1]
    assert (lo, hi) == pytest.approx(
        ((1 - b) / np.sqrt(t), (1 + b) / np.sqrt(t))
    )


def test_estimates_refuse_a_look_not_stopped_at():
    design = group_sequential_design(3)
    with pytest.raises(ValueError, match="repeated_interval"):
        design.p_value(1, 1.0)
    with pytest.raises(ValueError, match="repeated_interval"):
        design.estimate(2, 1.0)
    with pytest.raises(ValueError, match="look must be at most 3"):
        design.decide(4, 1.0)
    with pytest.raises(ValueError, match="z must be a finite"):
        design.p_value(3, np.nan)


# --- re-computing at the looks that happened -----------------------------


def test_update_at_the_planned_looks_changes_nothing():
    design = group_sequential_design(
        3, futility="hwang-shih-decani", futility_parameter=-2, **ONE_SIDED
    )
    same = design.update(design.timing)
    np.testing.assert_allclose(same.efficacy_z, design.efficacy_z, atol=1e-9)
    np.testing.assert_allclose(same.futility_z, design.futility_z, atol=1e-9)
    assert same.power == pytest.approx(design.power, abs=1e-9)


def test_update_spends_what_is_left_at_the_last_look():
    design = group_sequential_design(3, **ONE_SIDED)
    for last in (0.9, 1.0, 1.1):
        actual = design.update([0.35, 0.6, last])
        assert actual.alpha_spent[-1] == pytest.approx(0.025, abs=1e-9)
        assert actual.alpha_spent[0] == pytest.approx(
            2 * stats.norm.sf(stats.norm.isf(0.0125) / np.sqrt(0.35)),
            abs=1e-12,
        )
    # more information than planned: more power; less, less
    assert design.update([1 / 3, 2 / 3, 1.1]).power > design.power
    assert design.update([1 / 3, 2 / 3, 0.9]).power < design.power
    # an extra look
    four = design.update([0.25, 0.5, 0.75, 1.0])
    assert four.looks == 4
    assert four.alpha_spent[-1] == pytest.approx(0.025, abs=1e-9)


def test_update_refuses_classical_boundaries():
    design = group_sequential_design(3, efficacy="pocock")
    with pytest.raises(ValueError, match="spending function"):
        design.update([0.3, 0.7, 1.0])


def test_logrank_update_takes_events():
    design = logrank_group_sequential(0.7, 3, 0.9, **ONE_SIDED)
    actual = design.update([130, 240, 337])
    assert isinstance(actual, LogrankGroupSequentialDesign)
    assert actual.events.tolist() == [130, 240, 337]
    assert actual.max_events == design.max_events
    np.testing.assert_allclose(
        actual.timing, np.array([130, 240, 337]) / design.max_events
    )
    with pytest.raises(ValueError, match="events must be positive"):
        design.update([0, 240, 337])


# --- the log-rank front door ---------------------------------------------


@pytest.mark.parametrize(
    "hazard_ratio, kwargs",
    [
        (0.7, {}),
        (0.5, {"power": 0.9, "ratio": 2.0}),
        (1.6, {"method": "freedman", **ONE_SIDED}),
    ],
)
def test_logrank_events_follow_the_fixed_design(hazard_ratio, kwargs):
    design = logrank_group_sequential(hazard_ratio, 4, **kwargs)
    power = kwargs.pop("power", 0.8)
    fixed = logrank_events(hazard_ratio, power, **kwargs)
    assert np.ceil(design.fixed_events - 1e-9) == fixed
    exact = design.fixed_events * design.inflation_factor
    assert design.max_events == int(np.ceil(exact - 1e-9))
    assert design.events[-1] == design.max_events
    assert np.all(np.diff(design.events) > 0)


def test_logrank_units_follow_the_sample_size():
    design = logrank_group_sequential(0.7, 3, event_probability=(0.25, 0.35))
    fixed = logrank_sample_size(0.7, (0.25, 0.35))
    # the units grow with the inflation factor
    assert design.n[0] == pytest.approx(
        fixed[0] * design.inflation_factor, abs=2
    )
    one = logrank_group_sequential(0.7, 1, event_probability=(0.25, 0.35))
    assert one.n == fixed


def test_logrank_hazard_ratio_at_the_boundaries():
    design = logrank_group_sequential(0.7, 3)
    hr = design.hazard_ratio_bounds["efficacy"]
    # the observed hazard ratio that is on the boundary: below 1, nearer
    # 1 at later looks
    assert np.all(hr < 1.0) and np.all(np.diff(hr) > 0)
    d = design.events
    np.testing.assert_allclose(
        hr, np.exp(-design.efficacy_z / np.sqrt(d / 4)), rtol=1e-12
    )
    above = logrank_group_sequential(1 / 0.7, 3)
    np.testing.assert_allclose(
        above.hazard_ratio_bounds["efficacy"], 1 / hr, rtol=1e-12
    )


def test_logrank_hazard_ratio_estimate():
    design = logrank_group_sequential(0.7, 3, 0.9, **ONE_SIDED)
    est = design.hazard_ratio_estimate(2, 2.6)
    drift = design.estimate(2, 2.6)
    scale = np.sqrt(design.max_events / 4)
    assert est["estimate"] == pytest.approx(np.exp(-drift["estimate"] / scale))
    assert est["lower"] < est["estimate"] < est["upper"] < 1.0
    lo, hi = est["repeated"]
    assert lo < est["lower"] and hi > est["upper"]
    # at a look it went on from, only the repeated interval
    early = design.hazard_ratio_estimate(1, 1.0)
    assert early["estimate"] is None and len(early["repeated"]) == 2


# --- serialisation and reading -------------------------------------------


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_to_dict_round_trip(name):
    design = group_sequential_design(**DESIGNS[name])
    d = design.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    back = GroupSequentialDesign.from_dict(d)
    for attr in ("timing", "efficacy_z", "efficacy_p", "alpha_spent"):
        np.testing.assert_array_equal(
            getattr(back, attr), getattr(design, attr)
        )
    assert back.drift == design.drift
    assert back.decide(2, 2.3) == design.decide(2, 2.3)
    assert repr(back) == repr(design)
    if design.efficacy not in (
        "pocock",
        "obrien-fleming",
        "wang-tsiatis",
        "haybittle-peto",
    ):
        np.testing.assert_allclose(
            back.update([0.3, 0.7, 1.0]).efficacy_z,
            design.update([0.3, 0.7, 1.0]).efficacy_z,
        )


def test_logrank_to_dict_round_trip():
    design = logrank_group_sequential(
        0.7, 3, futility="kim-demets", event_probability=0.3, **ONE_SIDED
    )
    d = design.to_dict()
    assert d["result"] == "LogrankGroupSequentialDesign"
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    back = LogrankGroupSequentialDesign.from_dict(d)
    assert back.max_events == design.max_events and back.n == design.n
    assert back.events.tolist() == design.events.tolist()
    assert repr(back) == repr(design)
    with pytest.raises(ValueError, match="Not a GroupSequentialDesign"):
        GroupSequentialDesign.from_dict(d)


def test_infinite_boundary_is_written_as_none():
    # a look that spends (almost) nothing cannot stop for efficacy
    design = group_sequential_design([1e-3, 1.0])
    assert design.efficacy_z[0] == np.inf
    d = design.to_dict()
    assert d["efficacy_z"][0] is None
    assert "/efficacy_z/0" in d["non_finite"]["inf"]
    assert GroupSequentialDesign.from_dict(d).efficacy_z[0] == np.inf


def test_repr_reads_plainly():
    text = repr(
        group_sequential_design(
            3, futility="hwang-shih-decani", futility_parameter=-2, **ONE_SIDED
        )
    )
    assert "Looks            : 3, at 33.33%, 66.67%, 100%" in text
    assert "Hwang-Shih-DeCani spending (gamma -2), non-binding" in text
    assert "Futility z" in text and "Inflation factor" in text
    text = repr(logrank_group_sequential(0.7, 3, event_probability=0.3))
    assert "Events" in text and "Units" in text and "HR at bound" in text


# --- refusals ------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"looks": 0}, "looks must be a whole number"),
        ({"looks": [0.5, 0.9]}, "ending at 1"),
        ({"looks": [0.6, 0.4, 1.0]}, "increasing"),
        ({"power": 1.2}, "power must be between 0 and 1"),
        ({"power": 0.02}, "power must be more than the level"),
        ({"alternative": "less"}, "'alternative' must be one of"),
        ({"efficacy": "obf"}, "'efficacy' must be one of"),
        ({"efficacy": "wang-tsiatis"}, "needs efficacy_parameter"),
        (
            {"efficacy": "pocock", "efficacy_parameter": 1.0},
            "takes no parameter",
        ),
        (
            {"efficacy": "kim-demets", "efficacy_parameter": -1},
            "must be positive",
        ),
        ({"futility": "pocock", **ONE_SIDED}, "beta spending"),
        ({"futility": "kim-demets"}, "one-sided design"),
        (
            {
                "futility": "kim-demets",
                "binding": True,
                "efficacy": "pocock",
                **ONE_SIDED,
            },
            "alpha-spending",
        ),
        ({"binding": True}, "needs a futility boundary"),
        ({"futility_parameter": 2.0}, "without a futility boundary"),
        ({"binding": 1}, "binding must be True or False"),
        (
            {
                "efficacy": "haybittle-peto",
                "efficacy_parameter": 1.0,
                "looks": 4,
            },
            "raise efficacy_parameter",
        ),
    ],
)
def test_refusals_say_what_to_do(kwargs, match):
    with pytest.raises(ValueError, match=match):
        group_sequential_design(**kwargs)


def test_logrank_refusals():
    with pytest.raises(ValueError, match="hazard_ratio must differ from 1"):
        logrank_group_sequential(1.0)
    with pytest.raises(ValueError, match="'method' must be one of"):
        logrank_group_sequential(0.7, method="lachin")
    with pytest.raises(ValueError, match="event_probability"):
        logrank_group_sequential(0.7, event_probability=(0.2, 0.3, 0.4))


def test_boundary_crossing_refusals():
    with pytest.raises(ValueError, match="increasing"):
        boundary_crossing([0.5, 0.5], [-1, -1], [1, 1])
    with pytest.raises(ValueError, match="NaN"):
        boundary_crossing([0.5, 1.0], [-1, np.nan], [1, 1])
