"""Reliability growth projection with fix-effectiveness factors (#607):
``CrowAMSAA.projection``, the AMSAA-Crow projection model and Crow's
extended model."""

import numpy as np
import pytest

from surpyval.recurrent import CrowAMSAA

# One prototype tested to 400 hours: A modes a1, a2; BD modes b1..b4.
X = [15, 42, 60, 98, 130, 171, 205, 260, 310, 345, 390, 400]
MODES: list = ["b1", "a1", "b2", "b1", "b3", "a2", "b2", "b4", "b1", "a1"]
MODES += ["b3", None]
C = [0] * 11 + [1]
FEF = {"b1": 0.8, "b2": 0.7, "b3": 0.75, "b4": 0.6}


def _acpm(T, n_a, counts, firsts, fefs, k=1):
    """The ACPM projection written out from MIL-HDBK-189C section 6.2."""
    total = k * T
    counts, firsts, fefs = map(np.asarray, (counts, firsts, fefs))
    K = len(counts)
    beta_bar = (K - 1) / K * K / np.sum(np.log(T / firsts))
    h = K * beta_bar / total
    potential = n_a / total + np.sum((1 - fefs) * counts) / total
    return potential + fefs.mean() * h, potential


def test_607_projection_matches_the_handbook_formula():
    result = CrowAMSAA.projection(X, MODES, FEF, c=C)
    projected, potential = _acpm(
        400.0, 3, [3, 2, 2, 1], [15, 60, 130, 260], [0.8, 0.7, 0.75, 0.6]
    )
    assert result.failures == {"A": 3, "BC": 0, "BD": 8}
    assert result.demonstrated_intensity == pytest.approx(11 / 400)
    assert result.projected_intensity == pytest.approx(projected, rel=1e-12)
    assert result.growth_potential_intensity == pytest.approx(
        potential, rel=1e-12
    )
    assert result.projected_mtbf == pytest.approx(1 / projected)
    # h(T) = K beta_bar / T with beta_bar = (K - 1) / sum log(T / t_i)
    logs = np.log(400.0 / np.array([15, 60, 130, 260]))
    assert result.beta == pytest.approx(3 / logs.sum())
    assert result.new_mode_intensity == pytest.approx(4 * result.beta / 400)
    assert result.mean_fef == pytest.approx(0.7125)
    assert list(result.modes.index) == ["b1", "b2", "b3", "b4"]
    assert result.modes["failures"].tolist() == [3, 2, 2, 1]
    assert result.modes["first"].tolist() == [15, 60, 130, 260]
    # The ordering every projection has: the fixes help, but less than
    # they would with every BD mode found.
    assert (
        result.demonstrated_mtbf
        < result.projected_mtbf
        < result.growth_potential_mtbf
    )
    assert "Projected" in repr(result)


def test_607_several_systems_pool_their_test_time():
    # Two prototypes side by side to 400 hours: the intensities are per
    # system over k T of test, and a mode's first occurrence is its
    # earliest on either.
    x = [15, 98, 130, 310, 400, 42, 60, 205, 260, 390, 400]
    i = [1] * 5 + [2] * 6
    c = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    modes = ["b1", "b1", "b3", "b1", None, "a1", "b2", "b2", "b1", "b3", None]
    fef = {"b1": 0.8, "b2": 0.7, "b3": 0.75}
    result = CrowAMSAA.projection(x, modes, fef, i=i, c=c)
    projected, potential = _acpm(
        400.0, 1, [4, 2, 2], [15, 60, 130], [0.8, 0.7, 0.75], k=2
    )
    assert result.systems == 2
    assert result.demonstrated_intensity == pytest.approx(9 / 800)
    assert result.projected_intensity == pytest.approx(projected, rel=1e-12)
    assert result.growth_potential_intensity == pytest.approx(potential)


def test_607_bc_modes_use_the_demonstrated_crow_amsaa_intensity():
    # With a mode fixed during the test the system grew, so the
    # demonstrated intensity is the Crow-AMSAA one at T (with the
    # bias-corrected shape, (N - 1) / N of the MLE: #730), and the BD
    # modes' intensities are taken off it.
    result = CrowAMSAA.projection(X, MODES, FEF, c=C, bc=["a2"])
    model = CrowAMSAA.fit(X, c=C)
    lam_ca = 10 / 11 * float(model.iif(400.0))
    assert result.failures == {"A": 2, "BC": 1, "BD": 8}
    assert result.demonstrated_intensity == pytest.approx(lam_ca)
    _, potential_acpm = _acpm(
        400.0, 3, [3, 2, 2, 1], [15, 60, 130, 260], [0.8, 0.7, 0.75, 0.6]
    )
    fixed = np.sum((1 - np.array([0.8, 0.7, 0.75, 0.6])) * [3, 2, 2, 1])
    assert result.growth_potential_intensity == pytest.approx(
        lam_ca - 8 / 400 + fixed / 400
    )
    assert result.projected_intensity == pytest.approx(
        result.growth_potential_intensity + 0.7125 * result.new_mode_intensity
    )
    assert result.model.params == pytest.approx(model.params)
    assert potential_acpm != pytest.approx(result.growth_potential_intensity)


def test_607_few_bd_modes():
    # No BD mode: nothing to project, the projection is the demonstrated
    # intensity. One BD mode: no shape for new modes (beta_bar = 0).
    none = CrowAMSAA.projection(X, MODES, {}, c=C)
    assert none.projected_intensity == pytest.approx(11 / 400)
    assert none.growth_potential_intensity == pytest.approx(11 / 400)
    assert np.isnan(none.beta) and none.new_mode_intensity == 0
    assert none.modes.empty
    one = CrowAMSAA.projection(X, MODES, {"b1": 1.0}, c=C)
    assert one.new_mode_intensity == 0
    assert one.projected_intensity == pytest.approx(8 / 400)


def test_607_projection_estimates_the_post_fix_intensity():
    # The projection's definition, simulated: B modes with many small
    # rates, each with its FEF; after the test the true intensity is
    # lam_A + sum lam_i (1 - d_i [mode i seen]). The ACPM estimate is
    # close to it on average, where leaving out the unseen modes (the
    # mu_d h(T) term) is far below it.
    rng = np.random.default_rng(607)
    T, lam_a = 1000.0, 0.004
    lam_b = rng.gamma(0.5, 1.0, 300)
    lam_b *= 0.03 / lam_b.sum()
    d = rng.uniform(0.5, 0.9, lam_b.size)
    labels = np.arange(lam_b.size)
    truth, estimate, naive = [], [], []
    for _ in range(150):
        n_a = rng.poisson(lam_a * T)
        n_b = rng.poisson(lam_b * T)
        times = [rng.uniform(0, T, n_a)]
        modes = [np.full(n_a, -1)]
        for label in labels[n_b > 0]:
            times.append(rng.uniform(0, T, n_b[label]))
            modes.append(np.full(n_b[label], label))
        x = np.concatenate(times)
        m = np.concatenate(modes).astype(object)
        order = np.argsort(x)
        seen = n_b > 0
        fef = {int(k): float(d[k]) for k in labels[seen]}
        result = CrowAMSAA.projection(
            np.append(x[order], T),
            np.append(m[order], None),
            fef,
            c=np.append(np.zeros(x.size), 1),
        )
        truth.append(lam_a + np.sum(lam_b * (1 - d * seen)))
        estimate.append(result.projected_intensity)
        naive.append(
            result.projected_intensity
            - result.mean_fef * result.new_mode_intensity
        )
    truth, estimate, naive = map(np.mean, (truth, estimate, naive))
    assert estimate == pytest.approx(truth, rel=0.08)
    assert naive < 0.6 * truth


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"fef": {"b1": 0.8}, "bc": ["b1"]}, "not both"),
        ({"fef": {"b9": 0.8}}, "no failures"),
        ({"fef": {"b1": 1.5}}, r"\[0, 1\]"),
        ({"fef": [("b1", 0.5)]}, "dict"),
        ({"fef": {}, "bc": "a2"}, "single string"),
    ],
)
def test_607_projection_input_errors(kwargs, message):
    with pytest.raises(ValueError, match=message):
        CrowAMSAA.projection(X, MODES, c=C, **kwargs)


def test_607_projection_needs_labels_and_a_time_terminated_test():
    with pytest.raises(ValueError, match="mode label"):
        CrowAMSAA.projection(X, [None] * len(X), FEF, c=C)
    with pytest.raises(ValueError, match="one label per row"):
        CrowAMSAA.projection(X, MODES[:-1], FEF, c=C)
    with pytest.raises(ValueError, match="time-terminated"):
        CrowAMSAA.projection(X[:-1], MODES[:-1], FEF)


def test_663_projection_of_systems_ending_at_different_times():
    # projection has no method=: the message says what the data needs
    x = X + [20.0, 300.0, 380.0]
    i = [1] * len(X) + [2] * 3
    with pytest.raises(ValueError, match="same end of test") as info:
        CrowAMSAA.projection(
            x, MODES + ["b1", "a1", None], FEF, i=i, c=C + [0, 0, 1]
        )
    assert "method=" not in str(info.value)


# MIL-HDBK-00189A's test-fix-find-test example (section 7.5), as ReliaWiki's
# "Crow Extended Model Examples" gives it: one system to T = 400, 56
# failures: 10 A, 14 BC (modes BC17..BC28) and 32 BD (modes BD1..BD16).
HDBK = """
0.7 BC17, 3.7 BC17, 13.2 BC17, 15 BD1, 17.6 BC18, 25.3 BD2, 47.5 BD3,
54 BD4, 54.5 BC19, 56.4 BD5, 63.6 A, 72.2 BD5, 99.2 BC20, 99.6 BD6,
100.3 BD7, 102.5 A, 112 BD8, 112.2 BC21, 120.9 BD2, 121.9 BC22, 125.5 BD9,
133.4 BD10, 151 BC23, 163 BC24, 164.7 BD9, 174.5 BC25, 177.4 BD10,
191.6 BC26, 192.7 BD11, 213 A, 244.8 A, 249 BD12, 250.8 A, 260.1 BD1,
263.5 BD8, 273.1 A, 274.7 BD6, 282.8 BC27, 285 BD13, 304 BD9, 315.4 BD4,
317.1 A, 320.6 A, 324.5 BD12, 324.9 BD10, 342 BD5, 350.2 BD3, 355.2 BC28,
364.6 BD10, 364.9 A, 366.3 BD2, 373 BD8, 379.4 BD14, 389 BD15, 394.9 A,
395.2 BD16
"""
# The BD modes' fix-effectiveness factors, BD1..BD16.
HDBK_FEF = [0.67, 0.72, 0.77, 0.77, 0.87, 0.92, 0.5, 0.85, 0.89, 0.74, 0.7]
HDBK_FEF += [0.63, 0.64, 0.72, 0.69, 0.46]


def _handbook_example(fefs):
    rows = [row.split() for row in HDBK.split(",")]
    x = [float(t) for t, _ in rows] + [400.0]
    modes = [m for _, m in rows] + [None]
    c = [0] * len(rows) + [1]
    fef = {"BD{}".format(j + 1): d for j, d in enumerate(fefs)}
    bc = ["BC{}".format(j) for j in range(17, 29)]
    return CrowAMSAA.projection(x, modes, fef, c=c, bc=bc)


def test_730_handbook_test_fix_find_test_example():
    # The published results: the Crow-AMSAA fit to all 56 failures has
    # beta 0.91026 and lambda 0.23969 (0.91026 is 55/56 of the MLE,
    # 0.9268), demonstrating 0.12744 (MTBF 7.84708); the BD modes' first
    # occurrences give beta 0.7970, unbiased ((K - 1) / K) 0.7472, and
    # lambda 0.1820; the growth potential intensity is 0.0670 and the
    # projection 0.0885, an MTBF of 11.29418.
    result = _handbook_example(HDBK_FEF)
    assert result.failures == {"A": 10, "BC": 14, "BD": 32}
    assert len(result.modes) == 16
    beta_mle = float(result.model.params[1])
    assert beta_mle == pytest.approx(0.9268, abs=1e-4)
    beta_bar = 55 / 56 * beta_mle
    assert beta_bar == pytest.approx(0.91026, abs=1e-5)
    assert 56 / 400**beta_bar == pytest.approx(0.23969, abs=1e-5)
    assert result.demonstrated_intensity == pytest.approx(0.12744, abs=1e-5)
    assert result.demonstrated_mtbf == pytest.approx(7.84708, abs=1e-5)
    assert result.beta * 16 / 15 == pytest.approx(0.7970, abs=1e-4)
    assert result.beta == pytest.approx(0.7472, abs=1e-4)
    assert 16 / 400**result.beta == pytest.approx(0.1820, abs=1e-4)
    assert result.mean_fef == pytest.approx(0.72125)
    assert result.growth_potential_intensity == pytest.approx(0.0670, abs=1e-4)
    assert result.projected_intensity == pytest.approx(0.0885, abs=1e-4)
    assert result.projected_mtbf == pytest.approx(11.29418, abs=1e-5)


def test_730_weibull_example_with_one_decimal_factors():
    # Weibull++'s version of the example (and ReliaSoft's Hotwire 36): the
    # same data with the factors to one decimal (0.7, 0.7, 0.8, 0.8, 0.9,
    # 0.9, 0.5, 0.9, 0.9, ...; mean 0.725) projects an MTBF of 11.3182
    # with a growth potential of 14.9957. The difference from 11.29418 is
    # the factors, not the choice of beta.
    fefs = [0.7, 0.7, 0.8, 0.8, 0.9, 0.9, 0.5, 0.9, 0.9, 0.7, 0.7, 0.6]
    fefs += [0.6, 0.7, 0.7, 0.5]
    result = _handbook_example(fefs)
    assert result.mean_fef == pytest.approx(0.725)
    assert result.demonstrated_mtbf == pytest.approx(7.8471, abs=1e-4)
    assert result.projected_mtbf == pytest.approx(11.3182, abs=1e-4)
    assert result.growth_potential_mtbf == pytest.approx(14.9957, abs=1e-4)


def test_730_bc_modes_need_two_failures():
    with pytest.raises(ValueError, match="at least 2 failures"):
        CrowAMSAA.projection(
            [15.0, 400.0], ["a1", None], {}, c=[0, 1], bc=["a1"]
        )
