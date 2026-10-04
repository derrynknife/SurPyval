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
    # demonstrated intensity is the Crow-AMSAA one at T, and the BD
    # modes' intensities are taken off it.
    result = CrowAMSAA.projection(X, MODES, FEF, c=C, bc=["a2"])
    model = CrowAMSAA.fit(X, c=C)
    lam_ca = float(model.iif(400.0))
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
