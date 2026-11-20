"""``RayleighDefects``, ``phases_to_data`` and ``expected_events``.

The model's functions against closed forms worked in 50-digit decimal
arithmetic, the likelihood of phase counts against a sum written out,
the fit against its own likelihood equations, recovery from simulated
phase counts (seeds fixed before the results were seen), the refusal of
counts that have not turned down, and the phase helper.

No published phase table is fitted: none could be read (Kan's and
Putnam's are in books). See DECISIONS.md.
"""

from decimal import Decimal, getcontext

import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.special import gammaln

import surpyval as sp
from surpyval.recurrent import (
    GoelOkumoto,
    RayleighDefects,
    YamadaDelayedS,
    phases_to_data,
)
from surpyval.recurrent.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.recurrent.parametric.software_growth import limit_centroids

getcontext().prec = 50

K, C = 200.0, 2.5
TIMES = [
    1e-7, 1e-4, 0.01, 0.1, 0.5, 1.0, 1.7, 2.5, 3.0, 4.0,
    5.0, 6.5, 8.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 90.0,
]  # fmt: skip
COUNTS = [18, 44, 52, 41, 24]


def test_functions_match_the_closed_forms():
    k, c = Decimal(repr(K)), Decimal(repr(C))
    cif, iif = [], []
    for value in TIMES:
        t = Decimal(repr(value))
        decay = (-(t * t) / (2 * c * c)).exp()
        cif.append(k * (1 - decay))
        iif.append(k * t / (c * c) * decay)
    t = np.array(TIMES)
    np.testing.assert_allclose(
        RayleighDefects.cif(t, K, C), [float(v) for v in cif], rtol=4e-15
    )
    np.testing.assert_allclose(
        RayleighDefects.iif(t, K, C), [float(v) for v in iif], rtol=4e-14
    )
    np.testing.assert_allclose(
        RayleighDefects.log_iif(t, K, C),
        [float(v.ln()) for v in iif],
        # (a sum of terms of size 5 to 650: the log is right to their
        # rounding, a few 1e-16 of the largest)
        rtol=4e-15,
        atol=1e-14,
    )
    # the peak of the intensity is at c, where 1 - exp(-1/2) are found
    around = RayleighDefects.iif(np.array([C - 0.1, C, C + 0.1]), K, C)
    assert around[1] > around[0] and around[1] > around[2]
    assert RayleighDefects.cif(C, K, C) == pytest.approx(
        K * (1 - np.exp(-0.5)), rel=1e-15
    )


def test_inverse_and_the_total():
    t = np.array([v for v in TIMES if v <= 15.0])
    back = RayleighDefects.inv_cif(RayleighDefects.cif(t, K, C), K, C)
    np.testing.assert_allclose(back[t <= 10], t[t <= 10], rtol=1e-9)
    np.testing.assert_allclose(back, t, rtol=1e-4)
    model = RayleighDefects.from_params([K, C])
    assert RayleighDefects.total_parameter == "K"
    assert model.cif(np.inf) == K and model.iif(np.inf) == 0.0
    assert np.all(np.isinf(model.inv_cif([K, 2 * K])))
    # (the total less the count so far: exact to the rounding of K)
    np.testing.assert_allclose(
        model.remaining_faults(t),
        K * np.exp(-0.5 * (t / C) ** 2),
        rtol=1e-12,
        atol=1e-13 * K,
    )
    np.testing.assert_allclose(
        model.expected_events(2.0, 3.0),
        K * (np.exp(-0.5 * (2 / C) ** 2) - np.exp(-0.5 * (3 / C) ** 2)),
        rtol=1e-12,
    )


def test_likelihood_of_phase_counts_is_the_sum_written_out():
    data = phases_to_data(COUNTS)
    events = sp.handle_xicn(**data, as_recurrent_data=True)
    for params in ([K, C], [150.0, 1.9]):
        mean = RayleighDefects.cif(
            data["x"][:, 1], *params
        ) - RayleighDefects.cif(data["x"][:, 0], *params)
        direct = float(
            np.sum(
                np.array(COUNTS) * np.log(mean)
                - mean
                - gammaln(np.array(COUNTS) + 1.0)
            )
        )
        seen = -RayleighDefects.create_negll_func(events)(np.array(params))
        assert seen == pytest.approx(direct, rel=1e-13)


def test_fit_solves_its_likelihood_equations():
    # For counts n_j over (j - 1, j] to phase J, with F(t) = 1 -
    # exp(-s t**2), s = 1 / (2 c**2): K = sum(n) / F(J), and s is the
    # root of the profile's slope, sum(n_j d log(F(j) - F(j - 1)) / ds)
    # - sum(n) d log F(J) / ds.
    n = np.array(COUNTS, dtype=float)
    lo, hi = np.arange(5.0), np.arange(1.0, 6.0)

    def slope(s):
        gained = np.exp(-s * lo**2) - np.exp(-s * hi**2)
        d_gained = hi**2 * np.exp(-s * hi**2) - lo**2 * np.exp(-s * lo**2)
        whole = -np.expm1(-s * 25.0)
        return (
            np.sum(n * d_gained / gained)
            - n.sum() * 25.0 * np.exp(-s * 25.0) / whole
        )

    s = brentq(slope, 1e-4, 5.0, xtol=1e-15, rtol=1e-15)
    c = np.sqrt(1 / (2 * s))
    total = n.sum() / -np.expm1(-s * 25.0)
    model = RayleighDefects.fit(**phases_to_data(COUNTS))
    assert model.maximum == "verified"
    gap = np.abs(model.params - [total, c])
    assert np.all(gap < 1e-3 * model.standard_errors()), gap
    np.testing.assert_allclose(model.params, [total, c], rtol=2e-4)


@pytest.mark.parametrize("projects", [1, 5, 25])
def test_recovery_from_simulated_phase_counts(projects):
    # Poisson counts of six phases from K = 200, c = 2.5 (seed fixed
    # before the run), pooled over the projects: each estimate within 3
    # standard errors of the truth.
    rng = np.random.default_rng(20261120)
    edges = np.arange(0.0, 7.0)
    mean = projects * np.diff(RayleighDefects.cif(edges, K, C))
    counts = rng.poisson(mean)
    model = RayleighDefects.fit(**phases_to_data(counts))
    z = (model.params - [projects * K, C]) / model.standard_errors()
    print(projects, counts, model.params, z)
    assert np.all(np.abs(z) < 3.0), z


def test_recovery_from_defect_times():
    truth = RayleighDefects.from_params([K, C])
    d = truth.time_terminated_simulation_data(6.0, random_state=20261120)
    model = RayleighDefects.fit(d.x, c=d.c)
    z = (model.params - [K, C]) / model.standard_errors()
    assert np.all(np.abs(z) < 3.0), z


def test_predictions_of_later_phases_and_the_field():
    model = RayleighDefects.fit(**phases_to_data(COUNTS))
    later = model.expected_events([5, 6], [6, 7])
    assert later[0] > later[1] > 0
    cb = model.expected_events_cb([5, 6], [6, 7])
    assert cb.shape == (2, 2)
    assert np.all(cb[:, 0] < later) and np.all(later < cb[:, 1])
    # the phases to come and the field add up to what is left now
    field = model.remaining_faults(7.0)
    assert later.sum() + field == pytest.approx(
        float(model.remaining_faults(5.0)), rel=1e-12
    )
    lower, upper = model.remaining_faults_cb(7.0)
    assert lower < field < upper
    assert model.expected_events_cb(5, 6, bound="upper") == pytest.approx(
        model.expected_events_cb(5, 6, alpha_ci=0.1)[1]
    )
    with pytest.raises(ValueError, match="end must be at or after start"):
        model.expected_events(6, 5)
    back = ParametricRecurrenceModel.from_dict(model.to_dict())
    assert back.dist is RayleighDefects
    np.testing.assert_allclose(
        back.expected_events_cb(5, 6), model.expected_events_cb(5, 6)
    )


def test_expected_events_is_for_every_poisson_process():
    model = GoelOkumoto.from_params([100.0, 0.05])
    np.testing.assert_allclose(
        model.expected_events([0, 20], [20, 40]),
        100 * (np.exp([-0.0, -1.0]) - np.exp([-1.0, -2.0])),
        rtol=1e-13,
    )
    with pytest.raises(ValueError):
        model.expected_events_cb(0, 20)  # no fit, no bounds


def test_counts_that_have_not_turned_down_are_refused():
    # Counts in proportion to 2 j - 1 are those of an intensity rising
    # in proportion to time (the area of t under each phase): the edge.
    with pytest.raises(ValueError) as error:
        RayleighDefects.fit(**phases_to_data([5, 15, 25, 35]))
    message = str(error.value)
    assert "show no reliability growth that the Rayleigh defects" in message
    assert "mean squared time" in message
    assert "an intensity that rises in proportion to time" in message
    assert "the total number of defects K has no estimate" in message
    # just under that, a fit; just over, none
    assert (
        RayleighDefects.fit(**phases_to_data([5, 15, 25, 34])).params[0] > 80
    )
    with pytest.raises(ValueError, match="no reliability growth"):
        RayleighDefects.fit(**phases_to_data([5, 15, 25, 36]))


def test_the_rule_is_where_the_profile_likelihood_turns():
    for last, finite in ((34, True), (36, False)):
        data = phases_to_data([5, 15, 25, last])
        events = sp.handle_xicn(**data, as_recurrent_data=True)
        split = events.split_for_nhpp_likelihood()
        seen, limit = limit_centroids(split, 2, 2)
        assert (seen < limit) == finite
        # the mean of t**2 over (l, r] under an intensity proportional
        # to t is (r**4 - l**4) / (2 (r**2 - l**2))
        lo, hi = data["x"][:, 0], data["x"][:, 1]
        means = (hi**4 - lo**4) / (2 * (hi**2 - lo**2))
        counts = np.array([5, 15, 25, last])
        assert seen == pytest.approx(np.sum(counts * means) / counts.sum())
        assert limit == pytest.approx(4.0**4 / (2 * 4.0**2))
        negll = RayleighDefects.create_negll_func(events)

        def profile(c):
            total = counts.sum() / -np.expm1(-0.5 * (4.0 / c) ** 2)
            return -negll(np.array([total, c]))

        c = np.exp(np.linspace(np.log(1.0), np.log(1e3), 400))
        values = np.array([profile(v) for v in c])
        if finite:
            assert 0 < values.argmax() < 399
        else:
            assert np.all(np.diff(values[100:]) > 0)


def test_yamada_still_compares_plain_mean_times():
    # (the statistic gained a power; the other models keep theirs)
    assert YamadaDelayedS._limit_moment == 1
    assert GoelOkumoto._limit_moment == 1
    assert RayleighDefects._limit_moment == 2


def test_phases_to_data():
    data = phases_to_data([18, 44, 52, 0])
    np.testing.assert_array_equal(data["x"], [[0, 1], [1, 2], [2, 3], [3, 4]])
    assert data["c"].tolist() == [2, 2, 2, 2]
    assert data["n"].tolist() == [18, 44, 52, 0]
    assert set(data) == {"x", "c", "n"}
    weeks = phases_to_data([18, 44, 52], lengths=[6, 10, 4])
    np.testing.assert_array_equal(weeks["x"], [[0, 6], [6, 16], [16, 20]])
    # phase lengths set the clock: the peak moves with them
    model = RayleighDefects.fit(**phases_to_data(COUNTS, lengths=[4] * 5))
    plain = RayleighDefects.fit(**phases_to_data(COUNTS))
    np.testing.assert_allclose(
        model.params, plain.params * [1.0, 4.0], rtol=1e-4
    )


def test_phases_to_data_refusals():
    with pytest.raises(ValueError) as error:
        phases_to_data([18, 44])
    assert "counts has 2 phase(s); a curve needs at least 3" in str(
        error.value
    )
    with pytest.raises(ValueError, match="whole numbers of at least 0"):
        phases_to_data([18, 44.5, 52])
    with pytest.raises(ValueError, match="whole numbers of at least 0"):
        phases_to_data([18, -1, 52])
    with pytest.raises(ValueError, match="whole numbers of at least 0"):
        phases_to_data(["many", 2, 3])
    with pytest.raises(ValueError, match="flat list"):
        phases_to_data([[1, 2, 3]])
    with pytest.raises(ValueError, match="one length for each of the 3"):
        phases_to_data([18, 44, 52], lengths=[6, 10])
    with pytest.raises(ValueError, match="each be above 0"):
        phases_to_data([18, 44, 52], lengths=[6, 0, 4])
