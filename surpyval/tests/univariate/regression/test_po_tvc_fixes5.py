r"""
Time-varying-covariate evaluation for proportional odds (#236).

The PO hazard at time ``x`` under a covariate held at ``z`` is

.. math::
    h(x \mid z) = \frac{h_0(x)}{F_0(x) + \phi(z)\,S_0(x)},
    \qquad \phi(z) = e^{\beta' z},

a function of ``x`` and the *current* covariate only. Integrating it along a
step path therefore splits into one constant-``z`` integral per segment, and
each of those is the difference of the closed-form constant-covariate
cumulative hazard ``H(x | z) = -log S(x | z)``: PO telescopes exactly like PH.
These tests check that against references that do not use the PO ``Hf`` at
all (a quadrature of the hazard, and the survival ratio built from the
baseline), and that ``sf_tvc`` behaves like the other families' version.
"""

import numpy as np
import pytest
from scipy.integrate import quad

from surpyval import AH, PH, PO, Gumbel, Logistic, Normal, Weibull, WeibullPO
from surpyval.univariate.regression import StepSchedule


def _fit_po(seed=0, n=300):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 1))
    x = np.abs(rng.weibull(1.6, n) * 10 * np.exp(0.4 * Z[:, 0])) + 0.5
    return WeibullPO.fit(x=x, Z=Z)


def _fit_real_line(F, seed=1, n=200):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 1))
    x = rng.logistic(10, 2, n) + Z[:, 0]
    return F.fit(x, Z)


XL = np.array([0.0, 4.0, 9.0])
ZS = np.array([[0.2], [1.5], [-0.8]])


def _z_at(t):
    return ZS[np.searchsorted(XL, t, side="right") - 1]


def _quad_H(m, x):
    """Integrate the PO hazard ``hf`` along the path, segment by segment."""
    edges = np.concatenate([XL[XL < x], [x]])
    total = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        z = _z_at(a).reshape(1, -1)

        def h(u, z=z):
            return float(
                np.asarray(m.model.hf(np.array([u]), z, *m.params)).ravel()[0]
            )

        total += quad(h, a, b, epsabs=1e-13, epsrel=1e-12)[0]
    return total


def test_po_constant_schedule_reduces_to_sf():
    m = _fit_po()
    x = np.array([0.5, 2.0, 5.0, 9.0, 30.0])
    for z in (-1.0, 0.0, 0.7, 2.5):
        a = m.sf_tvc(x, StepSchedule.constant([z]))
        b = np.asarray(m.sf(x, np.array([[z]])), dtype=float).ravel()
        np.testing.assert_allclose(a, b, rtol=1e-13, atol=0)


def test_po_array_form_matches_schedule():
    m = _fit_po()
    x = [3.0, 6.0, 12.0]
    a = m.sf_tvc(x, ZS, xl=XL)
    b = m.sf_tvc(x, StepSchedule.from_changepoints(XL, ZS))
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(
        m.Hf_tvc(x, ZS, xl=XL),
        m.Hf_tvc(x, StepSchedule.from_changepoints(XL, ZS)),
    )


@pytest.mark.parametrize("x", [2.0, 4.0, 6.5, 9.0, 14.0, 40.0])
def test_po_hf_tvc_matches_quadrature_of_the_hazard(x):
    # Independent reference: numerically integrate h(u | z(u)) from 0 to x.
    m = _fit_po()
    got = m.Hf_tvc([x], ZS, xl=XL)[0]
    np.testing.assert_allclose(got, _quad_H(m, x), rtol=1e-9)


def test_po_sf_tvc_matches_baseline_survival_ratio():
    # Independent reference built from the Weibull baseline alone: on each
    # segment the survival is multiplied by S(b | z) / S(a | z), with
    # S(t | z) = phi S0 / (F0 + phi S0) and phi = exp(beta z).
    m = _fit_po()
    alpha, beta_shape = m.params[: m.k_dist]
    beta = m.params[m.k_dist]

    def S(t, z):
        S0 = np.exp(-((t / alpha) ** beta_shape))
        phi = np.exp(beta * z)
        return phi * S0 / (1 - S0 + phi * S0)

    x = 11.0
    manual = (
        S(4.0, 0.2) * (S(9.0, 1.5) / S(4.0, 1.5)) * (S(x, -0.8) / S(9.0, -0.8))
    )
    np.testing.assert_allclose(m.sf_tvc([x], ZS, xl=XL)[0], manual, 1e-12)


def test_po_hazard_follows_the_current_covariate():
    # The slope of Hf_tvc just after a change-point is the PO hazard at the
    # *new* covariate value: nothing about the earlier path carries over.
    m = _fit_po()
    t, eps = 6.0, 1e-5
    H = m.Hf_tvc([t - eps, t + eps], ZS, xl=XL)
    slope = (H[1] - H[0]) / (2 * eps)
    hf_new = float(np.asarray(m.hf([t], np.array([[1.5]]))).ravel()[0])
    np.testing.assert_allclose(slope, hf_new, rtol=1e-6)


def test_po_vectorised_x_matches_pointwise():
    m = _fit_po()
    x = np.array([12.0, 0.3, 4.0, 7.5, 9.0, 25.0])
    vec = m.sf_tvc(x, ZS, xl=XL)
    one = np.array([m.sf_tvc([v], ZS, xl=XL)[0] for v in x])
    np.testing.assert_allclose(vec, one, rtol=1e-14)
    assert vec.shape == x.shape


def test_po_conditional_survival():
    m = _fit_po()
    sched = StepSchedule.from_changepoints(XL, ZS)
    cond = m.sf_tvc([6.0, 12.0], sched, given=3.0)
    manual = m.sf_tvc([6.0, 12.0], sched) / m.sf_tvc([3.0], sched)
    np.testing.assert_allclose(cond, manual, rtol=1e-12)


def test_po_sf_tvc_monotone_and_bounded():
    m = _fit_po()
    sched = StepSchedule.cyclic([0, 8], [[-1.0], [2.0]], 24)
    s = m.sf_tvc(np.linspace(0.1, 100, 200), sched)
    assert np.all((s >= 0) & (s <= 1))
    assert np.all(np.diff(s) <= 1e-15)


def test_po_sf_tvc_far_tail_is_zero_not_nan():
    # A change-point beyond where the baseline survival underflows: computed
    # as -log(sf), Hf is inf at both ends of that segment and inf - inf = nan
    # poisoned every query time, including x = 5 long before it.
    m = _fit_po()
    xl = np.concatenate([XL, [1500.0]])
    Z = np.vstack([ZS, [[1.0]]])
    x = np.array([5.0, 1000.0, 2000.0])
    s = m.sf_tvc(x, Z, xl=xl)
    np.testing.assert_allclose(s[0], m.sf_tvc([5.0], ZS, xl=XL)[0], 1e-14)
    assert s[1] == 0.0 and s[2] == 0.0
    H = m.Hf_tvc(x, Z, xl=xl)
    assert np.all(np.isfinite(H)) and np.all(np.diff(H) > 0)


def test_po_errors_match_the_other_families():
    m = _fit_po()
    with pytest.raises(ValueError, match="covariate"):
        m.sf_tvc([2.0], StepSchedule.constant([0.5, 0.5]))
    with pytest.raises(ValueError, match="xl"):
        m.sf_tvc([2.0], ZS)
    with pytest.raises(ValueError, match="xl must not be given"):
        m.sf_tvc([2.0], StepSchedule.constant([0.5]), xl=[0.0])
    # Time 0 is a valid query (#435 item 4): sf(0, Z) = 1 for a Weibull.
    assert m.sf_tvc([0.0], StepSchedule.constant([0.5])) == 1.0


def test_po_generic_factory_supports_tvc():
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(150, 1))
    x = Weibull.random(150, 10, 2) * np.exp(0.3 * Z[:, 0])
    m = PO(Weibull).fit(x, Z)
    np.testing.assert_allclose(
        m.sf_tvc([3.0, 8.0], StepSchedule.constant([0.4])),
        np.asarray(m.sf([3.0, 8.0], [[0.4]])).ravel(),
        rtol=1e-13,
    )


# -- baselines supported below zero ---------------------------------------
#
# The path starts at 0, but a Logistic / Normal / Gumbel baseline puts
# survival probability below 1 at 0. The first segment is held back to the
# bottom of the support, so a constant path still gives the ordinary sf(x, Z)
# rather than the conditional S(x | z) / S(0 | z) it gave before.


@pytest.mark.parametrize(
    "F", [PO(Logistic), PH(Normal), AH(Normal), PH(Gumbel)]
)
def test_real_line_baseline_constant_path_reduces_to_sf(F):
    m = _fit_real_line(F)
    x = np.array([5.0, 10.0, 14.0])
    a = m.sf_tvc(x, StepSchedule.constant([0.5]))
    b = np.asarray(m.sf(x, [[0.5]]), dtype=float).ravel()
    np.testing.assert_allclose(a, b, rtol=1e-10)


def test_real_line_baseline_switch_is_consistent_with_sf():
    # Before the first change-point the path is one constant segment, so it
    # must match sf there; after it, the telescoped increment is added.
    m = _fit_real_line(PO(Logistic))
    xl, Z = [0.0, 8.0], [[0.0], [1.0]]
    s_before = m.sf_tvc([6.0], Z, xl=xl)[0]
    np.testing.assert_allclose(
        s_before, float(np.asarray(m.sf([6.0], [[0.0]])).ravel()[0]), 1e-10
    )
    after = m.sf_tvc([12.0], Z, xl=xl)[0]
    S = lambda t, z: float(np.asarray(m.sf([t], [[z]])).ravel()[0])  # noqa
    manual = S(8.0, 0.0) * S(12.0, 1.0) / S(8.0, 1.0)
    np.testing.assert_allclose(after, manual, rtol=1e-10)


def test_real_line_baseline_negative_times_use_the_first_segment():
    m = _fit_real_line(PO(Logistic))
    x = np.array([-3.0, 0.0, 6.0])
    got = m.sf_tvc(x, [[0.4], [1.0]], xl=[0.0, 8.0])
    want = np.asarray(m.sf(x, [[0.4]]), dtype=float).ravel()
    np.testing.assert_allclose(got, want, rtol=1e-10)


def test_po_hf_is_finite_where_the_baseline_survival_underflows():
    m = _fit_po()
    H = np.asarray(m.Hf([5.0, 3000.0], [[0.5]]), dtype=float).ravel()
    assert np.all(np.isfinite(H))
    np.testing.assert_allclose(
        H[0], -np.log(np.asarray(m.sf([5.0], [[0.5]])).ravel()[0]), 1e-12
    )
