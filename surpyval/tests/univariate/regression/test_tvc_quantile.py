"""``qf_tvc``, the inverse of the cumulative hazard along a covariate path,
and ``Hf_tvc(..., start=)``, the cumulative hazard from an age."""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval import CovariatePath, StepSchedule

P = np.array([1e-6, 0.01, 0.3, 0.5, 0.9, 0.999999])
RAMP = CovariatePath.from_points(
    [0, 10, 50, 60, 100], [0.0, 1.0, 0.2, 1.5, 0.1]
)
STEPS = StepSchedule.from_changepoints([0, 6, 30], [[0.0], [1.0], [0.3]])


def _fit(name, seed=0):
    rng = np.random.default_rng(seed)
    Z = rng.uniform(0, 2, (300, 1))
    x = 100 * rng.weibull(2, 300) * np.exp(-0.5 * Z[:, 0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return getattr(sp, name).fit(x, Z)


FAMILIES = [
    "WeibullPH",
    "WeibullAFT",
    "WeibullPO",
    "LogNormalAFT",
    "WeibullAH",
    "ExponentialPH",
    "NormalPH",
    "GumbelAFT",
]


@pytest.mark.parametrize("name", FAMILIES)
@pytest.mark.parametrize("path", [RAMP, STEPS], ids=["ramp", "steps"])
@pytest.mark.parametrize("start", [None, 0.0, 7.0, 55.0])
def test_qf_tvc_inverts_hf_tvc(name, path, start):
    m = _fit(name)
    real_line = float(m.distribution.support[0]) < 0
    if real_line and start is None:
        # Survival from -inf needs the path from -inf (refused below).
        return
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        q = m.qf_tvc(P, path, start=start)
        H = m.Hf_tvc(q, path, start=start)
    target = -np.log1p(-P)
    # To the quadrature's accuracy on H (rtol 1e-10, solved to 4e-10).
    np.testing.assert_allclose(H, target, rtol=1e-9, atol=0)
    assert np.all(np.diff(q) > 0)
    if start is not None:
        assert np.all(q > start)
        np.testing.assert_allclose(
            m.sf_tvc(q, path, given=start), 1 - P, rtol=1e-8
        )


def test_qf_tvc_closed_form_exponential_ramp():
    # Exponential PH along Z(t) = 0.1 t: H(t) = lam (e^{0.1 b t} - 1) /
    # (0.1 b), so t(p) = log(1 + q 0.1 b / lam) / (0.1 b), q = -log(1-p);
    # from an age a, q is added to H(a).
    m = _fit("ExponentialPH")
    lam, beta = m.params
    ramp = CovariatePath.from_callable(lambda t: 0.1 * t)
    k = 0.1 * beta
    q = -np.log1p(-P)
    exact = np.log1p(q * k / lam) / k
    np.testing.assert_allclose(m.qf_tvc(P, ramp), exact, rtol=1e-9)
    a = 4.0
    H_a = lam * np.expm1(k * a) / k
    exact_a = np.log1p((q + H_a) * k / lam) / k
    np.testing.assert_allclose(m.qf_tvc(P, ramp, start=a), exact_a, rtol=1e-9)


def test_qf_tvc_constant_path_is_qf():
    for name in ["WeibullPH", "WeibullAFT", "LogNormalAFT", "WeibullPO"]:
        m = _fit(name)
        flat = CovariatePath.from_points([0], [0.7])
        np.testing.assert_allclose(
            m.qf_tvc(P, flat), m.qf(P, [[0.7]]), rtol=1e-9
        )


def test_qf_tvc_from_minus_inf_on_real_line():
    m = _fit("NormalPH")
    path = CovariatePath.from_callable(lambda t: 0.5 + 0.0 * t)
    np.testing.assert_allclose(m.qf_tvc(P, path), m.qf(P, [[0.5]]), rtol=1e-8)


def test_qf_tvc_real_line_path_from_zero_is_refused():
    m = _fit("GumbelAFT")
    with pytest.raises(ValueError, match="where this path starts"):
        m.qf_tvc([1e-6], RAMP)
    m = _fit("NormalPH")
    with pytest.raises(ValueError, match="back to -inf"):
        m.qf_tvc([0.5], RAMP)


def test_qf_tvc_ends_and_missing():
    m = _fit("WeibullPH")
    out = m.qf_tvc([0.0, 1.0, np.nan, 0.5], RAMP, start=3.0)
    assert out[0] == 3.0 and out[1] == np.inf and np.isnan(out[2])
    assert out[3] > 3.0
    assert m.qf_tvc(0.0, RAMP) == 0.0
    assert np.isnan(m.qf_tvc([0.5], RAMP, start=np.nan)).all()
    with pytest.raises(ValueError, match=r"in \[0, 1\]"):
        m.qf_tvc([1.5], RAMP)


def test_qf_tvc_keeps_query_shape():
    m = _fit("WeibullPH")
    out = m.qf_tvc(np.full((2, 3), 0.4), RAMP)
    assert out.shape == (2, 3)
    assert np.ndim(m.qf_tvc(0.4, RAMP)) == 0


def test_qf_tvc_each_value_on_its_own():
    # A value does not depend on the others asked for with it.
    m = _fit("WeibullAFT")
    many = m.qf_tvc(np.linspace(0.01, 0.99, 200), RAMP, start=5.0)
    one = m.qf_tvc(np.linspace(0.01, 0.99, 200)[[17]], RAMP, start=5.0)
    assert one[0] == pytest.approx(many[17], rel=1e-9)


def test_qf_tvc_limited_failure_is_inf():
    # A hazard that dies away: an exponential PH model whose stress is
    # driven down so fast that the integral of the hazard converges, to
    # H(inf) = lam / c for phi(t) = exp(-c t), so a fraction exp(-lam/c)
    # never fails.
    m = _fit("ExponentialPH")
    lam, beta = m.params
    c = 0.05
    path = CovariatePath.from_callable(lambda t: -c * t / beta)
    H_inf = lam / c
    p_cap = -np.expm1(-H_inf)
    p = np.array([0.5 * p_cap, 0.999 * p_cap, min(1.0, 1.001 * p_cap)])
    out = m.qf_tvc(p[p < 1], path)
    exact = -np.log1p(-(-np.log1p(-p[:2])) * c / lam) / c
    np.testing.assert_allclose(out[:2], exact, rtol=1e-8)
    if p[2] < 1:
        assert out[2] == np.inf


def test_qf_tvc_draws_lives_along_a_path():
    # Inverse transform: the draws' empirical survival matches sf_tvc.
    m = _fit("WeibullPH")
    u = np.random.default_rng(3).uniform(size=4000)
    lives = m.qf_tvc(u, RAMP, start=7.0)
    t = np.array([20.0, 50.0, 80.0])
    emp = (lives[:, None] > t[None, :]).mean(axis=0)
    np.testing.assert_allclose(emp, m.sf_tvc(t, RAMP, given=7.0), atol=0.025)


@pytest.mark.parametrize("name", ["WeibullPH", "WeibullAFT", "WeibullAH"])
def test_hf_tvc_start_is_minus_log_conditional_survival(name):
    m = _fit(name)
    x = np.array([2.0, 7.0, 9.0, 40.0, 120.0])
    for path in (RAMP, STEPS):
        H = m.Hf_tvc(x, path, start=7.0)
        np.testing.assert_allclose(
            H, -np.log(m.sf_tvc(x, path, given=7.0)), rtol=1e-12, atol=0
        )
        assert H[0] == 0.0 and H[1] == 0.0
        full = m.Hf_tvc(x, path)
        np.testing.assert_allclose(
            H[2:], full[2:] - m.Hf_tvc(7.0, path), rtol=1e-8
        )


def test_hf_tvc_start_needs_path_only_from_start_for_ph():
    m = _fit("WeibullPH")
    late = CovariatePath.from_points([5, 10], [0.0, 1.0])
    H = m.Hf_tvc([20.0], late, start=6.0)
    assert np.isfinite(H).all() and H[0] > 0
    assert m.qf_tvc(0.5, late, start=6.0) > 6.0
