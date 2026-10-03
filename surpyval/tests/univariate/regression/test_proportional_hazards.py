import itertools
import time
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.special import logsumexp

import surpyval as sp
from surpyval import CoxPH, ExponentialPH, Weibull, WeibullPH
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    CompetingRisksProportionalHazards,
    FineGray,
)
from surpyval.univariate.regression.proportional_hazards.cox_ph import CoxPH_
from surpyval.utils import validate_coxph


@pytest.fixture(autouse=True)
def set_random_seed():
    np.random.seed(42)


def test_cox_ph_hospital():
    """
    'A single binary covariate' example from 'Proportional hazards model'
    Wikipedia page: https://en.wikipedia.org/wiki/Proportional_hazards_model
    """
    # X = Hospital (1=A or 2=B)
    # T = period of time measure before death in month
    # T=60 => end of 5 year study period reached before death (right-censored)
    # C = censoring (1=right-censored)
    X = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    T = [60, 32, 60, 60, 60, 4, 18, 60, 9, 31, 53, 17]
    C = [1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0]

    # Fit model
    model = CoxPH.fit(x=T, Z=X, c=C)

    # beta_0 should be 2.12
    assert pytest.approx(2.12, abs=0.01) == model.params[0]


def test_cox_ph_company_death():
    """
    'A single continuous covariate' example from 'Proportional hazards model'
    Wikipedia page: https://en.wikipedia.org/wiki/Proportional_hazards_model
    """
    # P_on_E = price-to-earnings ratio on their 1-year IPO anniversary
    # T = days between 1-year IPO anniversary and death (or end of study)
    # C = censoring (1=right-censored)
    P_on_E = [9.7, 12, 3, 5.3, 10.8, 6.3, 11.6, 10.3, 8, 4, 5.9, 8.3]
    T = [3730, 849, 450, 269, 6036, 774, 1025, 5210, 1404, 371, 1948, 1126]
    C = [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]

    # Fit model
    model = CoxPH.fit(x=T, Z=P_on_E, c=C)

    # beta_0 should be -0.34
    assert pytest.approx(-0.34, abs=0.01) == model.params[0]


def test_cox_ph_sim():
    """
    Generates samples randomly and tests convergence.
    """
    # Instantiate random number generator. Seeded so the tight
    # parameter-recovery tolerance below cannot flake (a bare default_rng()
    # ignores the module seed fixture and was non-deterministic).
    rng = np.random.default_rng(2)

    # Construct covariant (Z) matrix
    n_samples = 100
    n_covariants = 3
    Z = rng.normal(size=(n_samples, n_covariants))

    # Baseline hazard function (i.e. an exponential survival function)
    baseline_hazard_rate = 0.01

    # Covariant coefficients
    beta = [0.1, -0.5, 0.8]

    # Take 50 samples per covariant sample for adequate fitting
    # Have to repeat Z for this
    samples_per_covariant_sample = 50
    Z_repeated = np.repeat(Z, samples_per_covariant_sample, axis=0)

    # Fill x samples
    x = np.zeros(n_samples * samples_per_covariant_sample)
    for i, Z_i in enumerate(Z_repeated):
        Z_i_hazard_rate = baseline_hazard_rate * np.exp(np.dot(Z_i, beta))
        x[i] = rng.exponential(1 / Z_i_hazard_rate)

    # Fit model
    model = CoxPH.fit(x=x, Z=Z_repeated, c=[0] * len(x))

    # Parameters should be approximately equal to the beta vector
    assert pytest.approx(beta, abs=0.05) == model.params


def test_exponential_ph_sim():
    """
    Same as test_cox_ph_sim_example but for ExponentialPH, checking the
    baseline hazard rate is fitted correctly.
    """
    # Instantiate random number generator. Seeded so the tight
    # parameter-recovery tolerance below cannot flake (a bare default_rng()
    # ignores the module seed fixture and was non-deterministic).
    rng = np.random.default_rng(2)

    # Construct covariant (Z) matrix
    n_samples = 100
    n_covariants = 3
    Z = rng.normal(size=(n_samples, n_covariants))

    # Baseline hazard function (i.e. an exponential survival function)
    baseline_hazard_rate = 0.01

    # Covariant coefficients
    beta = [0.1, -0.5, 0.8]

    # Take 50 samples per covariant sample for adequate fitting
    # Have to repeat Z for this
    samples_per_covariant_sample = 50
    Z_repeated = np.repeat(Z, samples_per_covariant_sample, axis=0)

    # Fill x samples
    x = np.zeros(n_samples * samples_per_covariant_sample)
    for i, Z_i in enumerate(Z_repeated):
        Z_i_hazard_rate = baseline_hazard_rate * np.exp(np.dot(Z_i, beta))
        x[i] = rng.exponential(1 / Z_i_hazard_rate)

    # Fit model
    model = ExponentialPH.fit(x=x, Z=Z_repeated, c=[0] * len(x))

    # Parameters should be approximately equal to the baseline hazard + the
    # beta vector
    assert (
        pytest.approx([baseline_hazard_rate] + beta, abs=0.05) == model.params
    )


def test_weibull_ph_sim():
    """
    Same as test_cox_ph_sim_example but with a Weibull baseline hazard, and
    testing WeibullPH can get the Weibull and covariant parameters correct.
    """
    # Instantiate random number generator. Seeded so the tight
    # parameter-recovery tolerance below cannot flake (a bare default_rng()
    # ignores the module seed fixture and was non-deterministic).
    rng = np.random.default_rng(2)

    # Construct covariant (Z) matrix
    n_samples = 100
    n_covariants = 3
    Z = rng.normal(size=(n_samples, n_covariants))

    # Baseline hazard function (i.e. a Weibull survival function)
    # For numpy, alpha_w is 'lambda', and beta_w is 'a'
    alpha_w = 0.7
    beta_w = 1.3

    # Covariant coefficients
    beta = [0.1, -0.5, 0.8]

    # Take 50 samples per covariant sample for adequate fitting
    # Have to repeat Z for this
    samples_per_covariant_sample = 50
    Z_repeated = np.repeat(Z, samples_per_covariant_sample, axis=0)

    # Fill x samples
    x = np.zeros(n_samples * samples_per_covariant_sample)
    for i, Z_i in enumerate(Z_repeated):
        alpha_w_i = alpha_w * np.exp(-np.dot(Z_i, beta) / beta_w)
        x[i] = alpha_w_i * rng.weibull(beta_w)

    # Fit model
    model = WeibullPH.fit(x=x, Z=Z_repeated, c=[0] * len(x))

    # Parameters should be approximately equal to the Weibull params + the
    # beta vector
    assert pytest.approx([alpha_w, beta_w] + beta, abs=0.05) == model.params


def test_efron_breslow_differ_on_ties():
    # Efron and Breslow are different approximations for tied event times.
    # They should give different beta estimates when ties exist.
    x = np.array([1.0, 1.0, 2.0, 3.0, 3.0, 4.0, 5.0])
    c = np.array([0, 0, 0, 0, 0, 1, 0])
    Z = np.array([[0.5], [1.2], [0.3], [0.8], [1.5], [0.2], [1.1]])

    m_breslow = CoxPH.fit(x=x, Z=Z, c=c, tie_method="breslow")
    m_efron = CoxPH.fit(x=x, Z=Z, c=c, tie_method="efron")

    assert not np.allclose(m_breslow.beta, m_efron.beta)


def test_count_weights_equivalent_to_repeated():
    # Fitting with n=[2, 2, 2] must give the same beta as repeating each row.
    x = np.array([1.0, 2.5, 4.0])
    Z = np.array([[0.5], [1.2], [0.3]])
    c = np.array([0, 0, 0])

    m_rep = CoxPH.fit(
        x=np.repeat(x, 2),
        Z=np.repeat(Z, 2, axis=0),
        c=np.repeat(c, 2),
        tie_method="breslow",
    )
    m_cnt = CoxPH.fit(
        x=x, Z=Z, c=c, n=np.array([2, 2, 2]), tie_method="breslow"
    )

    assert np.allclose(m_rep.beta, m_cnt.beta, atol=1e-6)


def test_efron_returns_p_values():
    # Efron now returns its (corrected) analytic Hessian, so p_values are
    # produced -- finite probabilities in [0, 1], one per covariate.
    x = [1.0, 2.0, 3.0, 4.0, 5.0]
    Z = [[0.1], [0.5], [0.3], [0.8], [0.2]]
    c = [0, 0, 0, 0, 0]

    model = CoxPH.fit(x=x, Z=Z, c=c, tie_method="efron")

    assert model.p_values is not None
    assert model.p_values.shape == (1,)
    assert np.all(np.isfinite(model.p_values))
    assert np.all((model.p_values >= 0) & (model.p_values <= 1))


def test_efron_hessian_matches_finite_difference():
    # The analytic Efron information (the Hessian Newton-Raphson steps with)
    # must match a central-difference Jacobian of the score, including the
    # off-diagonal terms that the old inner-product bug corrupted. Uses a
    # multi-covariate design with tied event times.
    from surpyval.utils import validate_coxph

    rng = np.random.default_rng(0)
    N, p = 300, 3
    Z = rng.normal(size=(N, p))
    eta = np.exp(Z @ [0.5, 0.0, -0.4])
    t = np.round(-np.log(rng.uniform(size=N)) / eta, 1)
    c = (rng.uniform(size=N) < 0.25).astype(int)

    x, cc, nn, tl, ZZ = validate_coxph(t, c, np.ones(N), Z, None, "efron")
    _, jac_hess = CoxPH.create_efron_ll_jac_hess(x, ZZ, cc, nn, tl)

    beta = np.array([0.1, -0.1, 0.2])
    H = jac_hess(beta)[1]
    eps = 1e-6
    H_num = np.zeros((p, p))
    for k in range(p):
        bp, bm = beta.copy(), beta.copy()
        bp[k] += eps
        bm[k] -= eps
        H_num[:, k] = (jac_hess(bp)[0] - jac_hess(bm)[0]) / (2 * eps)

    assert np.allclose(H, H_num, atol=1e-5)
    assert np.allclose(H, H.T)  # symmetric
    assert np.all(np.linalg.eigvalsh(H) > 0)  # positive definite


def test_efron_and_breslow_p_values_agree_without_ties():
    # With no tied event times Efron and Breslow reduce to the same partial
    # likelihood, so their standard errors (hence p-values) should agree.
    rng = np.random.default_rng(1)
    N = 2000
    Z = rng.normal(size=(N, 2))
    x = -np.log(rng.uniform(size=N)) / np.exp(Z @ [0.7, -0.5])
    c = (rng.uniform(size=N) < 0.2).astype(int)

    m_ef = CoxPH.fit(x=x, Z=Z, c=c, tie_method="efron")
    m_br = CoxPH.fit(x=x, Z=Z, c=c, tie_method="breslow")

    assert np.allclose(m_ef.beta, m_br.beta, atol=1e-3)
    assert np.allclose(m_ef.p_values, m_br.p_values, atol=1e-2)


def test_cox_delayed_entry_episode_split_invariance():
    # Splitting each subject's follow-up at an interior time, carrying the
    # SAME covariate on both pieces (the first piece censored, the second a
    # delayed entry), is an exact identity of the Cox partial likelihood: it
    # must not change the estimated coefficient. This is the foundation of
    # start-stop / time-varying-covariate fitting, and it exercises the
    # staggered-risk-set (delayed-entry) path, which MINPACK's root-finder
    # used to stall on before the minimisation fallback was added.
    rng = np.random.default_rng(1)
    n = 500
    Z = rng.normal(size=(n, 1))
    T = rng.exponential(1 / np.exp(Z[:, 0] * 0.7))
    c = np.zeros(n, dtype=int)

    ref = CoxPH.fit(x=T, Z=Z, c=c, tl=np.zeros(n))

    s = T * rng.uniform(0.2, 0.8, size=n)
    split = CoxPH.fit(
        x=np.concatenate([s, T]),
        Z=np.vstack([Z, Z]),
        c=np.concatenate([np.ones(n, dtype=int), np.zeros(n, dtype=int)]),
        tl=np.concatenate([np.zeros(n), s]),
    )

    assert split.res.success
    assert np.isclose(ref.beta[0], split.beta[0], atol=1e-3)


def test_cox_rejects_right_truncation():
    # The Cox partial likelihood has no way to incorporate right or interval
    # truncation, so a 2-D ``tl`` (a [tl, tr] pair) must raise a clear,
    # Cox-specific error rather than the generic truncation message.
    rng = np.random.default_rng(0)
    x = rng.exponential(10, 40)
    Z = rng.normal(size=(40, 1))
    c = np.zeros(40, dtype=int)
    t_pair = np.column_stack([np.zeros(40), x + 5.0])
    with pytest.raises(ValueError, match="left-truncation"):
        CoxPH.fit(x=x, Z=Z, c=c, tl=t_pair)


def test_baseline_hazard_properties():
    # h0 must be non-negative and H0 must be monotonically non-decreasing.
    n = 100
    x = np.random.exponential(10, n)
    Z = np.random.normal(0, 1, (n, 2))
    c = np.zeros(n, dtype=int)

    model = CoxPH.fit(x=x, Z=Z, c=c, tie_method="breslow")

    assert np.all(model.h0 >= 0)
    assert np.all(np.diff(model.H0) >= 0)
    assert model.H0.shape == model.x.shape


# ---------------------------------------------------------------------------
# Exact and Kalbfleisch-Prentice tie-handling methods (issue #142)
# ---------------------------------------------------------------------------


def test_tie_methods_agree_without_ties():
    # With every event time distinct there are no ties to break, so all four
    # tie-handling methods must return the identical coefficient.
    rng = np.random.default_rng(0)
    n = 60
    Z = rng.normal(size=(n, 1))
    x = np.sort(rng.uniform(1, 100, n)) + np.arange(n) * 1e-3
    c = np.zeros(n, dtype=int)

    betas = {
        m: CoxPH.fit(x=x, Z=Z, c=c, tie_method=m).beta[0]
        for m in ["breslow", "efron", "exact", "kalbfleisch-prentice"]
    }
    assert max(betas.values()) - min(betas.values()) < 1e-8


def _neg_ll(method, x, Z, c):
    from surpyval.univariate.regression.proportional_hazards.cox_ph import (
        CoxPH_,
    )

    tl = np.full(len(x), -np.inf)
    gen = CoxPH_()._resolve_func_generator(method)
    neg_ll, _ = gen(
        np.asarray(x, float),
        np.asarray(Z, float),
        np.asarray(c, int),
        np.ones(len(x)),
        tl,
    )
    return neg_ll


def test_kp_negll_matches_conditional_logistic():
    # Two deaths tie at t=1; the KP contribution there is the exact discrete
    # (conditional-logistic) term exp(b*(z0+z1)) / e_2, with e_2 the 2nd
    # elementary symmetric polynomial of the four risk scores.
    from itertools import combinations

    x = np.array([1.0, 1.0, 2.0, 3.0])
    Z = np.array([[0.5], [-0.5], [1.0], [0.0]])
    c = np.array([0, 0, 0, 1])
    zf = Z.ravel()
    neg_ll = _neg_ll("kp", x, Z, c)

    def brute(b):
        s = np.exp(b * zf)
        e2 = sum(s[i] * s[j] for i, j in combinations(range(4), 2))
        ll = b * (zf[0] + zf[1]) - np.log(e2)
        ll += b * zf[2] - np.log(s[2] + s[3])
        return -ll

    for b in (-0.7, 0.0, 0.3, 1.1):
        assert np.isclose(float(neg_ll(np.array([b]))), brute(b))


def test_exact_negll_matches_average_over_orderings():
    # Same tie set under the exact (average-over-orderings) method: the two
    # tied deaths could have occurred in either order, and the contribution is
    # the sequential Cox term averaged over both orderings.
    x = np.array([1.0, 1.0, 2.0, 3.0])
    Z = np.array([[0.5], [-0.5], [1.0], [0.0]])
    c = np.array([0, 0, 0, 1])
    zf = Z.ravel()
    neg_ll = _neg_ll("exact", x, Z, c)

    def brute(b):
        s = np.exp(b * zf)
        R = s.sum()
        a0, a1 = s[0], s[1]
        T = (1 / R) * (1 / (R - a0)) + (1 / R) * (1 / (R - a1))
        ll = np.log(a0 * a1) + np.log(T)
        ll += b * zf[2] - np.log(s[2] + s[3])
        return -ll

    for b in (-0.7, 0.0, 0.3, 1.1):
        assert np.isclose(float(neg_ll(np.array([b]))), brute(b))


def test_exact_and_kp_differ_on_ties():
    # On tied data the exact, KP, breslow and efron estimates should differ.
    rng = np.random.default_rng(5)
    n = 30
    Z = rng.normal(size=(n, 1))
    x = rng.integers(1, 11, size=n).astype(float)
    c = (rng.uniform(size=n) < 0.2).astype(int)

    betas = {
        m: CoxPH.fit(x=x, Z=Z, c=c, tie_method=m).beta
        for m in ["breslow", "efron", "exact", "kp"]
    }
    assert not np.allclose(betas["exact"], betas["efron"])
    assert not np.allclose(betas["kp"], betas["exact"])
    assert not np.allclose(betas["kp"], betas["breslow"])


@pytest.mark.parametrize("method", ["exact", "kp"])
def test_exact_kp_return_finite_p_values_and_pd_hessian(method):
    rng = np.random.default_rng(3)
    n = 28
    Z = rng.normal(size=(n, 1))
    x = rng.integers(1, 7, size=n).astype(float)
    c = (rng.uniform(size=n) < 0.2).astype(int)

    model = CoxPH.fit(x=x, Z=Z, c=c, tie_method=method)
    assert model.p_values is not None
    assert np.all(np.isfinite(model.p_values))
    assert np.all((model.p_values >= 0) & (model.p_values <= 1))

    H = model.jac(model.beta)[1]
    assert np.allclose(H, H.T)
    assert np.all(np.linalg.eigvalsh(H) > 0)


@pytest.mark.parametrize("method", ["exact", "kp"])
def test_exact_kp_hessian_matches_finite_difference(method):
    from surpyval.univariate.regression.proportional_hazards.cox_ph import (
        CoxPH_,
    )
    from surpyval.utils import validate_coxph

    rng = np.random.default_rng(9)
    n = 40
    Z = rng.normal(size=(n, 2))
    x = rng.integers(1, 10, size=n).astype(float)
    c = (rng.uniform(size=n) < 0.25).astype(int)
    x, c, nn, tl, ZZ = validate_coxph(x, c, None, Z, None, method)

    gen = CoxPH_()._resolve_func_generator(method)
    neg_ll, jac_hess = gen(x, ZZ, c, nn, tl)

    b = np.array([0.3, -0.2])
    score, H = jac_hess(b)
    eps = 1e-6
    eye = np.eye(2)
    g_num = np.array(
        [
            (float(neg_ll(b + eps * eye[i])) - float(neg_ll(b - eps * eye[i])))
            / (2 * eps)
            for i in range(2)
        ]
    )
    H_num = np.zeros((2, 2))
    for i in range(2):
        sp = jac_hess(b + eps * eye[i])[0]
        sm = jac_hess(b - eps * eye[i])[0]
        H_num[:, i] = (sp - sm) / (2 * eps)

    assert np.allclose(score, g_num, atol=1e-5)
    assert np.allclose(H, H_num, atol=1e-4)


def test_kp_count_weights_equivalent_to_repeated_rows():
    # A count of two must be identical to two repeated rows for KP, since the
    # generator expands counts into individual tied observations.
    rng = np.random.default_rng(11)
    n = 24
    Z = rng.normal(size=(n, 1))
    x = rng.integers(1, 8, size=n).astype(float)
    c = (rng.uniform(size=n) < 0.2).astype(int)

    m_rep = CoxPH.fit(
        x=np.repeat(x, 2),
        Z=np.repeat(Z, 2, axis=0),
        c=np.repeat(c, 2),
        tie_method="kp",
    )
    m_cnt = CoxPH.fit(x=x, Z=Z, c=c, n=np.full(n, 2), tie_method="kp")
    assert np.allclose(m_rep.beta, m_cnt.beta, atol=1e-5)


def test_exact_handles_large_tie_sets():
    # The exact method used to be O(2^d) in the tie multiplicity d and
    # refused more than 12 ties; its integral form has no such limit. With
    # every observation dying at once the contribution is exactly one for
    # any beta, so the likelihood is flat.
    rng = np.random.default_rng(3)
    n = 80
    Z = rng.normal(size=(n, 1))
    x = np.ones(n)  # every observation ties at the same time
    c = np.zeros(n, dtype=int)
    neg_ll, _ = CoxPH.create_exact_ll_jac_hess(
        x, Z, c, np.ones(n), np.full(n, -np.inf)
    )
    assert neg_ll(np.array([0.7])) == pytest.approx(0.0, abs=1e-12)
    # A flat likelihood determines no coefficient: the fit says so, and
    # reports it as nan (#409, #476).
    with pytest.warns(UserWarning, match="partial likelihood does not"):
        model = CoxPH.fit(x=x, Z=Z, c=c, tie_method="exact")
    assert np.isnan(model.beta).all()


def test_kp_handles_heavy_ties():
    # KP uses the elementary-symmetric recursion, which is polynomial in the
    # tie size, so it fits heavily tied data the exact method would refuse.
    rng = np.random.default_rng(3)
    n = 40
    Z = rng.normal(size=(n, 1))
    x = rng.integers(1, 4, size=n).astype(float)
    c = (rng.uniform(size=n) < 0.2).astype(int)

    model = CoxPH.fit(x=x, Z=Z, c=c, tie_method="kp")
    assert np.all(np.isfinite(model.beta))
    assert np.all(np.isfinite(model.p_values))


def test_ph_fixed_covariate_coefficient_pins_correct_parameter():
    # ``fixed={"beta_0": v}`` must pin the first covariate coefficient, not
    # the first distribution parameter (#251: the phi param map was merged
    # without the k_dist offset, so beta_0 collided with alpha).
    np.random.seed(5)
    x = Weibull.random(200, 10, 3)
    Z = np.random.normal(size=(200, 2))

    free = WeibullPH.fit(x, Z=Z)
    fixed_beta0 = WeibullPH.fit(x, Z=Z, fixed={"beta_0": 0.5})

    assert fixed_beta0.params[2] == pytest.approx(0.5, abs=1e-12)
    # The distribution parameters must remain close to the free fit, not be
    # pinned to the fixed value.
    assert fixed_beta0.params[0] == pytest.approx(free.params[0], rel=0.2)

    # Fixing a distribution parameter by name still works.
    fixed_shape = WeibullPH.fit(x, Z=Z, fixed={"beta": 3.0})
    assert fixed_shape.params[1] == pytest.approx(3.0, abs=1e-12)


def _weibull_ph_ll(x, Z, c, alpha, shape, gamma):
    """Weibull PH log-likelihood, written out independently of the fitter
    so a test can score two candidate answers against each other."""
    lin = Z @ np.asarray(gamma)
    log_h = np.log(shape / alpha) + (shape - 1) * np.log(x / alpha) + lin
    log_sf = -((x / alpha) ** shape) * np.exp(lin)
    return log_h[np.asarray(c) == 0].sum() + log_sf.sum()


@pytest.mark.parametrize("scale", [1e-3, 1e0, 1e3, 1e6, 1e9])
def test_weibull_ph_fit_is_invariant_to_the_units_of_x(scale):
    # Multiplying every event time by k must move alpha by exactly k and
    # leave the shape and the covariate coefficients alone -- the model is
    # closed under a change of time units.
    #
    # It was not. The PH ladder ran BFGS on a finite-difference gradient
    # with no preconditioning, so scipy's absolute ``gtol`` was met well
    # short of the optimum once the data grew: at scale 1e6 the fit gave up
    # 1.5 nats of log-likelihood and landed ~1e-2 away in the coefficients
    # (#328).
    rng = np.random.default_rng(17)
    n, p = 800, 3
    Z = rng.normal(size=(n, p))
    beta = np.array([0.6, 0.1, -0.4])
    t = 4.0 * (-np.log(rng.random(n)) / np.exp(Z @ beta)) ** (1 / 1.5)
    cens = rng.exponential(np.quantile(t, 0.6), n)
    x, c = np.minimum(t, cens), (t > cens).astype(int)

    base = WeibullPH.fit(x=x, Z=Z, c=c)
    scaled = WeibullPH.fit(x=x * scale, Z=Z, c=c)

    assert scaled.params[0] == pytest.approx(base.params[0] * scale, rel=1e-4)
    assert scaled.params[1] == pytest.approx(base.params[1], rel=1e-4)
    assert scaled.params[2:] == pytest.approx(base.params[2:], abs=1e-4)


def test_weibull_ph_at_large_scale_reaches_the_optimum():
    # The invariance test above would also pass if the fit were equally
    # wrong at both scales, so anchor one end of it. Fit at unit scale,
    # where the old ladder was fine, then carry that answer over to the
    # large-scale data by the exact change of units. The large-scale fit
    # must score at least as well on the large-scale likelihood as the
    # transported one does -- it had the same optimum available to it.
    #
    # Under the old ladder it gave up 1.5 nats here.
    rng = np.random.default_rng(23)
    n, p = 800, 3
    Z = rng.normal(size=(n, p))
    beta = np.array([0.6, 0.1, -0.4])
    t = 4.0 * (-np.log(rng.random(n)) / np.exp(Z @ beta)) ** (1 / 1.5)
    cens = rng.exponential(np.quantile(t, 0.6), n)
    x, c = np.minimum(t, cens), (t > cens).astype(int)

    scale = 1e6
    small = np.asarray(WeibullPH.fit(x=x, Z=Z, c=c).params, dtype=float)
    large = np.asarray(
        WeibullPH.fit(x=x * scale, Z=Z, c=c).params, dtype=float
    )

    transported = _weibull_ph_ll(
        x * scale, Z, c, small[0] * scale, small[1], small[2:]
    )
    at_fit = _weibull_ph_ll(x * scale, Z, c, large[0], large[1], large[2:])

    assert at_fit >= transported - 1e-6


def test_optimise_ph_never_returns_a_worse_point_than_it_started_from():
    # A contract guard rather than a reproducer: the old ladder returned
    # TNC unconditionally, so a rung that could only ever be an improvement
    # was free to be a regression (#328). On these smooth objectives it
    # happened not to be, and this test passes either way -- it is here so
    # that a future rung added to the ladder cannot reintroduce the hazard
    # unnoticed. Whatever the rungs do individually, the ladder as a whole
    # must not hand back a point worse than its starting guess.
    from surpyval.univariate.regression._fit_skeleton import optimise_ph

    def rosenbrock(v):
        return (1 - v[0]) ** 2 + 100 * (v[1] - v[0] ** 2) ** 2

    for start in ([-1.2, 1.0], [3.0, -4.0], [0.0, 0.0]):
        x0 = np.array(start)
        res = optimise_ph(rosenbrock, x0)
        assert res.fun <= rosenbrock(x0)
        assert res.fun == pytest.approx(rosenbrock(res.x), rel=1e-8, abs=1e-12)


# The per-time ``efron_hess`` and its double-loop oracle are gone (#516):
# the information is now formed over the rows, and test_cox_newton.py
# checks it against a literal loop over the event times and tied deaths,
# fractional counts included.


@pytest.mark.parametrize(
    "counts", [[1.0, 1.0, 1.0], [5.0, 1.0, 2.0], [2.5, 0.0, 3.5]]
)
def test_efron_log_denominator_matches_the_loop_it_replaced(counts):
    from surpyval.univariate.regression.proportional_hazards.cox_ph import (
        efron_log_denominator,
    )

    rng = np.random.default_rng(37)
    m = len(counts)
    n_d = np.array(counts)
    Ri = rng.uniform(50, 100, (m, 1))
    Di = rng.uniform(0, 10, (m, 1))

    want = np.zeros(m)
    for i in range(m):
        if n_d[i] == 0:
            continue
        c_vals = np.arange(int(n_d[i])) / n_d[i]
        want[i] = np.log(Ri[i] - c_vals[:, None] * Di[i]).sum()

    got = efron_log_denominator(n_d, Ri, Di)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def _efron_jac_masked_reference(n_d, Ri, ZRi, Di, ZDi):
    """The ``numpy.ma`` implementation ``efron_jac`` replaced (#515), kept
    as an oracle: an ``(times x largest tie x p)`` masked array, masked
    where ``j >= d``, summed over the tie axis. Integer counts only (the
    fitters require integer ``n``); a time with no deaths is masked out,
    which is 0 in the score's sum."""
    import numpy.ma as ma

    n_d = n_d.reshape(-1, 1)
    arr = np.repeat([np.arange(int(n_d.max()))], len(n_d), axis=0)
    mask = 1 - (arr < n_d).astype(int)
    r = ma.array(arr, mask=mask) / n_d
    denom = np.expand_dims(Ri - Di * r, axis=-1)
    numer = np.expand_dims(ZRi, axis=1) - np.expand_dims(
        ZDi, axis=1
    ) * np.expand_dims(r, axis=-1)
    return (numer / denom).sum(axis=1).filled(0.0)


def _efron_risk_sums(counts, p=4, seed=43):
    rng = np.random.default_rng(seed)
    m = len(counts)
    Ri = rng.uniform(50, 100, (m, 1))
    Di = rng.uniform(0, 10, (m, 1))
    ZRi = rng.normal(size=(m, p))
    ZDi = rng.normal(size=(m, p))
    return np.array(counts, dtype=float), Ri, ZRi, Di, ZDi


@pytest.mark.parametrize(
    "counts",
    [
        [1.0, 1.0, 1.0, 1.0],  # no ties at all -- the continuous case
        [1.0, 0.0, 3.0, 1.0],  # a time with no deaths mixed in
        [7.0, 12.0, 1.0, 4.0],  # heavy ties
        [1.0] * 30 + [51.0] + [0.0] * 5,  # one large tie among many
        [2.0],  # a single event time
    ],
)
def test_efron_score_matches_the_masked_array_it_replaced(counts):
    from surpyval.univariate.regression.proportional_hazards.cox_ph import (
        efron_jac,
    )

    n_d, Ri, ZRi, Di, ZDi = _efron_risk_sums(counts)
    got = efron_jac(n_d, Ri, ZRi, Di, ZDi)
    want = _efron_jac_masked_reference(n_d, Ri, ZRi, Di, ZDi)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)
    # A time with one death is ZR / R, bit for bit as before.
    one = n_d == 1
    np.testing.assert_array_equal(got[one], want[one])


def test_efron_score_takes_int_d_terms_for_fractional_counts():
    # The convention of efron_log_denominator and the information
    # (_cox_information): range(int(d))
    # terms with c = j / d, so the score is the derivative of the same
    # log-likelihood.
    from surpyval.univariate.regression.proportional_hazards.cox_ph import (
        efron_jac,
    )

    n_d, Ri, ZRi, Di, ZDi = _efron_risk_sums([2.5, 1.0, 3.5, 0.0, 0.5])
    want = np.zeros_like(ZRi)
    for i, d in enumerate(n_d):
        for j in range(int(d)):
            k = j / d
            want[i] += (ZRi[i] - k * ZDi[i]) / (Ri[i] - k * Di[i])
    got = efron_jac(n_d, Ri, ZRi, Di, ZDi)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)


def test_efron_fit_builds_no_masked_array():
    # The masked (times x largest tie x p) array made one 51-way tie among
    # 30 000 times cost 12.9 s against 1.15 s untied (#515).
    from unittest import mock

    rng = np.random.default_rng(515)
    x = rng.exponential(1.0, 200)
    x[:51] = np.median(x)
    Z = rng.normal(size=(200, 2))
    with mock.patch("numpy.ma.array", side_effect=AssertionError("ma")):
        model = CoxPH.fit(x, Z, np.zeros(200), tie_method="efron")
    assert np.all(np.isfinite(model.beta))


def _tied_weighted_truncated_data():
    rng = np.random.default_rng(515)
    N = 120
    Z = rng.normal(size=(N, 2))
    x = np.round(rng.exponential(3 / np.exp(Z @ [0.5, -0.4])), 1) + 0.1
    c = (rng.random(N) < 0.25).astype(int)
    x[:15] = 1.0  # a 15-way tie
    c[:15] = 0
    n = rng.integers(1, 4, N)
    tl = np.where(
        rng.random(N) < 0.4, np.round(x * rng.uniform(0, 0.8, N), 1), 0.0
    )
    strata = rng.integers(0, 2, N)
    return x, Z, c, n, tl, strata


# beta, se and H0 at t = 0.5, 1, 3 computed with the masked-array Efron
# score (before #515).
_EFRON_BEFORE_515 = {
    "plain": (
        [0.5782139726212264, -0.2972629120131516],
        [0.12427888727863272, 0.1053039983259275],
        [0.16991079058102748, 0.4765249269989844, 0.9015016215346601],
    ),
    "weighted": (
        [0.6645107694788093, -0.23918750208182854],
        [0.09616762502048894, 0.07779164293811058],
        [0.17943686817959503, 0.4603489994662496, 0.8659552192449375],
    ),
    "truncated": (
        [0.5260180982815772, -0.10550167238643304],
        [0.12566664271497632, 0.11250267353133918],
        [0.2470686108848548, 0.6328945355305517, 1.145562987910888],
    ),
    "stratified": (
        [0.6635357859303955, -0.06413378628381788],
        [0.10045854267463622, 0.08172838981094684],
        [0.2823723979675468, 0.5797935356723387, 1.0742277199693688],
    ),
}


@pytest.mark.parametrize("case", sorted(_EFRON_BEFORE_515))
def test_efron_fit_unchanged_by_the_ragged_score(case):
    x, Z, c, n, tl, strata = _tied_weighted_truncated_data()
    kw = {
        "plain": {},
        "weighted": {"n": n},
        "truncated": {"tl": tl},
        "stratified": {"n": n, "tl": tl, "strata": strata},
    }[case]
    model = CoxPH.fit(x, Z, c, tie_method="efron", **kw)
    stratum = {"stratum": 1} if case == "stratified" else {}
    H = model.Hf([0.5, 1.0, 3.0], np.zeros((3, 2)), **stratum)
    beta, se, H0 = _EFRON_BEFORE_515[case]
    np.testing.assert_allclose(model.beta, beta, rtol=1e-12)
    np.testing.assert_allclose(model.standard_errors(), se, rtol=1e-12)
    np.testing.assert_allclose(H, H0, rtol=1e-12)


def test_cox_fit_is_unaffected_by_the_order_of_the_rows():
    # ``create_*_ll_jac_hess`` now sorts by event time so ``_GroupBy`` can
    # skip its permutation (#329). Nothing downstream may depend on that,
    # so a shuffled copy of the same data must fit identically.
    rng = np.random.default_rng(41)
    n, p = 400, 3
    Z = rng.normal(size=(n, p))
    t = 10 * (-np.log(rng.random(n)) / np.exp(Z @ [0.6, 0.1, -0.4])) ** (
        1 / 1.5
    )
    cens = rng.exponential(np.quantile(t, 0.6), n)
    x, c = np.minimum(t, cens), (t > cens).astype(int)
    tl = rng.uniform(0, 0.4 * np.median(x), n)
    keep = x > tl
    x, Z, c, tl = x[keep], Z[keep], c[keep], tl[keep]

    shuffle = rng.permutation(len(x))
    for method in ("efron", "breslow"):
        a = CoxPH.fit(x=x, Z=Z, c=c, tl=tl, tie_method=method)
        b = CoxPH.fit(
            x=x[shuffle],
            Z=Z[shuffle],
            c=c[shuffle],
            tl=tl[shuffle],
            tie_method=method,
        )
        np.testing.assert_allclose(a.beta, b.beta, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(
            a.jac(a.beta)[1], b.jac(b.beta)[1], rtol=1e-9, atol=1e-10
        )


# ---------------------------------------------------------------------------
# #271: parametric PH ``random()`` samples the model's own
# distribution and supports multi-covariate models.
# ---------------------------------------------------------------------------


class TestPHRandom:
    def test_random_matches_model_sf(self):
        # 271: draws must come from S(x|Z) = S0(x)^phi.
        np.random.seed(1)
        u = np.random.uniform(size=3000)
        Z = np.random.binomial(1, 0.5, 3000).reshape(-1, 1)
        phi = np.exp(1.0 * Z[:, 0])
        t = 10 * (-np.log(u) / phi) ** 0.5
        m = WeibullPH.fit(x=t, Z=Z)

        np.random.seed(2)
        xs, zs = m.random(100_000, np.array([[1.0]]))
        for tt in (2.0, 4.0, 6.0):
            emp = (np.asarray(xs) > tt).mean()
            mod = float(np.ravel(m.sf(tt, np.array([[1.0]])))[0])
            assert emp == pytest.approx(mod, abs=0.01)

    def test_random_two_covariates(self):
        # 271: used to raise a broadcast ValueError for >= 2 covariates.
        np.random.seed(3)
        t = 10 * np.random.weibull(2, 500)
        Z = np.hstack(
            [
                np.random.binomial(1, 0.5, 500).reshape(-1, 1),
                np.random.normal(size=(500, 1)),
            ]
        )
        m = WeibullPH.fit(x=t, Z=Z)
        x, z_out = m.random(7, np.array([[1.0, 0.5]]))
        assert np.shape(x) == (7,)
        assert np.shape(z_out) == (7, 2)
        assert np.all(np.isfinite(x))


# ---------------------------------------------------------------------------
# A plain-list ``Z`` and an ndarray ``init`` (#261).
# ---------------------------------------------------------------------------


def test_ph_fit_accepts_plain_list_Z():
    np.random.seed(5)
    x = Weibull.random(100, 10, 3)
    Z = [[float(v)] for v in np.random.normal(size=100)]
    m = WeibullPH.fit(x, Z=Z)
    assert np.isfinite(m.params).all()


def test_fit_accepts_ndarray_init():
    np.random.seed(6)
    x = Weibull.random(100, 10, 3)
    m = Weibull.fit(x, init=np.array([10.0, 3.0]))
    assert np.isfinite(m.params).all()


# ---------------------------------------------------------------------------
# Cox refuses left- and interval-censored rows; ``fit_from_df``
# with delayed entry; the Kalbfleisch-Prentice and exact tie
# likelihoods against brute force.
# ---------------------------------------------------------------------------


def _cox_data() -> tuple:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    Z = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]).reshape(-1, 1)
    return x, Z


def test_cox_rejects_left_censoring():
    # c = -1 used to be read as right-censored: the fit matched c[2] = 1.
    x, Z = _cox_data()
    with pytest.raises(ValueError, match="left-censored"):
        CoxPH.fit(x, Z, c=[0, 0, -1, 0, 0, 1])


def test_cox_rejects_interval_censoring():
    # Interval rows used to fail with an IndexError deep in the generator.
    x, Z = _cox_data()
    x2 = [[1, 2], 2, 3, 4, 5, 6]
    with pytest.raises(ValueError, match="interval-censored"):
        CoxPH.fit(x2, Z, c=[2, 0, 0, 0, 0, 1])


@pytest.mark.parametrize("method", ["breslow", "efron", "exact", "kp"])
def test_cox_rejects_left_censoring_every_entry_point(method):
    x, Z = _cox_data()
    c = [0, 0, -1, 0, 0, 1]
    with pytest.raises(ValueError, match="parametric regression"):
        CoxPH.fit(x, Z, c=c, tie_method=method, strata=[0, 0, 0, 1, 1, 1])
    df = pd.DataFrame({"x": x, "z": Z[:, 0], "c": c})
    with pytest.raises(ValueError, match="parametric regression"):
        CoxPH.fit_from_df(
            df, x_col="x", Z_cols="z", c_col="c", tie_method=method
        )


def test_cox_accepts_two_column_exact_times():
    # A two-column x with xl == xr everywhere is exact / right-censored data
    # written as intervals, and fits as such.
    x, Z = _cox_data()
    c = [0, 0, 1, 0, 0, 1]
    two_col = CoxPH.fit(np.column_stack([x, x]), Z, c=c)
    assert np.allclose(two_col.params, CoxPH.fit(x, Z, c=c).params)


def test_cox_fit_from_df_tl_col_matches_fit():
    rng = np.random.default_rng(1)
    n = 120
    z = rng.normal(size=n)
    tl = rng.uniform(0, 1.5, size=n)
    x = tl + rng.exponential(np.exp(-0.7 * z))
    c = (rng.uniform(size=n) < 0.2).astype(int)
    df = pd.DataFrame({"x": x, "z": z, "c": c, "entry": tl})

    for method in ("breslow", "efron"):
        from_df = CoxPH.fit_from_df(
            df,
            x_col="x",
            Z_cols="z",
            c_col="c",
            tl_col="entry",
            tie_method=method,
        )
        direct = CoxPH.fit(x, z.reshape(-1, 1), c=c, tl=tl, tie_method=method)
        assert np.allclose(from_df.params, direct.params)
        # ... and the entry ages change the answer.
        ignored = CoxPH.fit(x, z.reshape(-1, 1), c=c, tie_method=method)
        assert not np.allclose(from_df.params, ignored.params)


def test_cox_fit_from_df_masks_tl_and_strata_with_missing_covariates():
    # Rows with a missing covariate are dropped; the entry ages and stratum
    # labels must drop with them (strata used to raise a length mismatch).
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4, 5, 6, 7, 8.0],
            "z": [0, 1, np.nan, 1, 0, 1, 0, 1],
            "s": [0, 0, 0, 0, 1, 1, 1, 1],
            "tl": [0, 0, 0, 1, 1, 2, 2, 0.0],
        }
    )
    kept = df.dropna()
    m = CoxPH.fit_from_df(df, "x", Z_cols="z", tl_col="tl", strata_col="s")
    direct = CoxPH.fit(
        kept.x.values,
        kept[["z"]].values,
        tl=kept.tl.values,
        strata=kept.s.values,
        tie_method="efron",
    )
    assert np.allclose(m.params, direct.params)


def _brute_kp_neg_ll(beta, x, Z, c, tl):
    """Kalbfleisch-Prentice by listing every d-subset of each risk set."""
    ll = 0.0
    eta = Z @ beta
    for tau in np.unique(x[c == 0]):
        deaths = np.flatnonzero((x == tau) & (c == 0))
        risk = np.flatnonzero((tl < tau) & (x >= tau))
        subsets = [
            eta[list(s)].sum()
            for s in itertools.combinations(risk, len(deaths))
        ]
        ll += eta[deaths].sum() - logsumexp(subsets)
    return -ll


def _brute_exact_neg_ll(beta, x, Z, c, tl):
    """The exact partial likelihood by summing over every ordering."""
    ll = 0.0
    a = np.exp(Z @ beta)
    for tau in np.unique(x[c == 0]):
        deaths = np.flatnonzero((x == tau) & (c == 0))
        risk = np.flatnonzero((tl < tau) & (x >= tau))
        total = 0.0
        for order in itertools.permutations(deaths):
            remaining, prob = a[risk].sum(), 1.0
            for j in order:
                prob *= a[j] / remaining
                remaining -= a[j]
            total += prob
        ll += np.log(total)
    return -ll


def _small_tied_data(seed):
    rng = np.random.default_rng(seed)
    n = 18
    Z = rng.normal(size=(n, 2))
    x = rng.integers(1, 6, size=n).astype(float)
    c = (rng.uniform(size=n) < 0.25).astype(int)
    tl = np.where(rng.uniform(size=n) < 0.3, rng.uniform(0, 2, n), -np.inf)
    tl = np.minimum(tl, x - 0.5)
    return x, Z, c, tl


@pytest.mark.parametrize(
    "method, brute",
    [("kp", _brute_kp_neg_ll), ("exact", _brute_exact_neg_ll)],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_tie_likelihoods_match_brute_force(method, brute, seed):
    x, Z, c, tl = _small_tied_data(seed)
    xv, cv, nv, tlv, Zv = validate_coxph(x, c, None, Z, tl, method)
    neg_ll, jac_hess = CoxPH_()._resolve_func_generator(method)(
        xv, Zv, cv, nv, tlv
    )
    eps = 1e-5
    eye = np.eye(2)
    for beta in (np.zeros(2), np.array([0.4, -0.9]), np.array([1.5, 2.0])):
        ref = brute(beta, x, Z, c, tl)
        assert neg_ll(beta) == pytest.approx(ref, rel=1e-11)

        score, hess = jac_hess(beta)
        score_fd = np.array(
            [
                (brute(beta + eps * e, x, Z, c, tl))
                - brute(beta - eps * e, x, Z, c, tl)
                for e in eye
            ]
        ) / (2 * eps)
        assert np.allclose(score, score_fd, atol=1e-6)
        hess_fd = np.column_stack(
            [
                (jac_hess(beta + eps * e)[0] - jac_hess(beta - eps * e)[0])
                / (2 * eps)
                for e in eye
            ]
        )
        assert np.allclose(hess, hess_fd, atol=1e-5)


def _heavy_ties(seed=0):
    # 18 distinct times, 107 failures tied at one of them: the old KP took
    # over nine minutes on data like this.
    rng = np.random.default_rng(seed)
    x = np.concatenate(
        [np.full(107, 5.0), rng.integers(1, 19, size=150).astype(float)]
    )
    Z = rng.normal(size=(x.size, 2))
    c = (rng.uniform(size=x.size) < 0.2).astype(int)
    return x, Z, c


@pytest.mark.parametrize("method", ["kp", "exact"])
def test_heavy_ties_fit_quickly(method):
    x, Z, c = _heavy_ties()
    start = time.perf_counter()
    model = CoxPH.fit(x, Z, c=c, tie_method=method)
    elapsed = time.perf_counter() - start
    assert elapsed < 5.0
    assert np.all(np.isfinite(model.beta))
    assert np.all(np.isfinite(model.p_values))


@pytest.mark.parametrize("method", ["kp", "exact"])
def test_heavy_tie_fit_is_the_likelihood_maximum(method):
    # Score zero and a positive-definite information at the fitted beta.
    x, Z, c = _heavy_ties(1)
    model = CoxPH.fit(x, Z, c=c, tie_method=method)
    score, hess = model.jac(model.beta)
    assert np.allclose(score, 0.0, atol=1e-6)
    assert np.all(np.linalg.eigvalsh(hess) > 0)


def test_exact_every_unit_tied():
    # One tie set of 75 used to be refused (a cap of 12 ties) and, below
    # the cap, took tens of seconds.
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(80, 1))
    x = np.ones(80)
    c = np.zeros(80, dtype=int)
    c[:5] = 1  # five survivors keep the risk set larger than the tie set
    start = time.perf_counter()
    model = CoxPH.fit(x=x, Z=Z, c=c, tie_method="exact")
    assert time.perf_counter() - start < 5.0
    assert np.isfinite(model.beta[0])


# ---------------------------------------------------------------------------
# #394: an exactly observed time of inf.
# ---------------------------------------------------------------------------


_Z5 = [[-1.0], [1.0], [0.0], [2.0], [1.0], [0.0]]
_X5 = [np.inf, 0.5, 1.0, 2.0, 3.0, 4.0]
_E5 = ["a", "b", "a", "b", "a", "a"]


@pytest.mark.parametrize(
    "fit",
    [
        lambda: sp.CoxPH.fit(x=_X5, Z=_Z5),
        lambda: sp.CoxPH.fit(x=_X5, Z=_Z5, strata=[1, 1, 1, 2, 2, 2]),
        lambda: sp.AdditiveHazards.fit(_X5, _Z5),
        lambda: sp.BuckleyJames.fit(_X5, _Z5),
        lambda: CompetingRisksProportionalHazards.fit(_X5, _Z5, _E5),
        lambda: FineGray.fit(_X5, _Z5, _E5, event="a"),
        lambda: CompetingRisks.fit(_X5, _E5),
    ],
    ids=[
        "CoxPH",
        "stratified",
        "LinYing",
        "BuckleyJames",
        "CR-Cox",
        "FG",
        "CR",
    ],
)
def test_semi_parametric_fitters_refuse_an_infinite_event_time(fit):
    # Each used to take it as an event at infinity: Cox gave beta 19.4 on
    # two rows, the Aalen-Johansen incidence counted it, and Lin-Ying
    # failed with a LinAlgError from an SVD.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError, match=r"\(c=0\) must be finite"):
            fit()


def test_cox_accepts_an_infinite_censoring_time():
    model = sp.CoxPH.fit(
        [np.inf, 0.5, 1.0, 2.0], [[-1.0], [1.0], [0.0], [1.0]], c=[1, 0, 0, 0]
    )
    assert np.isfinite(model.beta).all()


# R survival 3.x, on the Rossi data (``Surv(week, arrest) ~ fin + age +
# prio``): logLik(fit), AIC(fit) and BIC(fit), which are on the partial
# likelihood with k the coefficients and BIC's n the events, nevent = 114
# (logLik.coxph, nobs.coxph); and the same with ``+ strata(wexp)``.
R_COX_ROSSI = {
    "efron": (-660.857025384416, 1327.714050768831, 1335.922646114015),
    "breslow": (-661.2326104166907, 1328.4652208333814, 1336.673816178565),
    "strata": (-582.473259917228, 1170.946519834455, 1179.155115179639),
}


def _rossi():
    from surpyval.datasets import load_rossi_static

    df = load_rossi_static()
    return (
        df["week"].values,
        df[["fin", "age", "prio"]].values,
        1 - df["arrest"].values,
        df["wexp"].values,
    )


@pytest.mark.parametrize("fit", sorted(R_COX_ROSSI))
def test_604_cox_model_comparison_values_are_r_survivals(fit):
    x, Z, c, wexp = _rossi()
    if fit == "strata":
        model = CoxPH.fit(x, Z, c=c, strata=wexp)
    else:
        model = CoxPH.fit(x, Z, c=c, tie_method=fit)
    ll, aic, bic = R_COX_ROSSI[fit]
    # A number, and methods, as on every other model (#604)
    assert isinstance(model.log_likelihood, float)
    assert model.log_likelihood == pytest.approx(ll, rel=1e-10)
    assert model.neg_ll() == pytest.approx(-ll, rel=1e-10)
    assert model.aic() == pytest.approx(aic, rel=1e-10)
    assert model.bic() == pytest.approx(bic, rel=1e-10)
    # The small-sample correction with the same k = 3 and n = 114
    assert model.aic_c() == pytest.approx(aic + 24 / 110, rel=1e-10)


def test_604_cox_neg_ll_of_beta_is_the_partial_likelihood_function():
    x, Z, c, _ = _rossi()
    model = CoxPH.fit(x, Z, c=c)
    assert model.neg_ll_of(model.params) == pytest.approx(model.neg_ll())
    # The old spelling still gives the function, deprecated, at the caller
    with pytest.warns(DeprecationWarning, match="neg_ll_of") as caught:
        value = model.neg_ll(np.zeros(3))
    assert caught[0].filename == __file__
    assert value == pytest.approx(model.neg_ll_of(np.zeros(3)))
    assert value > model.neg_ll()


def test_604_cox_restored_model_keeps_its_comparison_values():
    x, Z, c, _ = _rossi()
    model = CoxPH.fit(x, Z, c=c)
    restored = sp.from_dict(model.to_dict())
    for name in ("neg_ll", "aic", "aic_c", "bic"):
        assert getattr(restored, name)() == getattr(model, name)()
    assert restored.log_likelihood == model.log_likelihood
    # The function is not saved, and the old spelling says so
    assert restored.neg_ll_of is None
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError):
        restored.neg_ll(np.zeros(3))
    # A dict written before v0.23 stored the value under another key and
    # no sample size: the events, which the baseline counts, stand in.
    old = model.to_dict()
    old["_neg_log_like"] = old.pop("_neg_ll")
    del old["ic_n"]
    legacy = sp.from_dict(old)
    assert legacy.neg_ll() == model.neg_ll()
    assert legacy.bic() == pytest.approx(model.bic(), rel=1e-15)


def test_604_cox_aliased_coefficient_is_not_counted():
    x, Z, c, _ = _rossi()
    model = CoxPH.fit(x, Z, c=c)
    with pytest.warns(UserWarning, match="alias"):
        doubled = CoxPH.fit(x, np.column_stack([Z, Z[:, 0]]), c=c)
    # R's logLik.coxph counts sum(!is.na(coef))
    assert doubled.aic() == pytest.approx(model.aic(), rel=1e-10)


@pytest.mark.parametrize("ties", ["efron", "breslow"])
@pytest.mark.parametrize("seed", range(6))
def test_551_information_operator_is_the_generators(ties, seed):
    # CoxInformation gives the information as Z' (M Z): the generators'
    # matrix, with ties, counts and delayed entry.
    from surpyval.univariate.regression.proportional_hazards import (
        cox_likelihood as cl,
    )

    rng = np.random.default_rng(seed)
    N, p = 150, 3
    Z = rng.normal(size=(N, p))
    beta = rng.normal(size=p) * 0.5
    x = (
        np.ceil(rng.exponential(1, N) * 4) / 4
        if seed % 2
        else rng.exponential(1, N)
    )
    c = (rng.uniform(size=N) < 0.3).astype(int)
    n = rng.integers(1, 4, N).astype(float)
    tl = np.full(N, -np.inf)
    if seed >= 3:
        tl = np.where(rng.uniform(size=N) < 0.5, -np.inf, 0.5 * x)
    _, jac_hess = CoxPH._resolve_func_generator(ties)(x, Z, c, n, tl)
    info = cl.CoxInformation(x, c, n, Z @ beta, ties, tl=tl)
    np.testing.assert_allclose(
        Z.T @ info.apply(Z), jac_hess(beta)[1], rtol=1e-12, atol=1e-12
    )


def test_593_parametric_ph_fit_evaluates_each_point_once(monkeypatch):
    # The search used to evaluate the likelihood plainly at each point
    # and again in the gradient's autograd pass (13 such points here);
    # it now takes the value and the gradient from one pass.
    from autograd.tracer import Box

    from surpyval.univariate.regression.proportional_hazards import (
        proportional_hazards_fitter as ph_module,
    )

    def unboxed(value):
        while isinstance(value, Box):
            value = value._value
        return float(value)

    calls = []
    original = ph_module.regression_neg_ll

    def counted(model, data, *params):
        boxed = any(isinstance(p, Box) for p in params)
        calls.append((boxed, tuple(unboxed(p) for p in params)))
        return original(model, data, *params)

    monkeypatch.setattr(ph_module, "regression_neg_ll", counted)
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(300, 2))
    x = Weibull.random(300, 10, 1.5, random_state=3) * np.exp(
        Z @ np.array([0.3, -0.2])
    )
    model = WeibullPH.fit(x, Z)
    plain = {point for boxed, point in calls if not boxed}
    in_pass = {point for boxed, point in calls if boxed}
    assert len(plain & in_pass) <= 1
    assert model.maximum == "verified"
