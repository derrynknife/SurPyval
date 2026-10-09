"""Card: an accelerated life test of a capacitor, in temperature and voltage.

Persona: a component reliability engineer qualifying a capacitor for a
45 C / 400 V application from a test at 85, 105 and 125 C and three
voltages, and from a step-stress test of a second batch. Questions
(Nelson, Accelerated Testing, ch. 2-6; Meeker and Escobar ch. 19-20):

1. What are the activation energy and the voltage exponent, and the
   reliability over five years at use, with bounds?
2. Do the equivalent ways of writing the model -- the Arrhenius-power
   life model, an AFT model on (1/T, log V), the general log-linear
   life model -- give the same answer?
3. From a step-stress test (85 -> 105 -> 125 C), under Nelson's
   cumulative exposure model: the same questions.

Truth: Weibull, beta = 2.2; life(T, V) = c exp(a / T) V^n with
Ea = 0.7 eV (a = 8123 K) and n = -3; eta = 150,000 h at use.
"""

import numpy as np
import pytest
from scipy.stats import chi2

import surpyval as sp
from surpyval.tests.scenarios._oracles import best_of, contains

KELVIN = 273.15
A_TRUE, N_TRUE, BETA_TRUE = 0.7 / 8.617e-5, -3.0, 2.2
USE = (45 + KELVIN, 400.0)
ETA_USE = 150_000.0
C_TRUE = ETA_USE / (np.exp(A_TRUE / USE[0]) * USE[1] ** N_TRUE)
FIVE_YEARS = 5 * 8760.0
TEST_END = 3000.0


def _eta(T, V):
    return C_TRUE * np.exp(A_TRUE / T) * V**N_TRUE


def _alt():
    rng = np.random.default_rng(7)
    T, V, x = [], [], []
    for celsius in (85, 105, 125):
        for volts in (400, 500, 600):
            T += [celsius + KELVIN] * 12
            V += [volts] * 12
            x += list(
                _eta(celsius + KELVIN, volts) * rng.weibull(BETA_TRUE, 12)
            )
    x = np.array(x)
    c = (x > TEST_END).astype(int)
    return np.minimum(x, TEST_END), c, np.column_stack([T, V])


X, C, Z = _alt()
ALT = sp.AcceleratedLife(sp.Weibull, sp.life_models.PowerExponential)


def _alt_neg_ll(p):
    log_c, a, n, log_beta = p
    beta = np.exp(log_beta)
    eta = np.exp(log_c + a / Z[:, 0] + n * np.log(Z[:, 1]))
    z = (X / eta) ** beta
    log_f = np.log(beta / eta) + (beta - 1) * np.log(X / eta) - z
    return -np.where(C == 0, log_f, -z).sum()


def test_dual_stress_fit_is_the_mle():
    model = ALT.fit(x=X, Z=Z, c=C)
    c, a, n = model.phi_params
    beta = model.params[model.parameter_names.index("beta")]
    ref = best_of(
        _alt_neg_ll,
        [[np.log(c), a, n, np.log(beta)], [np.log(C_TRUE), 8000, -2, 0.5]],
    )
    assert model.maximum == "verified"
    assert model.neg_ll() == pytest.approx(ref.fun, abs=1e-5)


def test_dual_stress_recovers_the_truth():
    model = ALT.fit(x=X, Z=Z, c=C)
    assert contains(model.param_cb("a", alpha_ci=0.05), A_TRUE)
    assert contains(model.param_cb("n", alpha_ci=0.05), N_TRUE)
    assert contains(model.param_cb("beta", alpha_ci=0.05), BETA_TRUE)
    truth = np.exp(-((FIVE_YEARS / ETA_USE) ** BETA_TRUE))
    assert contains(
        model.cb(FIVE_YEARS, np.array([USE]), alpha_ci=0.05), truth
    )


def test_equivalent_parameterisations_agree():
    use = np.array([USE])
    use_t = np.array([[1 / USE[0], np.log(USE[1])]])
    Zt = np.column_stack([1 / Z[:, 0], np.log(Z[:, 1])])
    fits = [
        (ALT.fit(x=X, Z=Z, c=C), use),
        (sp.WeibullAFT.fit(x=X, Z=Zt, c=C), use_t),
        (
            sp.AcceleratedLife(
                sp.Weibull, sp.life_models.GeneralLogLinear
            ).fit(x=X, Z=Zt, c=C),
            use_t,
        ),
    ]
    ref_model, ref_z = fits[0]
    for model, z in fits[1:]:
        assert model.neg_ll() == pytest.approx(ref_model.neg_ll(), abs=1e-5)
        np.testing.assert_allclose(
            model.sf(FIVE_YEARS, z), ref_model.sf(FIVE_YEARS, ref_z), rtol=1e-4
        )


# ---------------------------------------------------------------------------
# Step stress, cumulative exposure
# ---------------------------------------------------------------------------
STEPS = [
    (0.0, 1000.0, 85 + KELVIN),
    (1000.0, 2000.0, 105 + KELVIN),
    (2000.0, 3000.0, 125 + KELVIN),
]


def _step_stress():
    """Start-stop rows (unit, start, stop, censored, T) of 60 units under
    Nelson's cumulative exposure: each unit fails when its exposure, the
    integral of 1 / eta(T(t)), reaches a unit Weibull draw."""
    rng = np.random.default_rng(3)
    rows = []
    for k, u in enumerate(rng.weibull(BETA_TRUE, 60)):
        exposure = 0.0
        for start, stop, T in STEPS:
            eta = _eta(T, USE[1])
            if u - exposure <= (stop - start) / eta:
                rows.append((k, start, start + (u - exposure) * eta, 0, T))
                break
            rows.append((k, start, stop, 1, T))
            exposure += (stop - start) / eta
    return map(np.array, zip(*rows))


I, XL, XR, CS, TS = _step_stress()


def _exposure_neg_ll(p):
    """Cumulative exposure Weibull with an Arrhenius scale."""
    log_eta_use, a, log_beta = p
    beta = np.exp(log_beta)
    eta = np.exp(log_eta_use + a * (1 / TS - 1 / USE[0]))
    total = 0.0
    for k in np.unique(I):
        rows = I == k
        e_end = np.cumsum((XR[rows] - XL[rows]) / eta[rows])
        e = e_end[-1]
        if CS[rows][-1] == 0:
            total += (
                np.log(beta) + (beta - 1) * np.log(e) - np.log(eta[rows][-1])
            )
        total -= e**beta
    return -total


def test_step_stress_cumulative_exposure_is_the_mle():
    model = sp.WeibullAFT.fit_tvc(I, XL, XR, CS, (1000 / TS)[:, None])
    ref = best_of(
        _exposure_neg_ll,
        [
            [np.log(ETA_USE), A_TRUE, np.log(BETA_TRUE)],
            [np.log(3e4), 5000.0, 1.0],
        ],
    )
    assert model.maximum == "verified"
    assert model.neg_ll() == pytest.approx(ref.fun, abs=1e-5)


def test_step_stress_in_the_arrhenius_covariate():
    # #577: Z = 1/T in kelvin (about 3e-3) is a reparameterisation of
    # 1000/T: the same maximum, verified, with the coefficient scaled by
    # 1000 (the fit once stopped at its start, coefficient 0).
    raw = sp.WeibullPH.fit_tvc(I, XL, XR, CS, (1 / TS)[:, None])
    scaled = sp.WeibullPH.fit_tvc(I, XL, XR, CS, (1000 / TS)[:, None])
    assert raw.maximum == "verified"
    assert raw.neg_ll() == pytest.approx(scaled.neg_ll(), abs=1e-5)
    assert raw.params[-1] == pytest.approx(1000 * scaled.params[-1], rel=1e-4)


def _profile_neg_ll(sf_use, start):
    """The ALT likelihood maximised with R(5 years) at use held at
    ``sf_use``: the scale is fixed by it, given a, n and beta."""

    def neg_ll(p):
        a, n, log_beta = p
        eta_use = FIVE_YEARS / (-np.log(sf_use)) ** np.exp(-log_beta)
        log_c = np.log(eta_use) - a / USE[0] - n * np.log(USE[1])
        return _alt_neg_ll([log_c, a, n, log_beta])

    return best_of(neg_ll, [start]).fun


def test_likelihood_ratio_bounds_at_use():
    # #583: the likelihood-ratio bounds on R(5 years) at use, from the
    # profile likelihood: at each end the deviance is chi2(1)'s 95%
    # point, and the bounds cover the truth.
    model = ALT.fit(x=X, Z=Z, c=C)
    use = np.array([USE])
    lr = np.ravel(model.cb(FIVE_YEARS, use, alpha_ci=0.05, method="lr"))
    _, a, n = model.phi_params
    start = [a, n, np.log(model.params[model.parameter_names.index("beta")])]
    for end in lr:
        deviance = 2 * (_profile_neg_ll(end, start) - model.neg_ll())
        assert deviance == pytest.approx(chi2.ppf(0.95, 1), abs=1e-3)
    assert lr[0] < np.ravel(model.sf(FIVE_YEARS, use))[0] < lr[1]
    assert contains(lr, np.exp(-((FIVE_YEARS / ETA_USE) ** BETA_TRUE)))
