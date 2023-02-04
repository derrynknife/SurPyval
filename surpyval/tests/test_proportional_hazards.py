import numpy as np
import pytest

from surpyval import CoxPH


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
    assert pytest.approx(2.12, abs=0.01) == model.parameters[0]


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

    # beta_0 should be 2.12
    assert pytest.approx(-0.34, abs=0.01) == model.parameters[0]


def test_cox_ph_sim_example():
    """
    Generates samples randomly and tests convergence.
    """
    # Instantiate random number generator
    rng = np.random.default_rng()

    # Construct covariant (Z) matrix
    n_samples = 100
    n_covariants = 3
    Z = rng.normal(size=(n_samples, n_covariants))

    # Baseline hazard function (i.e. an exponential survival function)
    baseline_hazard_rate = 0.01

    # Covariant coefficients
    beta = [0.1, -0.5, 0.8]

    # Take 20 samples per covariant sample for adequate fitting
    # Have to repeat Z for this
    samples_per_covariant_sample = 20
    Z_repeated = np.repeat(Z, samples_per_covariant_sample, axis=0)

    # Fill x samples
    x = np.zeros(n_samples * samples_per_covariant_sample)
    for i, Z_i in enumerate(Z_repeated):
        Z_i_hazard_rate = baseline_hazard_rate * np.exp(np.dot(Z_i, beta))
        x[i] = rng.exponential(1 / Z_i_hazard_rate)

    # Fit model
    model = CoxPH.fit(x=x, Z=Z_repeated, c=[0] * len(x))

    # Parameters should be approximately equal to the beta vector
    assert pytest.approx(beta, abs=0.05) == model.parameters
