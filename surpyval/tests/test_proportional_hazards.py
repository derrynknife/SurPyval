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
