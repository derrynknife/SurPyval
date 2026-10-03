"""Card: repairs of a fleet of field units, and how good a repair is.

Persona: a maintenance engineer with eight years of failure and repair
records for 40 inverters, each repaired after every failure. Questions
(Kijima, 1989; Yanez, Joglar and Modarres, 2002):

1. Do repairs restore a unit to as good as new, as bad as old, or
   between -- the restoration factor q, with its interval?
2. Is the answer significantly different from perfect and from minimal
   repair?
3. For each unit, from its own history: its virtual age now, the chance
   it fails in the next year, and the fleet's failures over the year.

Truth: Kijima type I, q = 0.4, lives Weibull(30 months, 2.5).
"""

import numpy as np

import surpyval as sp
from surpyval.recurrent import GeneralizedRenewal
from surpyval.tests.scenarios._oracles import contains

Q, ALPHA, BETA, T_END = 0.4, 30.0, 2.5, 96.0


def _histories():
    rng = np.random.default_rng(13)
    x, i, c = [], [], []
    for unit in range(40):
        t = virtual = 0.0
        while True:
            # Next life given the virtual age: H(v + X) = H(v) + E.
            h = (virtual / ALPHA) ** BETA + rng.exponential()
            life = ALPHA * h ** (1 / BETA) - virtual
            if t + life >= T_END:
                x.append(T_END), i.append(unit), c.append(1)
                break
            t += life
            x.append(t), i.append(unit), c.append(0)
            virtual += Q * life
    return np.array(x), np.array(i), np.array(c)


X, I, C = _histories()


def test_restoration_factor_recovered():
    model = GeneralizedRenewal.fit(X, I, c=C, kijima="i")
    assert model.maximum == "verified"
    assert contains(model.param_cb("q", alpha_ci=0.05), Q)
    assert contains(model.param_cb("alpha", alpha_ci=0.05), ALPHA)
    assert contains(model.param_cb("beta", alpha_ci=0.05), BETA)


def test_neither_perfect_nor_minimal_repair():
    model = GeneralizedRenewal.fit(X, I, c=C, kijima="i")
    result = model.repair_test()
    text = str(result).lower()
    assert "both perfect and minimal repair rejected" in text


def test_each_units_state_and_its_next_year():
    # #581: from its own history, each unit's virtual age now (Kijima I:
    # q times its lives so far, plus the time since its last repair), the
    # chance it fails in the next year from there, and the fleet's count.
    model = GeneralizedRenewal.fit(X, I, c=C, kijima="i")
    q, alpha, beta = model.params
    states = model.unit_states()
    virtual, since = [], []
    for unit in np.unique(I):
        repairs = X[(I == unit) & (C == 0)]
        last = repairs[-1] if repairs.size else 0.0
        virtual.append(q * last + (T_END - last))
        since.append(T_END - last)
    np.testing.assert_allclose(states.virtual_age, virtual, rtol=1e-10)
    np.testing.assert_allclose(states.since_failure, since, rtol=1e-10)
    v = np.array(virtual)
    p = 1 - np.ravel(model.next_failure_sf(12.0))
    expected = 1 - np.exp((v / alpha) ** beta - ((v + 12.0) / alpha) ** beta)
    np.testing.assert_allclose(p, expected, rtol=1e-8)
    fleet = sp.forecast(model, horizon=12.0, random_state=1)
    assert fleet.lower[0] <= fleet.expected[0] <= fleet.upper[0]
    # A unit's expected failures are at least its chance of one.
    assert np.all(np.ravel(fleet.per_unit) >= p - 0.05)
