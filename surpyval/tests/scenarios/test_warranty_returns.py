"""Card: warranty returns from a Nevada chart.

Persona: a product reliability engineer with 24 monthly shipment cohorts
(about 125,000 units) and the returns of each cohort by month in
service -- interval-censored counts, the survivors of each cohort
censored at its own age. Questions (Meeker and Escobar ch. 7 and 9;
Kalbfleisch, Lawless and Robinson, 1991; Abernethy ch. 8):

1. Is there a defective sub-population (infant mortality) as well as
   wear-out, and what fraction of units is defective?
2. Which model -- a single Weibull, a limited failure population, a
   two-Weibull mixture -- and how far does each extrapolate past the
   warranty?
3. How many returns in the next six months, with a prediction interval?

Truth: 3% of units defective, Weibull(eta = 2 months, beta = 0.7); the
rest Weibull(120 months, 3).
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.tests.scenarios._oracles import (
    best_of,
    interval_counts_neg_ll,
    weibull_sf,
)

P_DEFECTIVE, DEFECT, WEAR = 0.03, (2.0, 0.7), (120.0, 3.0)
WARRANTY, AHEAD = 36, 6  # months of cover; months to forecast


def _nevada():
    """The Nevada chart as interval-censored counts, and each cohort's
    (age, survivors, returns in the next six months under warranty): the
    units' lives are drawn whole, so what happens next is known."""
    rng = np.random.default_rng(21)
    x, c, n, cohorts = [], [], [], []
    for k, shipped in enumerate(rng.integers(3500, 6500, 24), start=1):
        age = 24 - k + 1  # months in service at the cut-off
        bad = rng.random(shipped) < P_DEFECTIVE
        life = np.where(
            bad,
            DEFECT[0] * rng.weibull(DEFECT[1], shipped),
            WEAR[0] * rng.weibull(WEAR[1], shipped),
        )
        month = np.ceil(life)
        for j in range(1, age + 1):
            returned = int((month == j).sum())
            if returned:
                x.append([j - 1, j])
                c.append(2)
                n.append(returned)
        x.append([age, age])
        c.append(1)
        n.append(int((month > age).sum()))
        ahead = (month > age) & (month <= min(age + AHEAD, WARRANTY))
        cohorts.append((age, n[-1], int(ahead.sum())))
    return (
        np.array(x, dtype=float),
        np.array(c),
        np.array(n),
        np.array(cohorts),
    )


X, C, N, COHORTS = _nevada()


def _lfp_neg_ll(p):
    alpha, beta = np.exp(p[:2])
    q = 1 / (1 + np.exp(-p[2]))
    return interval_counts_neg_ll(
        lambda t: q * (1 - weibull_sf(t, alpha, beta)), X, C, N
    )


def _mixture_neg_ll(p):
    w = 1 / (1 + np.exp(-p[0]))
    a1, b1, a2, b2 = np.exp(p[1:])
    return interval_counts_neg_ll(
        lambda t: w * (1 - weibull_sf(t, a1, b1))
        + (1 - w) * (1 - weibull_sf(t, a2, b2)),
        X,
        C,
        N,
    )


def test_single_weibull_is_the_mle():
    model = sp.Weibull.fit(x=X, c=C, n=N)
    ref = best_of(
        lambda p: interval_counts_neg_ll(
            lambda t: 1 - weibull_sf(t, *np.exp(p)), X, C, N
        ),
        [np.log(model.params), [np.log(50.0), 0.0]],
    )
    assert model.maximum == "verified"
    assert model.neg_ll() == pytest.approx(ref.fun, abs=1e-4)


def test_defective_fraction_by_lfp():
    # #579: the limited-failure-population fit of the counts reaches the
    # MLE (it stuck at a fraction of 1, no defective sub-population, and
    # reported a verified maximum). The fraction is lfp_p.
    model = sp.Weibull.fit(x=X, c=C, n=N, lfp=True)
    ref = best_of(
        _lfp_neg_ll,
        [[np.log(2.0), np.log(0.7), np.log(0.03 / 0.97)], [1.6, 0.0, -3.0]],
    )
    assert model.maximum == "verified"
    assert model.lfp_p < 0.5
    assert model.neg_ll() == pytest.approx(ref.fun, abs=1e-3)


def test_defective_fraction_by_mixture():
    # #582: the two-Weibull mixture's EM reaches the MLE (it landed about
    # 100 log-likelihood units short, a defective fraction of 0.26), and
    # its neg_ll is the negative log-likelihood, as on every model (#572).
    model = sp.MixtureModel.fit(x=X, c=C, n=N, dist=sp.Weibull, m=2)
    ref = best_of(
        _mixture_neg_ll,
        [
            [
                np.log(0.03 / 0.97),
                np.log(2.0),
                np.log(0.7),
                np.log(120.0),
                np.log(3.0),
            ]
        ],
    )
    w = np.asarray(model.w)
    (a1, b1), (a2, b2) = np.asarray(model.params)
    at_fit = _mixture_neg_ll(
        [np.log(w[0] / w[1]), np.log(a1), np.log(b1), np.log(a2), np.log(b2)]
    )
    assert model.neg_ll() == pytest.approx(at_fit, abs=1e-6)
    assert model.neg_ll() == pytest.approx(ref.fun, abs=1e-2)
    assert min(w) == pytest.approx(P_DEFECTIVE, abs=0.01)


def test_returns_in_the_next_six_months():
    # #581: the cohort forecast from where each cohort is now -- its
    # survivors at their age, over the next six months, capped at the end
    # of cover -- is forecast(); it is the sum the analyst wrote by hand,
    # and the returns that then happen fall in its prediction interval.
    model = sp.MixtureModel.fit(x=X, c=C, n=N, dist=sp.Weibull, m=2)
    age, survivors, returned = COHORTS.T
    result = sp.forecast(
        model,
        age=age,
        n=survivors,
        horizon=np.arange(1, AHEAD + 1),
        limit=WARRANTY,
    )
    F = model.ff
    end = np.minimum(age + AHEAD, WARRANTY)
    by_hand = np.sum(survivors * (F(end) - F(age)) / (1 - F(age)))
    assert result.expected[-1] == pytest.approx(by_hand, rel=1e-8)
    assert result.period_expected.sum() == pytest.approx(by_hand, rel=1e-8)
    assert result.lower[-1] <= returned.sum() <= result.upper[-1]
