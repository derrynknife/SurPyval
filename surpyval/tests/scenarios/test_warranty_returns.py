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
3. How many returns in the next six months, and at what cost?
   (Not yet a test: the cohort forecast is hand-written; #581.)

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


def _nevada():
    rng = np.random.default_rng(21)
    x, c, n = [], [], []
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
    return np.array(x, dtype=float), np.array(c), np.array(n)


X, C, N = _nevada()


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


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#579: lfp=True on interval-censored counts sticks at p = 1 (no "
        "defective sub-population) and reports a verified maximum"
    ),
)
def test_defective_fraction_by_lfp():
    model = sp.Weibull.fit(x=X, c=C, n=N, lfp=True)
    ref = best_of(
        _lfp_neg_ll,
        [[np.log(2.0), np.log(0.7), np.log(0.03 / 0.97)], [1.6, 0.0, -3.0]],
    )
    assert model.p < 0.5
    assert model.neg_ll() == pytest.approx(ref.fun, abs=1e-3)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "#582: the two-Weibull mixture's EM lands about 100 log-likelihood "
        "units short of the MLE (defective fraction 0.26 for 0.03)"
    ),
)
def test_defective_fraction_by_mixture():
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
    # The likelihood at the fitted parameters, by the oracle (MixtureModel's
    # own value is spelt differently from the other models', #572).
    w = np.asarray(model.w)
    (a1, b1), (a2, b2) = np.asarray(model.params)
    at_fit = _mixture_neg_ll(
        [np.log(w[0] / w[1]), np.log(a1), np.log(b1), np.log(a2), np.log(b2)]
    )
    assert at_fit == pytest.approx(ref.fun, abs=1e-2)
    assert min(w) == pytest.approx(P_DEFECTIVE, abs=0.01)
