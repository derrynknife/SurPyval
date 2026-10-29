"""The grid's posteriors against an independent sampler's.

The references were computed once with PyMC 5 (NUTS, 4 chains of 10,000
draws after 2,000 tuning steps, seeds 20261029 and 20261030, R-hat 1.00,
bulk ESS above 14,000, no divergences) in a separate environment, and are
fixed here: SurPyval does not depend on PyMC. The pass rule, fixed before
the first comparison: each posterior median and 2.5% and 97.5% point
within 4 Monte Carlo standard errors of the reference's.

Both cases use the same priors: log alpha Normal(log 10,000, 1.5) and
beta about 2, 90% between 1.5 and 3 (log beta Normal(log 2.121, 0.2107)).
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.bayes.prior import NormalOnScale

PRIORS = {
    "alpha": NormalOnScale(loc=np.log(1e4), scale=1.5, kind="log"),
    "beta": sp.prior.around(2, between=(1.5, 3)),
}

# (2.5%, 50%, 97.5%) and their Monte Carlo standard errors
REFERENCE = {
    "right": {
        "alpha": (
            (8298.213783438847, 14084.146436419494, 34034.544775347866),
            (30.82354470237533, 48.26738189331172, 422.2396660370323),
        ),
        "beta": (
            (1.4440780044880621, 2.1055033903617817, 3.047465346430154),
            (0.006407261752921167, 0.003535724934321083, 0.009733790515023877),
        ),
    },
    "interval": {
        "alpha": (
            (2278.9862285760582, 3326.60562639223, 5238.029747263928),
            (8.742208625849571, 4.48906866495463, 34.01508094613473),
        ),
        "beta": (
            (1.2964235366564187, 1.8509168060682248, 2.5887474810667896),
            (
                0.004245336837518265,
                0.0028173691187571848,
                0.0069742830175725246,
            ),
        ),
    },
}


def _right():
    x = np.r_[2100, 3400, 4050, np.linspace(1500, 6000, 37)]
    c = np.r_[0, 0, 0, np.ones(37)]
    return dict(x=x, c=c)


def _interval():
    xl = np.array([0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4000])
    xr = np.array([800, 1000, 1500, 2500, 3000, 3500, 4000, 5000, 6000, 6000])
    c = np.array([-1, 2, 2, 2, 2, 2, 2, 2, 1, 1])
    x = np.where((c == 2)[:, None], np.c_[xl, xr], np.c_[xr, xr])
    x = np.where((c == 1)[:, None], np.c_[xl, xl], x)
    return dict(x=x, c=c, tl=300.0)


@pytest.mark.parametrize("case", ["right", "interval"])
def test_matches_an_independent_sampler(case):
    data = _right() if case == "right" else _interval()
    model = sp.Weibull.fit_bayes(**data, priors=PRIORS)
    for name, (q, mcse) in REFERENCE[case].items():
        got = model.posterior.quantile(name, [0.025, 0.5, 0.975])
        assert np.all(np.abs(got - np.asarray(q)) <= 4 * np.asarray(mcse)), (
            name,
            got,
            q,
        )
