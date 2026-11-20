"""The shared gamma-frailty Poisson process against R's ``MASS::glm.nb``
(scripts/reference/reference_r_shared_frailty.R).

Units all watched from 0 to one common time ``T`` under a power-law
intensity with a gamma frailty have a likelihood that splits in two: the
counts are negative binomial with one common mean (the cumulative
intensity at ``T``) and size ``1 / theta``, and, given the counts, the
event times carry the shape alone. So the fit's frailty variance and
mean count are those of a negative binomial fit to the counts, and its
shape is the closed form ``N / sum(log(T / t))``.

The tolerances were fixed before the comparison was run: 1e-4 (relative)
on ``theta`` (``glm.nb`` stops its own iteration at about that), 1e-6
(relative) on the mean count and the shape, and 1e-6 on the
log-likelihood of the counts.
"""

import warnings

import numpy as np
import pytest
from scipy.special import gammaln

from surpyval.recurrent import CrowAMSAA, Duane

COUNTS = [0, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 7, 7, 8, 9]
COUNTS += [10, 11, 12, 14, 16, 18, 21, 25]
T_END = 100.0
# R 4.6.2, MASS 7.3.66: glm.nb(n ~ 1)
R_MEAN = 7.23333333333333
R_THETA = 0.534971191863765
R_LOGLIK = -89.4685504679051


def _fleet():
    x, i, c = [], [], []
    for unit, n in enumerate(COUNTS):
        # Times spread over (0, T): any times do, the counts carry theta
        times = T_END * ((np.arange(n) + 0.5) / max(n, 1)) ** (1 / 1.3)
        x += [*times, T_END]
        i += [unit] * (n + 1)
        c += [0] * n + [1]
    return np.array(x), np.array(i), np.array(c)


@pytest.mark.parametrize("fitter", [CrowAMSAA, Duane])
def test_gamma_frailty_is_the_negative_binomial_fit_of_the_counts(fitter):
    x, i, c = _fleet()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = fitter.fit(x, i, c, frailty="gamma")
    assert model.maximum == "verified"
    assert model.theta == pytest.approx(R_THETA, rel=1e-4)
    assert float(model.cif(T_END)) == pytest.approx(R_MEAN, rel=1e-6)
    events = x[c == 0]
    shape = events.size / np.sum(np.log(T_END / events))
    beta = model.params[1] if fitter is CrowAMSAA else model.params[0]
    assert beta == pytest.approx(shape, rel=1e-6)
    # The counts' part of the log-likelihood: the whole less the event
    # times' density given the counts, with the counts' factorials (the
    # times are ordered).
    times_part = np.sum(
        np.log(shape) + (shape - 1) * np.log(events) - shape * np.log(T_END)
    )
    counts_part = (
        model.log_likelihood
        - times_part
        - np.sum(gammaln(np.array(COUNTS) + 1.0))
    )
    assert counts_part == pytest.approx(R_LOGLIK, abs=1e-6)
