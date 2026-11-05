"""The interval-censored log-rank and Wilcoxon-type tests against R.

The values are ``interval::ictest(L, R, group, scores = ..., method =
"pclt")`` (R 4.6.2, interval 1.1.1.0, Icens 1.84.0, perm 1.0.0.4),
printed by ``scripts/reference/reference_r_interval_logrank.R``.

The data are the breast cosmesis data of Finkelstein and Wolfe (1985),
as shipped with the interval package, and 45 simulated units in three
groups with exact, right, left and interval-censored rows (typed out in
the script and below).

The statistic is a closed form in the pooled Turnbull NPMLE, so the
agreement depends on how far each NPMLE converged. Both stop on the same
Karush-Kuhn-Tucker gap: R's ``epsilon = 1e-12`` against SurPyval's
default ``tol = 1e-12``. Those agree to 1e-8 relative (seen: about
1e-13). R's default ``epsilon = 1e-6`` stops earlier; its values are
within 1e-6 relative of SurPyval's.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp

inf = np.inf

BCOS_LEFT = [
    45, 6, 0, 46, 46, 7, 17, 7, 37, 0, 4, 15, 11, 22, 46, 46, 25, 46, 26,
    46, 27, 36, 46, 36, 37, 40, 17, 46, 11, 38, 5, 37, 0, 18, 24, 36, 5, 19,
    17, 24, 32, 33, 19, 37, 34, 36, 8, 0, 24, 17, 17, 24, 16, 13, 11, 16, 18,
    17, 32, 23, 44, 14, 0, 5, 12, 11, 33, 31, 13, 19, 34, 13, 16, 35, 15, 11,
    22, 10, 30, 13, 10, 8, 4, 11, 14, 4, 34, 30, 18, 16, 35, 21, 11, 48,
]  # fmt: skip
BCOS_RIGHT = [
    inf, 10, 7, inf, inf, 16, inf, 14, 44, 8, 11, inf, 15, inf, inf, inf, 37,
    inf, 40, inf, 34, 44, inf, 48, inf, inf, 25, inf, 18, inf, 12, inf, 5,
    inf, inf, inf, 11, 35, 25, inf, inf, inf, 26, inf, inf, inf, 12, 22, 31,
    27, 23, 30, 24, inf, 13, 20, 25, 26, inf, inf, 48, 17, 5, 8, 20, inf, 40,
    inf, 39, 32, inf, inf, 24, inf, 22, 17, 32, 35, 34, inf, 17, 21, 9, inf,
    19, 8, inf, 36, 24, 60, 39, inf, 20, inf,
]  # fmt: skip
BCOS_GROUP = ["Rad"] * 46 + ["RadChem"] * 48

SIM_L = [
    11, 3, 3, 3, 6, 3.4, 12, 4.7, 5.3, 7.5, 0, 0, 6, 0, 0, 0, 5.6, 7.7, 24,
    6.7, 0, 12, 0, 6.3, 0, 12, 4.6, 6, 7.3, 2, 5.9, 2.2, 0, 0, 3.9, 7.9, 6, 6,
    0, 0, 0, 5.3, 15, 3, 6,
]  # fmt: skip
SIM_R = [
    inf, 6, 6, 6, 9, 3.4, 15, inf, 5.3, 7.5, 7.7, 7.8, 9, 10.5, 3, 16.6, 5.6,
    inf, 27, 6.7, 3, 15, 14.8, 6.3, 6.2, 15, inf, 6, inf, inf, inf, 2.2, 3,
    17.9, inf, 7.9, 9, 9, 3, 3, 4.7, inf, 18, 6, 9,
]  # fmt: skip
SIM_GROUP = ["a"] * 15 + ["b"] * 15 + ["c"] * 15

# (data, scores) -> (chisq, p, U of the groups in sorted order), R's
# epsilon = 1e-12.
R_TIGHT = {
    ("bcos", "logrank1"): (
        7.12028999430327,
        0.00762163771214238,
        [-9.1418461693435],
    ),
    ("bcos", "logrank2"): (
        7.20329827559263,
        0.00727697172067944,
        [-9.94418243726674],
    ),
    ("bcos", "wmw"): (
        4.69654185193163,
        0.0302233704557723,
        [-5.65672403980097],
    ),
    ("sim3", "logrank1"): (
        2.50745989616309,
        0.285438139367978,
        [2.2885950947861, -3.75538443642309, 1.46678934163698],
    ),
    ("sim3", "logrank2"): (
        2.49135189485758,
        0.287746340980266,
        [2.56791686209981, -4.24704650527002, 1.67912964317021],
    ),
    ("sim3", "wmw"): (
        1.8483214400796,
        0.396864359644243,
        [1.00728996221215, -2.22622031823376, 1.21893035602161],
    ),
    ("sim2", "logrank1"): (2.29160891892002, 0.130075024163803, None),
    ("sim2", "logrank2"): (2.34831286767238, 0.125418679450472, None),
    ("sim2", "wmw"): (1.9167933007178, 0.166210596303402, None),
}
# The same with R's default epsilon = 1e-6.
R_DEFAULT = {
    ("bcos", "logrank1"): (7.12029022815353, 0.00762163671800201),
    ("bcos", "logrank2"): (7.20329830031247, 0.00727697162044573),
    ("bcos", "wmw"): (4.69654190346449, 0.0302233695494887),
    ("sim3", "logrank1"): (2.50745990732205, 0.285438137775382),
    ("sim3", "logrank2"): (2.49135190295077, 0.287746339815874),
    ("sim3", "wmw"): (1.84832144312115, 0.396864359040701),
    ("sim2", "logrank1"): (2.29160894292948, 0.130075022151906),
    ("sim2", "logrank2"): (2.34831288094157, 0.125418678382778),
    ("sim2", "wmw"): (1.91679331195094, 0.166210595062043),
}
# R's Monte Carlo permutation p-values (method = "exact.mc", nmc = 99999;
# two-sided by the absolute statistic, tsmethod = "abs", for two groups).
R_MC = {
    ("bcos", "logrank1"): 0.007170,
    ("bcos", "wmw"): 0.029750,
    ("sim3", "logrank1"): 0.297570,
    ("sim3", "wmw"): 0.405250,
}


def _data(name):
    if name == "bcos":
        return np.array(BCOS_LEFT, float), np.array(BCOS_RIGHT), BCOS_GROUP
    L, R, g = np.array(SIM_L), np.array(SIM_R), np.array(SIM_GROUP)
    if name == "sim2":
        keep = g != "c"
        return L[keep], R[keep], g[keep]
    return L, R, g


@pytest.mark.parametrize("key", sorted(R_TIGHT))
def test_matches_ictest(key):
    name, scores = key
    xl, xr, g = _data(name)
    res = sp.interval_logrank(xl=xl, xr=xr, Z=g, scores=scores)
    chisq, p, U = R_TIGHT[key]
    assert res.npmle_converged
    assert_allclose(res.statistic, chisq, rtol=1e-8)
    assert_allclose(res.p_value, p, rtol=1e-8)
    assert res.dof == len(np.unique(g)) - 1
    if U is not None:
        # R lists the groups as they first appear (here, sorted); for two
        # groups only the first one's sum.
        got = list(res.U.values())[: len(U)]
        assert_allclose(got, U, rtol=1e-8, atol=1e-10)


@pytest.mark.parametrize("key", sorted(R_DEFAULT))
def test_within_r_default_convergence(key):
    name, scores = key
    xl, xr, g = _data(name)
    res = sp.interval_logrank(xl=xl, xr=xr, Z=g, scores=scores)
    chisq, p = R_DEFAULT[key]
    assert_allclose(res.statistic, chisq, rtol=1e-6)
    assert_allclose(res.p_value, p, rtol=1e-6)


@pytest.mark.parametrize("key", sorted(R_MC))
def test_permutation_p_value_agrees_with_r_monte_carlo(key):
    """Different random streams estimate the same exact p-value: the two
    estimates agree within four standard errors of their difference."""
    name, scores = key
    xl, xr, g = _data(name)
    B = 19999
    res = sp.interval_logrank(
        xl=xl,
        xr=xr,
        Z=g,
        scores=scores,
        method="permutation",
        n_permutations=B,
        random_state=231,
    )
    p_r = R_MC[key]
    se = np.sqrt(p_r * (1 - p_r) * (1 / B + 1 / 99999))
    assert abs(res.p_value - p_r) < 4 * se
