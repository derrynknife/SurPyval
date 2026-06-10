"""
Parametric fits validated against known expected parameter values.

Historically these tests refitted every distribution with ``lifelines`` and
``reliability`` at run time and compared the parameters. Both packages have
been dropped as test dependencies (``reliability`` and, since pandas 3, also
``lifelines`` pin ``pandas<3``). Instead, the expected MLE parameters below
were generated once with ``lifelines`` (``WeibullFitter``,
``ExponentialFitter``, ``LogNormalFitter``, ``LogLogisticFitter`` and their
``fit_interval_censoring`` variants) and ``reliability`` (``Fit_*_2P`` /
``Fit_Exponential_1P``) on the fixed datasets defined here, then hard-coded.

The expected values are stored in surpyval's own parameter convention. The
mapping from the external packages is:

* Weibull       -- ``(alpha, beta)`` matches lifelines ``(lambda_, rho_)`` and
                   reliability ``(alpha, beta)`` directly.
* Exponential   -- surpyval's parameter is the rate, equal to reliability's
                   ``Lambda`` and to ``1 / lifelines_scale``.
* LogNormal     -- ``(mu, sigma)`` matches both packages directly.
* LogLogistic   -- ``(alpha, beta)`` matches both packages directly.
* Gamma         -- surpyval's ``(shape, rate)`` relates to reliability's
                   ``(alpha=scale, beta=shape)`` by ``shape == beta`` and
                   ``rate == 1 / alpha``.
* Normal/Gumbel -- ``(mu, sigma)`` matches reliability directly.

surpyval reproduces every value below to at least six significant figures, so
a relative tolerance of 1e-3 is a meaningful agreement check, not a loose
convergence test.
"""

import numpy as np
import pytest

import surpyval as surv

# ---------------------------------------------------------------------------
# Fixed datasets (self-contained, no external dataset dependency)
# ---------------------------------------------------------------------------

# Dataset A -- Bofors steel tensile strength (real data, all observed).
_BOFORS_X = np.array(
    [
        40.800,
        42.075,
        43.350,
        44.625,
        45.900,
        47.175,
        48.450,
        49.725,
        51.000,
        52.275,
        53.550,
        54.825,
    ]
)
_BOFORS_N = np.array([10, 23, 48, 80, 63, 65, 45, 33, 19, 10, 3, 1])
A_X = np.repeat(_BOFORS_X, _BOFORS_N)
A_C = np.zeros_like(A_X, dtype=int)

# Dataset B -- automotive failure / right-censored survival times.
_B_FAILURES = np.array(
    [
        5248.0,
        7454.0,
        16890.0,
        17200.0,
        38700.0,
        45000.0,
        49390.0,
        69040.0,
        72280.0,
        131900.0,
    ]
)
_B_SURVIVED = np.array(
    [
        3961.0,
        4007.0,
        4734.0,
        6054.0,
        7298.0,
        10190.0,
        23060.0,
        27160.0,
        28690.0,
        37100.0,
        40060.0,
        45670.0,
        53000.0,
        67000.0,
        69630.0,
    ]
)
B_X = np.concatenate([_B_FAILURES, _B_SURVIVED])
B_C = np.concatenate(
    [np.zeros(len(_B_FAILURES), int), np.ones(len(_B_SURVIVED), int)]
)

# Interval-censored dataset (left/right bounds per observation).
INT_LEFT = np.array(
    [0.5, 1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.5, 9.0, 11.0, 13.0]
)
INT_RIGHT = np.array(
    [1.5, 2.0, 3.0, 4.0, 4.5, 6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 18.0]
)

# ---------------------------------------------------------------------------
# Expected MLE parameters (surpyval convention) -- see module docstring.
# ---------------------------------------------------------------------------

# Right-censored / observed data. Gamma is omitted for dataset A because its
# fit is degenerate (reliability reports a non-finite log-likelihood there).
XCN_EXPECTED = {
    ("A", "Weibull"): [47.60872015, 16.79055124],
    ("A", "Exponential"): [0.02160643872],
    ("A", "LogNormal"): [3.83301888, 0.05896327642],
    ("A", "LogLogistic"): [46.13467731, 29.23314098],
    ("A", "Normal"): [46.28249991, 2.746433999],
    ("A", "Gumbel"): [47.69627366, 2.868474177],
    ("B", "Weibull"): [76128.17306, 1.450233539],
    ("B", "Exponential"): [1.135439801e-05],
    ("B", "LogNormal"): [10.95718522, 1.084686899],
    ("B", "LogLogistic"): [59625.02306, 1.679125872],
    ("B", "Gamma"): [1.608266522, 1.0 / 45281.10806],
    ("B", "Normal"): [64326.73975, 36533.03734],
    ("B", "Gumbel"): [85845.5383, 34302.85524],
}

# Interval-censored data (lifelines fit_interval_censoring).
INTERVAL_EXPECTED = {
    "Weibull": [7.047279793, 1.501594688],
    "Exponential": [1.0 / 6.362860081],
    "LogNormal": [1.573762988, 0.8009838454],
    "LogLogistic": [5.047925354, 2.108218055],
}

DATASETS = {"A": (A_X, A_C), "B": (B_X, B_C)}


@pytest.mark.parametrize("key,expected", sorted(XCN_EXPECTED.items()))
def test_parametric_fit_matches_known_values(key, expected):
    dataset_name, dist_name = key
    x, c = DATASETS[dataset_name]
    model = getattr(surv, dist_name).fit(x, c)
    assert np.allclose(model.params, expected, rtol=1e-3)


@pytest.mark.parametrize(
    "dist_name,expected", sorted(INTERVAL_EXPECTED.items())
)
def test_interval_fit_matches_known_values(dist_name, expected):
    model = getattr(surv, dist_name).fit(xl=INT_LEFT, xr=INT_RIGHT)
    assert np.allclose(model.params, expected, rtol=1e-3)


@pytest.mark.parametrize("dataset_name", ["A", "B"])
def test_weibull_offset_fit_converges(dataset_name):
    x, c = DATASETS[dataset_name]
    fitted = surv.Weibull.fit(x, c, offset=True)
    assert fitted.res.success or ("Desired error" in fitted.res.message)
