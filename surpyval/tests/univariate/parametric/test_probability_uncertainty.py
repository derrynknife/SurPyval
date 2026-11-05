"""The uncertainty of the probability ``p`` of a fitted Bernoulli,
FixedEventProbability or Binomial: ``covariance()`` (the inverse observed
information, ``p (1 - p) / N``) and the Jeffreys bounds of ``param_cb``.

The bounds are checked against R 4.6.2 (stats::binom.test for
Clopper-Pearson, qbeta and binom 1.1.2's binom.confint for Jeffreys); the
values are printed by scripts/reference/reference_r_fixed_probability.R
and typed in here. Tolerance: rtol 1e-10 (both sides compute the same
beta quantiles; R prints 17 digits).
"""

import json
from typing import Any

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.univariate.parametric.distributions._single_probability import (
    probability_covariance,
)

RTOL = 1e-10

# (k, n) -> {(method, bound, alpha): value(s)}, from R (see the module
# docstring).
R_VALUES: dict[tuple[int, int], dict[tuple[str, str, float], Any]] = {
    (0, 10): {
        ("exact", "two-sided", 0.05): [0.0, 0.30849710781876083],
        ("exact", "upper", 0.1): 0.20567176527571862,
        ("jeffreys", "two-sided", 0.05): [0.0, 0.21719626750921051],
        ("jeffreys", "two-sided", 0.1): [0.0, 0.17077310824484782],
        ("jeffreys", "upper", 0.05): 0.17077310824484782,
        ("jeffreys", "upper", 0.1): 0.1235742063286801,
        ("jeffreys", "lower", 0.05): 0.0,
    },
    (1, 10): {
        ("exact", "two-sided", 0.05): [
            0.0025285785444617865,
            0.4450161170281954348,
        ],
        ("jeffreys", "two-sided", 0.05): [
            0.011011673763161143,
            0.381314771066616376,
        ],
        ("jeffreys", "two-sided", 0.1): [
            0.017891988055747841,
            0.330562803260134430,
        ],
        ("jeffreys", "upper", 0.05): 0.33056280326013443,
        ("jeffreys", "lower", 0.05): 0.017891988055747841,
    },
    (8, 10): {
        ("exact", "two-sided", 0.05): [
            0.44390453769235855,
            0.97478927367316659,
        ],
        ("jeffreys", "two-sided", 0.05): [
            0.49722550356000694,
            0.95594058644736923,
        ],
        ("jeffreys", "upper", 0.1): 0.91638484246766794,
        ("jeffreys", "lower", 0.1): 0.60517036390946466,
    },
    (10, 10): {
        ("exact", "two-sided", 0.05): [0.69150289218123917, 1.0],
        ("jeffreys", "two-sided", 0.05): [0.78280373249078938, 1.0],
        ("jeffreys", "upper", 0.05): 1.0,
        ("jeffreys", "lower", 0.05): 0.82922689175515207,
    },
    (3, 1200): {
        ("exact", "two-sided", 0.05): [
            0.00051585713223822958,
            0.00728852260949008901,
        ],
        ("jeffreys", "two-sided", 0.05): [
            0.00070459830063427222,
            0.00665669379847444800,
        ],
        ("jeffreys", "upper", 0.1): 0.004999782845462597,
        ("jeffreys", "lower", 0.1): 0.0011809947280597855,
    },
    (0, 1200): {
        ("exact", "upper", 0.1): 0.0019169811508958677,
        ("jeffreys", "two-sided", 0.05): [0.0, 0.0020906612964486327],
        ("jeffreys", "upper", 0.1): 0.0011264400083739203,
    },
    (2, 50): {
        ("jeffreys", "two-sided", 0.05): [
            0.008404035138946048,
            0.122178935587249349,
        ],
        ("jeffreys", "two-sided", 0.1): [
            0.011563048986448901,
            0.106328492500746719,
        ],
    },
    (49, 50): {
        ("exact", "two-sided", 0.05): [
            0.89353045428850009,
            0.99949377201695921,
        ],
        ("jeffreys", "two-sided", 0.05): [
            0.91031964894303596,
            0.99783350582470631,
        ],
        ("jeffreys", "upper", 0.1): 0.99414396647714265,
        ("jeffreys", "lower", 0.1): 0.93910216707981542,
    },
}

# p (1 - p) / n, from the same script.
R_VARIANCE = {
    (1, 10): 0.0090000000000000011,
    (8, 10): 0.015999999999999997,
    (3, 1200): 2.0781250000000001e-06,
    (2, 50): 0.00076799999999999991,
    (49, 50): 0.00039200000000000031,
}


def _bernoulli(k, n):
    return sp.Bernoulli.fit([1, 0], n=[k, n - k])


def _fits(k, n):
    """The three fits of k events in n trials."""
    return {
        "Bernoulli": _bernoulli(k, n),
        "FixedEventProbability": sp.FixedEventProbability.fit(
            [1, 0], n=[k, n - k]
        ),
        # n trials as one row of n, and the same as rows of 1
        "Binomial": sp.Binomial.fit([k], n_trials=n),
    }


_REF_CASES = [
    (kn, key, value)
    for kn, refs in R_VALUES.items()
    for key, value in refs.items()
]


@pytest.mark.parametrize("kn,key,expected", _REF_CASES)
def test_bounds_match_r(kn, key, expected):
    method, bound, alpha = key
    for name, model in _fits(*kn).items():
        got = model.param_cb("p", alpha_ci=alpha, bound=bound, method=method)
        np.testing.assert_allclose(
            got, np.atleast_1d(expected), rtol=RTOL, atol=0, err_msg=name
        )


@pytest.mark.parametrize("kn,expected", list(R_VARIANCE.items()))
def test_covariance_matches_r(kn, expected):
    fits = _fits(*kn)
    for name in ("Bernoulli", "FixedEventProbability"):
        cov = fits[name].covariance()
        assert cov.shape == (1, 1), name
        np.testing.assert_allclose(cov, [[expected]], rtol=RTOL)
        np.testing.assert_allclose(
            fits[name].standard_errors(), [np.sqrt(expected)], rtol=RTOL
        )
    # The Binomial's known n has a zero row and column, as a held
    # parameter has.
    cov = fits["Binomial"].covariance()
    np.testing.assert_allclose(
        cov, [[0.0, 0.0], [0.0, expected]], rtol=RTOL, atol=0
    )
    np.testing.assert_allclose(
        fits["Binomial"].standard_errors(), [0.0, np.sqrt(expected)], rtol=RTOL
    )


def test_issue_example():
    b = sp.Bernoulli.fit(np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 1]))
    np.testing.assert_allclose(b.covariance(), [[0.8 * 0.2 / 10]], rtol=1e-15)


def test_binomial_rows_of_different_sizes_pool_the_trials():
    lots = sp.Binomial.fit([1, 0, 3], n_trials=[20, 50, 80])
    p = 4 / 150
    np.testing.assert_allclose(
        lots.covariance(), [[0.0, 0.0], [0.0, p * (1 - p) / 150]], rtol=1e-14
    )


def test_counts_are_frequencies():
    """``n`` counts outcomes: the variance is over their total."""
    counted = sp.Bernoulli.fit([1, 0], n=[3, 7])
    listed = sp.Bernoulli.fit([1] * 3 + [0] * 7)
    np.testing.assert_allclose(counted.covariance(), listed.covariance())


@pytest.mark.parametrize("k", [0, 20])
def test_variance_at_the_edge_is_undefined(k):
    """No events (or all) would give variance 0, as if p were known; it is
    nan, the standard error nan, and the exact bounds still answer."""
    for name, model in _fits(k, 20).items():
        cov = model.covariance()
        assert np.isnan(cov[-1, -1]), name
        assert np.isnan(model.standard_errors()[-1]), name
        assert np.isfinite(model.param_cb("p")).all(), name
    assert np.isnan(probability_covariance(k, 20))


def test_model_from_params_has_no_covariance():
    for dist in (sp.Bernoulli, sp.FixedEventProbability):
        with pytest.raises(ValueError, match="no parameter covariance"):
            dist.from_params(0.3).covariance()
    with pytest.raises(ValueError, match="no parameter covariance"):
        sp.Binomial.from_params([5, 0.3]).covariance()


@pytest.mark.parametrize("k", [0, 3, 10])
def test_covariance_survives_a_dict_and_json(k):
    for name, model in _fits(k, 10).items():
        for restored in (
            sp.Parametric.from_dict(model.to_dict()),
            sp.Parametric.from_dict(json.loads(model.to_json())),
        ):
            np.testing.assert_array_equal(
                restored.covariance(), model.covariance(), err_msg=name
            )


def test_dict_without_covariance_gets_it_from_the_counts():
    """A dict saved before the covariance was stored (0.23 to 0.26) has
    the counts, from which the restored model computes it."""
    for name, model in _fits(3, 10).items():
        old = model.to_dict()
        del old["covariance"]
        restored = sp.Parametric.from_dict(old)
        np.testing.assert_array_equal(
            restored.covariance(), model.covariance(), err_msg=name
        )


def test_fits_refuse_weights():
    """These fits take no sampling weights (counts go in ``n``), so no
    variance is given for weighted data: ``w`` raises WeightedRefusal."""
    from surpyval.utils.weights import WeightedRefusal

    for dist in (sp.Bernoulli, sp.FixedEventProbability):
        with pytest.raises(WeightedRefusal):
            dist.fit([1, 0, 1], w=[1.0, 2.0, 1.0])
    with pytest.raises(WeightedRefusal):
        sp.Binomial.fit([1, 2], n_trials=5, w=[1.0, 2.0])


def test_jeffreys_is_the_posterior_interval_inside():
    """Inside (0 < k < n) the Jeffreys bounds are the equal-tailed
    credible interval of ``fit_bayes`` with the Beta(1/2, 1/2) prior."""
    outcomes = [1, 1, 0, 1, 1, 1, 0, 1, 1, 1]
    model = sp.Bernoulli.fit(outcomes)
    bayes = sp.Bernoulli.fit_bayes(
        outcomes, priors={"p": sp.Beta.from_params([0.5, 0.5])}
    )
    np.testing.assert_allclose(
        model.param_cb("p", method="jeffreys"),
        bayes.param_cri("p"),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        model.param_cb("p", method="jeffreys"),
        stats.beta(8.5, 2.5).ppf([0.025, 0.975]),
        rtol=1e-12,
    )


def test_jeffreys_is_inside_exact():
    for k in range(0, 21):
        model = _bernoulli(k, 20)
        lo_j, hi_j = model.param_cb("p", method="jeffreys")
        lo_e, hi_e = model.param_cb("p")
        assert lo_e <= lo_j <= k / 20 <= hi_j <= hi_e


def test_unknown_method_names_the_options():
    with pytest.raises(ValueError, match="jeffreys"):
        _bernoulli(3, 10).param_cb("p", method="agresti")
