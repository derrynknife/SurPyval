"""Accuracy and inputs of the discrete and Beta-family distributions.

The tail-accuracy check (``reference/test_tails.py``) holds the full grid;
these are the minimal reproductions of its issues, with values from
50-digit mpmath or a closed form, plus the distribution-level input rules
(lists and missing values).
"""

import warnings

import numpy as np
import pytest
from scipy import stats

import surpyval as sp
from surpyval.tests.conformance.registry import CASES, fitted, query
from surpyval.univariate.parametric.parametric_fitter import (
    ParametricFitter,
)

# -- Geometric with a small p (#446) ----------------------------------------


def test_geometric_keeps_a_small_p():
    p = 1e-9
    # log(1 - p) lost 8 of the 16 digits of p
    assert sp.Geometric.ff(1, p) == pytest.approx(1e-9, rel=1e-14)
    assert sp.Geometric.Hf(693147180, p) == pytest.approx(
        0.6931471803465736334, rel=1e-14
    )
    k = 69077552790  # sf is 1e-30 there
    assert sp.Geometric.sf(k, p) == pytest.approx(
        9.999999652825904029e-31, rel=1e-12
    )
    # log F where R is tiny is -R, not 0
    assert sp.Geometric.log_ff(k, p) == pytest.approx(
        -9.999999652825904029e-31, rel=1e-12
    )


def test_geometric_edges_stay_exact():
    x = np.array([-1.0, 0.0, 1.0, 2.0])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_allclose(
            sp.Geometric.sf(x, 0.3), [1.0, 1.0, 0.7, 0.49]
        )
        np.testing.assert_array_equal(
            sp.Geometric.log_ff(x[:2], 0.3), [-np.inf, -np.inf]
        )
        np.testing.assert_allclose(sp.Geometric.df(x, 0.3), [0, 0, 0.3, 0.21])


# -- BetaGeometric (#449) ---------------------------------------------------


def test_beta_geometric_at_a_large_k():
    # 50-digit mpmath: B(a, b + k) / B(a, b) at k = 1e12
    assert sp.BetaGeometric.ff(1e12, 0.001, 1000) == pytest.approx(
        0.020510503928784973635, rel=1e-12
    )
    assert sp.BetaGeometric.log_ff(1e12, 0.001, 1000) == pytest.approx(
        -3.8868181372930533946, rel=1e-12
    )
    assert sp.BetaGeometric.df(1e12, 0.001, 0.001) == pytest.approx(
        4.8609414845641305861e-16, rel=1e-10
    )
    # with a = 1 the survival is b / (b + k) exactly
    assert sp.BetaGeometric.sf(1e11, 1, 1000) == pytest.approx(
        1000 / (1000 + 1e11), rel=1e-13
    )


def test_beta_geometric_hazard_is_closed_form():
    # h(k) = a / (a + b + k - 1); the ratio df / sf(k - 1) was 0/0 = nan
    assert sp.BetaGeometric.hf(1e6, 1000, 0.001) == pytest.approx(
        1000 / (1000 + 0.001 + 1e6 - 1), rel=1e-14
    )
    np.testing.assert_array_equal(
        sp.BetaGeometric.hf(np.array([-1.0, 0.0]), 2.0, 3.0), [0.0, 0.0]
    )


def test_beta_geometric_quantile():
    assert sp.BetaGeometric.qf(1.0, 0.001, 0.001) == np.inf
    # a = b = 1: sf(k) = 1 / (1 + k), which reaches 1e-8 at k = 99999999
    assert sp.BetaGeometric.qf(0.99999999, 1, 1) == 99999999
    k = np.arange(1.0, 200.0)
    for a, b in [(2.0, 3.0), (0.01, 300.0), (50.0, 0.5)]:
        F = sp.BetaGeometric.ff(k, a, b)
        # where 1 - F is below 1e-4, F's own rounding is more than 1e-12
        # of it, and the exact quantile of that double can be k + 1
        keep = sp.BetaGeometric.sf(k, a, b) > 1e-4
        np.testing.assert_array_equal(
            sp.BetaGeometric.qf(F[keep], a, b), k[keep]
        )


def test_beta_geometric_fit_and_bounds():
    x = sp.BetaGeometric.random(300, 3.0, 5.0, random_state=1)
    model = sp.BetaGeometric.fit(x)
    assert np.all(np.isfinite(model.params))
    cb = model.cb(np.array([1.0, 3.0, 10.0]))
    assert np.all(np.isfinite(cb))
    # the fit's likelihood is the closed form's
    ll = np.sum(sp.BetaGeometric.log_df(x, *model.params))
    assert ll == pytest.approx(-model.neg_ll(), rel=1e-12)


# -- Beta4 at extreme shapes (#445) -----------------------------------------


def test_beta4_density_at_extreme_shapes():
    alpha, beta, a, b = 1000.0, 0.001, -1e6, 1e6
    z = np.array([0.99, 0.999, 0.9999])
    x = a + (b - a) * z
    want = stats.beta.pdf(z, alpha, beta) / (b - a)
    # (b - a) ** 999 raised OverflowError
    np.testing.assert_allclose(
        sp.Beta4.df(x, alpha, beta, a, b), want, rtol=1e-9
    )
    np.testing.assert_array_equal(
        sp.Beta4.df(np.array([a, b]), alpha, beta, a, b), [0.0, np.inf]
    )
    np.testing.assert_array_equal(
        sp.Beta4.hf(np.array([a, b]), alpha, beta, a, b), [0.0, np.inf]
    )


def test_beta4_density_at_the_edges():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # shape above, at and below 1 at the lower edge
        for alpha, edge in [(2.0, 0.0), (1.0, 3.0), (0.5, np.inf)]:
            got = sp.Beta4.df(0.0, alpha, 3.0, 0.0, 1.0)
            assert got == pytest.approx(edge, rel=1e-14)


def test_beta4_fit_and_bounds():
    x = sp.Beta4.random(200, 2.0, 3.0, 1.0, 5.0, random_state=2)
    model = sp.Beta4.fit(x)
    assert np.all(np.isfinite(model.params))
    np.testing.assert_allclose(
        model.df(np.array([2.0, 3.0])),
        stats.beta.pdf(
            (np.array([2.0, 3.0]) - model.params[2])
            / (model.params[3] - model.params[2]),
            model.params[0],
            model.params[1],
        )
        / (model.params[3] - model.params[2]),
        rtol=1e-10,
    )


# -- #458 and the Beta4 / Poisson tails (#442-#444, #447) -------------------
# True values from 50-digit mpmath (surpyval/tests/reference/data/
# tails_mpmath.json).


@pytest.mark.parametrize(
    "dist, fn, x, params, true",
    [
        # Beta: 1 - F was 0, the ratio df/sf NaN, B(1000, 1000) underflowed
        (
            "Beta",
            "sf",
            0.9999999985584711,
            (0.001, 3.0),
            9.999999393522811e-31,
        ),
        ("Beta", "Hf", 0.9999999985584711, (0.001, 3.0), 69.077552850469091),
        ("Beta", "df", 0.5, (1000.0, 1000.0), 35.678022291708641),
        ("Beta", "hf", 0.5, (1000.0, 1000.0), 71.356044583417283),
        ("Beta", "log_df", 0.0, (1.0, 0.001), -6.907755278982137),
        # Beta4 (#442, #443)
        (
            "Beta4",
            "sf",
            999999.9997052775,
            (0.5, 3.0, -1e6, 1e6),
            9.9999980019911524e-31,
        ),
        (
            "Beta4",
            "log_sf",
            999999.9997052775,
            (0.5, 3.0, -1e6, 1e6),
            -69.077552989622275,
        ),
        # Binomial: -log(sf) lost H of 2e-34; the log mass was -inf
        ("Binomial", "Hf", 4.0, (10.0, 0.999999), 2.0999928003717699e-34),
        ("Binomial", "log_df", 88.0, (1000.0, 1e-6), -920.99237704359311),
        ("Binomial", "hf", 88.0, (1000.0, 1e-6), 0.99998975280002216),
        # NegativeBinomial: NaN hazard, log sf capped at -708
        (
            "NegativeBinomial",
            "hf",
            1e12,
            (0.001, 1e-6),
            1.000000998998002e-6,
        ),
        (
            "NegativeBinomial",
            "log_sf",
            1e12,
            (0.001, 1e-6),
            -1000021.2088752651,
        ),
        (
            "NegativeBinomial",
            "df",
            999665687.0,
            (1000.0, 1e-6),
            1.2618122587268693e-8,
        ),
        # DiscreteWeibull: the difference of the powers cancelled
        (
            "DiscreteWeibull",
            "df",
            1e12,
            (1e-6, 0.1),
            1.7651246967462961e-106,
        ),
        (
            "DiscreteWeibull",
            "hf",
            1e12,
            (1e-6, 0.1),
            2.1896108633462356e-11,
        ),
        (
            "DiscreteWeibull",
            "log_ff",
            1653817168793.0,
            (1e-6, 0.1),
            -9.9999999998632149e-101,
        ),
        # Poisson (#442-#444)
        ("Poisson", "log_ff", 659.0, (1000.0,), -69.270192131515276),
        ("Poisson", "hf", 1e12, (1e-6,), 1.0),
        ("Poisson", "sf", 1005617.0, (1e6,), 9.9758575641932539e-9),
    ],
)
def test_tail_values(dist, fn, x, params, true):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        got = getattr(getattr(sp, dist), fn)(np.array([x]), *params)[0]
    assert got == pytest.approx(true, rel=1e-9, abs=0)


@pytest.mark.parametrize(
    "dist, params, true",
    [
        ("Poisson", (1e6,), 1008221.0),
        ("Binomial", (10.0, 1e-6), 3.0),
        ("Binomial", (1e6, 0.5), 504105.0),
        ("NegativeBinomial", (0.001, 1e-6), 26519273.0),
        ("NegativeBinomial", (1e3, 1e-6), 1282157898.0),
    ],
)
def test_discrete_quantile_just_below_one(dist, params, true):
    # scipy's ppf compares a CDF that has lost the small upper tail: it
    # was one short, or 2.5% short (#447, #458)
    assert getattr(sp, dist).qf(0.9999999999999999, *params) == true


def test_beta_quantile_below_the_doubles():
    # scipy stops at the smallest normal double (#458)
    assert sp.Beta.qf(1e-30, 0.001, 0.001) == 0.0
    assert sp.Beta.qf(1e-300, 3.0, 3.0) == pytest.approx(
        4.6415888336127789e-101, rel=1e-12
    )


# -- Discretize: qf(ff(k)) (#383) -------------------------------------------


@pytest.mark.parametrize(
    "dist, params",
    [
        (sp.Weibull, (5.0, 1.5)),
        (sp.Weibull, (20.0, 3.0)),
        (sp.Weibull, (4.22281098, 1.6285311)),
        (sp.Weibull, (10.0, 2.0)),
        (sp.LogNormal, (2.0, 1.0)),
    ],
)
def test_discretize_quantile_inverts_the_cdf(dist, params):
    D = sp.Discretize(dist)
    k = np.arange(1.0, 300.0)
    F = D.ff(k, *params)
    keep = F < 1
    np.testing.assert_array_equal(D.qf(F[keep], *params), k[keep])


def test_discretize_quantile_edges():
    D = sp.Discretize(sp.Weibull)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = D.qf(np.array([0.0, 0.3, 1.0, np.nan]), 10.0, 2.0)
    # F(6) = 1 - exp(-0.36) = 0.302
    np.testing.assert_array_equal(got, [1.0, 6.0, np.inf, np.nan])
    assert D.qf(0.3, 10.0, 2.0) == 6.0


# -- distribution-level functions of a list (#424) and of NaN (#382) --------


def _distributions():
    """One (distribution, params, query) per registered distribution."""
    seen = {}
    for case in CASES:
        if case.model_class.rsplit(".", 1)[-1] != "Parametric":
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = fitted(case)
        dist = getattr(model, "dist", None)
        params = getattr(model, "params", None)
        if not isinstance(dist, ParametricFitter) or dist.name in seen:
            continue
        if params is None or np.ndim(params) != 1:
            continue
        seen[dist.name] = (dist, [float(p) for p in params], case)
    return seen


_DISTRIBUTIONS = _distributions()
_FUNCTIONS = ("sf", "ff", "df", "hf", "Hf", "log_sf", "log_ff", "log_df")


def _calls(name):
    """(function, its query points) for every function the distribution
    has: the conformance case's own points (``qf`` at probabilities)."""
    dist, _, case = _DISTRIBUTIONS[name]
    out = []
    for fn in _FUNCTIONS + ("qf",):
        f = getattr(dist, fn, None)
        if f is None:
            continue
        q = [0.1, 0.5, 0.9] if fn == "qf" else list(query(case, "sf")[:3])
        out.append((fn, f, q))
    return out


def _value(f, q, params):
    """``f(q, *params)`` as a float array, or the exception it raises."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return np.asarray(f(q, *params), dtype=float)
        except Exception as e:  # noqa: BLE001 - compared by type
            return type(e)


@pytest.mark.parametrize("name", sorted(_DISTRIBUTIONS))
def test_functions_of_a_list_or_tuple_are_those_of_an_array(name):
    params = _DISTRIBUTIONS[name][1]
    for fn, f, q in _calls(name):
        want = _value(f, np.array(q), params)
        for kind in (list, tuple):
            got = _value(f, kind(q), params)
            if isinstance(want, type):
                # a function the distribution does not define (ExactEventTime
                # has no density) refuses a list as it refuses an array
                assert got is want, (fn, kind)
                continue
            assert not isinstance(got, type), (fn, kind, got)
            assert got.shape == want.shape, (fn, kind)
            np.testing.assert_array_equal(got, want, err_msg=fn)


def test_list_examples_from_the_issue():
    # list * int repeated the list: 6 values
    np.testing.assert_allclose(
        sp.Gamma.sf([5, 10], 8, 3), sp.Gamma.sf(np.array([5, 10]), 8, 3)
    )
    np.testing.assert_allclose(
        sp.Exponential.sf([2, 5], 1), np.exp(-np.array([2.0, 5.0]))
    )
    # list / int raised TypeError
    assert sp.Weibull.sf([5, 10], 8, 3).shape == (2,)
    assert sp.Gamma.df([5, 10], 8.0, 3.0).shape == (2,)
    assert sp.LogNormal.df((5, 10), 1.0, 1.0).shape == (2,)


@pytest.mark.parametrize("name", sorted(_DISTRIBUTIONS))
def test_a_missing_query_gives_nan_there_only(name):
    params = _DISTRIBUTIONS[name][1]
    for fn, f, q in _calls(name):
        want = _value(f, np.array(q), params)
        if isinstance(want, type):
            continue
        missing = np.array(q)
        missing[1] = np.nan
        got = _value(f, missing, params)
        assert not isinstance(got, type), (fn, got)
        assert np.isnan(got[1]), fn
        others = np.arange(len(q)) != 1
        np.testing.assert_array_equal(got[others], want[others], err_msg=fn)
        assert np.isnan(_value(f, np.nan, params)), fn


def test_degenerate_distributions_give_nan_for_nan():
    x = np.array([1.0, np.nan])
    for dist in (sp.NeverOccurs, sp.InstantlyOccurs):
        for fn in ("sf", "ff", "df", "hf", "Hf", "qf"):
            got = getattr(dist, fn)(x)
            assert np.isnan(got[1]) and not np.isnan(got[0]), (dist, fn)
