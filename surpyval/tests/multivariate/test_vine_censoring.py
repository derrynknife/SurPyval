"""Censored rows in a vine copula (``vine.py``): censoring in the ends of
its order (a D-vine's first and last series, a C-vine's last two).

The reference is the Gaussian (and Student-t) copula: a vine of Gaussian
pair-copulas whose parameters are the partial correlations of a
correlation matrix is the Gaussian copula of that matrix, so the two
likelihoods of any censored row agree; the elliptical copula's is in
closed form for one or two censored series (``elliptical_nd.py``). A vine
of two series is its pair-copula, whose censored fit is the bivariate
family's.

Seeds, sample sizes and tolerances were fixed before any result was
seen.
"""

import numpy as np
import pytest

import surpyval as sp
from surpyval.multivariate import (
    Clayton,
    Frank,
    Gaussian,
    Gumbel,
    StudentT,
    Vine,
)
from surpyval.multivariate.parametric.copula.elliptical_nd import for_axes
from surpyval.multivariate.parametric.copula.vine import VineCopula
from surpyval.multivariate.parametric.data import MultivariateSurpyvalData

R4 = np.array(
    [
        [1.0, 0.6, 0.3, 0.2],
        [0.6, 1.0, 0.5, 0.1],
        [0.3, 0.5, 1.0, 0.4],
        [0.2, 0.1, 0.4, 1.0],
    ]
)


def _margins(D):
    return [
        sp.Weibull.from_params([10.0 * (d + 1), 1.5 + 0.5 * d])
        for d in range(D)
    ]


def _partial(R, a, b, given):
    idx = [a, b] + list(given)
    P = np.linalg.inv(R[np.ix_(idx, idx)])
    return -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])


def _elliptical_vine(kind, order, family, nu=None):
    D = R4.shape[0]
    probe = VineCopula(kind, order, [[family] * (D - t) for t in range(1, D)])
    params = []
    for edge in probe.edges:
        params.append(_partial(R4, *edge.pair, edge.given))
        if nu is not None:
            params.append(nu + len(edge.given))
    return probe, np.asarray(params)


def _censored_rows(ends, seed, size=60):
    """Rows of the Gaussian copula of ``R4`` (Weibull margins) with each
    end censored in some rows: right, left and interval, alone and
    together."""
    margins = _margins(4)
    truth = Gaussian.from_params(
        [R4[i, j] for i, j in for_axes("Gaussian", 4, "unstructured").pairs],
        margins,
    )
    X = truth.random(size, random_state=seed)
    c = np.zeros(X.shape, int)
    xl, xr = X.copy(), X.copy()
    a, b = ends
    rng = np.random.default_rng(seed + 1)
    kinds = rng.integers(0, 4, (size, 2))  # 0 observed, 1 right, 2 left, 3 int
    for col, axis in enumerate((a, b)):
        for r in range(size):
            kind = kinds[r, col]
            if kind == 1:
                c[r, axis] = 1
                X[r, axis] *= 0.6
            elif kind == 2:
                c[r, axis] = -1
                X[r, axis] *= 1.7
            elif kind == 3:
                c[r, axis] = 2
                xl[r, axis] = 0.5 * X[r, axis]
                xr[r, axis] = 1.5 * X[r, axis]
    xl = np.where(c == 2, xl, X)
    xr = np.where(c == 2, xr, X)
    data = MultivariateSurpyvalData(X, c=c, xl=xl, xr=xr)
    return margins, data


def _dims(copula, margins, data):
    return [
        copula._prepare_dim(margins[d], *data.dimension(d))
        for d in range(data.D)
    ]


# -- the likelihood -----------------------------------------------------------
@pytest.mark.parametrize(
    "kind, order",
    [("d", [0, 1, 2, 3]), ("d", [2, 0, 3, 1]), ("c", [1, 3, 0, 2])],
)
def test_a_censored_gaussian_vine_is_the_gaussian_copula(kind, order):
    vine, params = _elliptical_vine(kind, order, Gaussian)
    margins, data = _censored_rows(vine.ends, seed=4701)
    assert np.sum(np.all(data.c[:, list(vine.ends)] != 0, axis=1)) > 5
    gaussian = for_axes("Gaussian", 4, "unstructured")
    rho = [R4[i, j] for i, j in gaussian.pairs]
    ours = vine._rows_loglik(params, _dims(vine, margins, data))
    expected = gaussian._rows_loglik(rho, _dims(gaussian, margins, data))
    np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-8)


def test_a_censored_student_t_vine_is_the_student_t_copula():
    vine, params = _elliptical_vine("d", [3, 1, 0, 2], StudentT, nu=4.0)
    margins, data = _censored_rows(vine.ends, seed=4702)
    t = for_axes("StudentT", 4, "unstructured")
    theta = [R4[i, j] for i, j in t.pairs] + [4.0]
    ours = vine._rows_loglik(params, _dims(vine, margins, data))
    expected = t._rows_loglik(theta, _dims(t, margins, data))
    np.testing.assert_allclose(ours, expected, rtol=0, atol=1e-7)


def test_fully_observed_rows_are_unchanged():
    vine, params = _elliptical_vine("d", [0, 1, 2, 3], Gaussian)
    margins, data = _censored_rows(vine.ends, seed=4703)
    dims = _dims(vine, margins, data)
    whole = np.all(data.c == 0, axis=1)
    U = np.column_stack([d["u"] for d in dims])
    logf = np.column_stack([d["logf"] for d in dims])
    expected = vine._log_density(U[whole], params) + logf[whole].sum(axis=1)
    np.testing.assert_array_equal(
        vine._rows_loglik(params, dims)[whole], expected
    )


# -- fitting ------------------------------------------------------------------
def test_a_censored_two_series_vine_is_the_bivariate_fit():
    margins = _margins(2)
    X = Clayton.from_params(2.0, margins).random(300, random_state=4704)
    c = np.zeros(X.shape, int)
    limit = np.array([12.0, 18.0])
    c[X > limit] = 1
    X = np.minimum(X, limit)
    assert 0.2 < c.mean() < 0.5
    vine = Vine.fit(
        X, c=c, margins=[sp.Weibull] * 2, families=Clayton, order=[0, 1]
    )
    pair = Clayton.fit(X, c=c, margins=[sp.Weibull] * 2)
    np.testing.assert_allclose(vine.params, pair.params, rtol=1e-6)
    np.testing.assert_allclose(
        vine.log_likelihood, pair.log_likelihood, rtol=1e-9
    )


def _censored_draws(seed, size=500):
    margins = _margins(3)
    truth = Vine.from_params(
        [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
    )
    X = truth.random(size, random_state=seed)
    c = np.zeros(X.shape, int)
    limit = np.array([14.0, np.inf, 40.0])  # ends 0 and 2 censored
    c[X > limit] = 1
    X = np.minimum(X, limit)
    return X, c


def test_the_fit_recovers_the_parameters_with_censored_ends():
    X, c = _censored_draws(4705)
    assert 0.1 < c[:, 0].mean() < 0.5 and 0.1 < c[:, 2].mean() < 0.5
    model = Vine.fit(
        X,
        c=c,
        margins=[sp.Weibull] * 3,
        order=[0, 1, 2],
        families=[[Clayton, Gaussian], [Frank]],
    )
    se = model.standard_errors()
    assert np.all(np.abs(model.params - [2.0, 0.5, 3.0]) < 3 * se), (
        model.params,
        se,
    )


def test_the_joint_fit_is_at_least_as_likely_as_the_two_stage_fit():
    X, c = _censored_draws(4706, size=200)
    options = dict(
        c=c,
        margins=[sp.Weibull] * 3,
        order=[0, 1, 2],
        families=[[Clayton, Gaussian], [Frank]],
    )
    ifm = Vine.fit(X, **options)
    mle = Vine.fit(X, how="MLE", **options)
    assert mle.log_likelihood >= ifm.log_likelihood - 1e-6


def test_the_default_order_puts_the_censored_series_at_the_ends():
    X, c = _censored_draws(4707, size=300)
    d_vine = Vine.fit(X, c=c, margins=[sp.Weibull] * 3)
    assert set(d_vine.copula.ends) == {0, 2}
    c_vine = Vine.fit(X, c=c, margins=[sp.Weibull] * 3, structure="c")
    assert c_vine.copula.order[0] == 1
    assert set(c_vine.copula.ends) == {0, 2}


def test_the_families_are_chosen_on_the_censored_likelihood():
    X, c = _censored_draws(4708, size=600)
    model = Vine.fit(
        X,
        c=c,
        margins=[sp.Weibull] * 3,
        order=[0, 1, 2],
        candidates=[Gaussian, Clayton, Gumbel, Frank],
    )
    assert model.edges[0]["family"] == "Clayton"


# -- refusals -----------------------------------------------------------------
def test_censoring_outside_the_ends_is_refused_with_what_to_do():
    X, c = _censored_draws(4709, size=100)
    with pytest.raises(ValueError, match=r"series 1, 2 in the order.*ends"):
        Vine.fit(
            X, c=c, margins=[sp.Weibull] * 3, order=[0, 1, 2], structure="c"
        )
    with pytest.raises(ValueError, match=r"series 0, 1 in the order"):
        Vine.fit(X, c=c, margins=[sp.Weibull] * 3, order=[0, 2, 1])
    c3 = c.copy()
    c3[:3, 1] = 1
    with pytest.raises(
        ValueError, match=r"at most two.*\[0, 1, 2\].*Gaussian"
    ):
        Vine.fit(X, c=c3, margins=[sp.Weibull] * 3)
