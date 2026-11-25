"""Vine (pair-copula) copulas (``vine.py``).

Seeds, sample sizes and tolerances were fixed before any result was
seen.
"""

import json
import pickle

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
from surpyval.multivariate import (
    Clayton,
    Frank,
    Gaussian,
    Gumbel,
    Independence,
    StudentT,
    Vine,
)
from surpyval.multivariate.parametric.copula.copula import _U_CLIP
from surpyval.multivariate.parametric.copula.elliptical_nd import for_axes
from surpyval.multivariate.parametric.copula.vine import VineCopula
from surpyval.utils.rng import as_generator

R4 = np.array(
    [
        [1.0, 0.6, 0.3, 0.2],
        [0.6, 1.0, 0.5, 0.1],
        [0.3, 0.5, 1.0, 0.4],
        [0.2, 0.1, 0.4, 1.0],
    ]
)


def _margins(D=3):
    return [
        sp.Weibull.from_params([10.0 * (d + 1), 1.5 + 0.5 * d])
        for d in range(D)
    ]


def _partial(R, a, b, given):
    """The partial correlation of ``a`` and ``b`` given ``given``."""
    idx = [a, b] + list(given)
    P = np.linalg.inv(R[np.ix_(idx, idx)])
    return -P[0, 1] / np.sqrt(P[0, 0] * P[1, 1])


def _elliptical_vine(kind, order, family, nu=None):
    """The vine of ``family`` pair-copulas whose parameters are the
    partial correlations of ``R4`` (and, for the t, the degrees of
    freedom of each conditional pair, ``nu`` plus the number given):
    the Gaussian (or t) copula of ``R4`` written as a vine."""
    D = R4.shape[0]
    probe = VineCopula(kind, order, [[family] * (D - t) for t in range(1, D)])
    params = []
    for edge in probe.edges:
        params.append(_partial(R4, *edge.pair, edge.given))
        if nu is not None:
            params.append(nu + len(edge.given))
    return probe, np.asarray(params)


def _points(seed, D=4, size=40):
    return np.random.default_rng(seed).uniform(0.02, 0.98, (size, D))


# -- the density -------------------------------------------------------------
@pytest.mark.parametrize(
    "kind, order",
    [("d", [0, 1, 2, 3]), ("d", [2, 0, 3, 1]), ("c", [1, 3, 0, 2])],
)
def test_a_gaussian_vine_is_the_gaussian_copula(kind, order):
    vine, params = _elliptical_vine(kind, order, Gaussian)
    U = _points(4661)
    gaussian = for_axes("Gaussian", 4, "unstructured")
    expected = gaussian.pdf(U, *[R4[i, j] for i, j in gaussian.pairs])
    np.testing.assert_allclose(vine.pdf(U, *params), expected, rtol=1e-10)


def test_a_student_t_vine_is_the_student_t_copula():
    vine, params = _elliptical_vine("d", [0, 1, 2, 3], StudentT, nu=4.0)
    U = _points(4662)
    t = for_axes("StudentT", 4, "unstructured")
    expected = t.pdf(U, *[R4[i, j] for i, j in t.pairs], 4.0)
    np.testing.assert_allclose(vine.pdf(U, *params), expected, rtol=1e-8)


def test_a_vines_density_is_the_product_of_its_pair_copulas():
    """Three series: ``c01(u0, u1) c12(u1, u2) c02|1(h(u0|u1),
    h(u2|u1))`` written out with the bivariate families."""
    margins = _margins()
    model = Vine.from_params(
        [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
    )
    x = np.array([[8.0, 15.0, 25.0], [3.0, 30.0, 12.0]])
    u = np.column_stack([m.ff(x[:, d]) for d, m in enumerate(margins)])
    a = Clayton.dv(u[:, 0], u[:, 1], 2.0)  # u0 given u1
    b = Gaussian.du(u[:, 1], u[:, 2], 0.5)  # u2 given u1
    expected = (
        Clayton.pdf(u[:, 0], u[:, 1], 2.0)
        * Gaussian.pdf(u[:, 1], u[:, 2], 0.5)
        * Frank.pdf(a, b, 3.0)
    )
    for d, m in enumerate(margins):
        expected = expected * m.df(x[:, d])
    np.testing.assert_allclose(model.pdf(x), expected, rtol=1e-12)


def test_the_last_series_given_the_others_is_the_gaussian_copulas():
    for kind, order in (("d", [2, 0, 3, 1]), ("c", [1, 3, 0, 2])):
        vine, params = _elliptical_vine(kind, order, Gaussian)
        U = _points(4663)
        gaussian = for_axes("Gaussian", 4, "unstructured")
        rho = [R4[i, j] for i, j in gaussian.pairs]
        expected = gaussian.conditional_cdf(U, order[-1], order[:-1], rho)
        got = vine.conditional_cdf(U, order[-1], order[:-1], params)
        np.testing.assert_allclose(got, expected, atol=1e-10)


# -- sampling ----------------------------------------------------------------
@pytest.mark.parametrize("kind", ["d", "c"])
def test_draws_invert_the_rosenblatt_transform(kind):
    """A draw is the inverse Rosenblatt transform of uniforms: the first
    series of the order is the first uniform, and the last series given
    the others is the last."""
    families = [
        [Clayton, Gumbel.rotated(90), Frank],
        [Gaussian, Clayton.rotated(180)],
        [Frank],
    ]
    order = [3, 1, 0, 2]
    vine = VineCopula(kind, order, families)
    params = np.array([2.0, 1.8, -4.0, 0.4, 1.5, 2.0])
    U = vine.sample(500, params, random_state=4664)
    P = as_generator(4664).uniform(_U_CLIP, 1 - _U_CLIP, size=(500, 4))
    np.testing.assert_allclose(U[:, order[0]], P[:, 0], atol=1e-15)
    last = vine.conditional_cdf(U, order[-1], order[:-1], params)
    np.testing.assert_allclose(last, P[:, -1], atol=1e-8)


def test_draws_have_the_first_trees_kendall_taus():
    margins = _margins()
    model = Vine.from_params(
        [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
    )
    X = model.random(4000, random_state=4665)
    assert X.shape == (4000, 3)
    tau = model.kendall_tau()
    from scipy.stats import kendalltau

    for i, j in ((0, 1), (1, 2)):
        assert abs(kendalltau(X[:, i], X[:, j]).statistic - tau[i, j]) < 0.03
    assert np.isnan(tau[0, 2])


# -- fitting -----------------------------------------------------------------
def test_two_series_vine_is_the_bivariate_fit():
    margins = _margins(2)
    X = Clayton.from_params([2.0], margins).random(300, random_state=4666)
    vine = Vine.fit(X, margins=[sp.Weibull] * 2, families=Clayton)
    pair = Clayton.fit(X, margins=[sp.Weibull] * 2)
    assert vine.parameter_names == ["theta_0_1"]
    # the same likelihood at the same parameters
    probe = Vine.from_params(pair.params, pair.margins, families=Clayton)
    pts = np.array([[5.0, 10.0], [12.0, 30.0], [2.0, 4.0]])
    np.testing.assert_allclose(probe.pdf(pts), pair.pdf(pts), rtol=1e-12)
    # and the same fit, to the search's tolerance
    np.testing.assert_allclose(vine.params, pair.params, rtol=1e-3)
    assert abs(vine.neg_ll() - pair.neg_ll()) < 1e-6


def test_the_fit_recovers_the_parameters():
    margins = _margins()
    truth = Vine.from_params(
        [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
    )
    X = truth.random(2000, random_state=4667)
    model = Vine.fit(
        X,
        margins=[sp.Weibull] * 3,
        order=[0, 1, 2],
        families=[[Clayton, Gaussian], [Frank]],
    )
    assert model.maximum == "verified"
    se = model.standard_errors()[:3]
    assert np.all(np.abs(model.params - [2.0, 0.5, 3.0]) < 4 * se)
    lo, hi = model.param_cb("theta_0_1")
    assert lo < model.params[0] < hi


def test_the_families_are_chosen_by_aic():
    margins = _margins()
    truth = Vine.from_params(
        [3.0, 2.5, 4.0], margins, families=[[Clayton, Gumbel], [Frank]]
    )
    X = truth.random(2000, random_state=4668)
    model = Vine.fit(X, margins=[sp.Weibull] * 3, order=[0, 1, 2])
    chosen = [(e["family"], e["rotation"]) for e in model.edges]
    assert chosen == [("Clayton", 0), ("Gumbel", 0), ("Frank", 0)]


def test_the_default_order_is_by_the_strongest_taus():
    margins = _margins(4)
    # series 2 and 0 the most dependent, then 0 and 3, then 3 and 1
    families = [[Clayton, Clayton, Clayton], [Independence] * 2]
    families.append([Independence])
    truth = Vine.from_params(
        [6.0, 3.0, 1.0], margins, families=families, order=[2, 0, 3, 1]
    )
    X = truth.random(1000, random_state=4669)
    d = Vine.fit(X, margins=[sp.Weibull] * 4, families=Clayton)
    assert d.copula.order in ([2, 0, 3, 1], [1, 3, 0, 2])
    c = Vine.fit(X, margins=[sp.Weibull] * 4, structure="c", families=Clayton)
    assert c.copula.order[0] in (0, 3)


def test_truncation_makes_the_later_trees_independence_copulas():
    margins = _margins(4)
    X = Vine.from_params([2.0] * 6, margins, families=Clayton).random(
        300, random_state=4670
    )
    model = Vine.fit(X, margins=[sp.Weibull] * 4, truncate=1)
    assert len(model.params) <= 3 + 3  # (a Student-t edge has two)
    assert all(e["family"] == "Independence" for e in model.edges[3:])


def test_joint_fit_is_at_least_as_likely_as_the_two_stage_fit():
    margins = _margins()
    X = Vine.from_params(
        [2.0, 0.5, 3.0], margins, families=[[Clayton, Gaussian], [Frank]]
    ).random(200, random_state=4671)
    options = dict(
        margins=[sp.Weibull] * 3,
        order=[0, 1, 2],
        families=[[Clayton, Gaussian], [Frank]],
    )
    ifm = Vine.fit(X, **options)
    mle = Vine.fit(X, how="MLE", **options)
    assert mle.neg_ll() <= ifm.neg_ll() + 1e-6


def test_fit_from_df_is_the_fit_of_the_arrays():
    margins = _margins()
    X = Vine.from_params([2.0, 1.0, 0.5], margins, families=Clayton).random(
        200, random_state=4672
    )
    df = pd.DataFrame(X, columns=["pump", "motor", "seal"])
    options = dict(margins=[sp.Weibull] * 3, families=Clayton, order=[0, 1, 2])
    a = Vine.fit_from_df(df, x_cols=["pump", "motor", "seal"], **options)
    b = Vine.fit(X, **options)
    np.testing.assert_array_equal(a.params, b.params)


# -- refusals ----------------------------------------------------------------
def test_truncated_data_are_refused():
    # (censored rows: test_vine_censoring.py)
    X = Vine.from_params([2.0, 1.0, 0.5], _margins(), families=Clayton).random(
        50, random_state=4673
    )
    t = np.full(X.shape + (2,), np.inf)
    t[:, :, 0] = -np.inf
    t[0, 0, 1] = 1e6  # one row truncated on the right in its first series
    with pytest.raises(ValueError, match="truncated"):
        Vine.fit(X, t=t, margins=[sp.Weibull] * 3)


def test_what_a_vine_does_not_compute_is_refused_with_what_to_do():
    model = Vine.from_params([2.0, 1.0, 0.5], _margins(), families=Clayton)
    pts = [[5.0, 10.0, 15.0]]
    for call in (model.sf, model.cdf, model.cb):
        with pytest.raises(ValueError, match="random"):
            call(pts)
    with pytest.raises(ValueError, match="last series"):
        model.conditional_cdf(pts, of=0)
    for call in (model.kendall_tau_cb, model.spearman_rho_cb):
        with pytest.raises(ValueError, match="param_cb"):
            call()
    with pytest.raises(ValueError, match="not a vine"):
        Gaussian.from_params([0.5, 0.2, 0.4], _margins()).edges


def test_arguments_are_checked():
    margins = _margins()
    X = Vine.from_params([2.0, 1.0, 0.5], margins, families=Clayton).random(
        50, random_state=4674
    )
    fit = dict(margins=[sp.Weibull] * 3)
    with pytest.raises(ValueError, match="families must"):
        Vine.fit(X, families=[[Clayton], [Frank]], **fit)
    with pytest.raises(ValueError, match="order must"):
        Vine.fit(X, order=[0, 1, 1], **fit)
    with pytest.raises(ValueError, match="truncate must"):
        Vine.fit(X, truncate=3, **fit)
    with pytest.raises(ValueError, match="not both"):
        Vine.fit(X, families=Clayton, candidates=[Frank], **fit)
    with pytest.raises(ValueError, match="candidates must"):
        Vine.fit(X, candidates=["Frank"], **fit)
    with pytest.raises(ValueError, match="structure"):
        Vine.fit(X, structure="r", **fit)
    with pytest.raises(ValueError, match="families must be given"):
        Vine.from_params([2.0, 1.0, 0.5], margins)


# -- saving ------------------------------------------------------------------
def test_models_round_trip_through_json_and_pickle():
    margins = _margins()
    families = [[Clayton.rotated(180), StudentT], [Frank]]
    X = Vine.from_params(
        [2.0, 0.5, 5.0, 3.0], margins, families=families, structure="c"
    ).random(200, random_state=4675)
    model = Vine.fit(
        X,
        margins=[sp.Weibull] * 3,
        structure="c",
        order=[1, 0, 2],
        families=families,
    )
    saved = json.loads(json.dumps(model.to_dict()))
    assert saved["copula"] == "Vine"
    pts = np.array([[5.0, 10.0, 15.0], [10.0, 20.0, 30.0]])
    for restored in (sp.from_dict(saved), pickle.loads(pickle.dumps(model))):
        assert type(restored) is type(model)
        assert restored.copula.order == [1, 0, 2]
        assert restored.edges[0]["rotation"] == 180
        np.testing.assert_array_equal(restored.params, model.params)
        np.testing.assert_array_equal(restored.pdf(pts), model.pdf(pts))
        np.testing.assert_allclose(restored.aic(), model.aic())
        np.testing.assert_allclose(
            restored.standard_errors(), model.standard_errors()
        )
