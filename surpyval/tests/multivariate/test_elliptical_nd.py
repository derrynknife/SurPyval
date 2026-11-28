"""The Gaussian and Student-t copulas of three or more series
(``elliptical_nd.py``).

Seeds, sample sizes and tolerances were fixed before any result was
seen.
"""

import json
import pickle

import numpy as np
import pytest
from scipy import integrate
from scipy.special import ndtr
from scipy.stats import kendalltau, multivariate_normal, multivariate_t
from scipy.stats import t as student_t

import surpyval as sp
from surpyval.multivariate import (
    CopulaModel,
    Gaussian,
    MultivariateSurpyvalData,
    StudentT,
)
from surpyval.multivariate.parametric.copula.elliptical_nd import (
    EllipticalCopulaND,
    for_axes,
)

RHO = [0.5, 0.2, 0.4]
NU = 5.0


def _margins(D=3):
    return [
        sp.Weibull.from_params([10.0 * (d + 1), 1.5 + 0.5 * d])
        for d in range(D)
    ]


def _model(family=Gaussian):
    params = RHO + ([NU] if family is StudentT else [])
    return family.from_params(params, _margins())


def _dims(copula, margins, data):
    return [
        copula._prepare_dim(margins[d], *data.dimension(d))
        for d in range(data.D)
    ]


def _every_pair_of_codes(seed):
    """Two series in every pair of censoring codes, four rows each."""
    rng = np.random.default_rng(seed)
    codes = [(a, b) for a in (0, 1, -1, 2) for b in (0, 1, -1, 2)]
    c = np.repeat(np.array(codes), 4, axis=0)
    x = rng.uniform(3.0, 30.0, c.shape)
    xl = x * rng.uniform(0.5, 0.9, c.shape)
    xr = x * rng.uniform(1.1, 1.8, c.shape)
    return x, c, xl, xr


# -- two series: today's bivariate copulas ---------------------------------
@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_two_series_likelihood_is_the_bivariate_copulas(family):
    x, c, xl, xr = _every_pair_of_codes(101)
    data = MultivariateSurpyvalData(x, c=c, xl=xl, xr=xr)
    margins = _margins(2)
    params = [0.45] + ([NU] if family is StudentT else [])
    dims = _dims(family, margins, data)
    bivariate = family._pair_loglik(params, dims[0], dims[1])
    for structure in ("unstructured", "exchangeable"):
        many = for_axes(family.name, 2, structure)
        np.testing.assert_allclose(
            many._rows_loglik(params, dims), bivariate, rtol=0, atol=1e-8
        )


@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_two_series_fit_is_todays_fit(family):
    truth = family.from_params(
        [0.45] + ([NU] if family is StudentT else []), _margins(2)
    )
    X = truth.random(250, random_state=102)
    c = np.zeros(X.shape, int)
    c[::7, 1] = 1  # some right-censored values in the second series
    today = family.fit(X, c=c, margins=[sp.Weibull, sp.Weibull])
    assert type(today) is CopulaModel
    assert today.parameter_names == family.parameter_names
    # the engine of three or more series, on two
    many = for_axes(family.name, 2, "unstructured")
    again = EllipticalCopulaND.fit(
        many, X, c=c, margins=[sp.Weibull, sp.Weibull]
    )
    np.testing.assert_allclose(again.params, today.params, rtol=1e-4)
    np.testing.assert_allclose(again.neg_ll(), today.neg_ll(), atol=1e-6)
    # ... and whatever `structure` says of two series
    same = family.fit(
        X, c=c, margins=[sp.Weibull, sp.Weibull], structure="exchangeable"
    )
    np.testing.assert_array_equal(same.params, today.params)


# -- the likelihood ---------------------------------------------------------
@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_a_series_censored_to_everything_leaves_the_bivariate_likelihood(
    family,
):
    """A third series interval censored over (almost) its whole range
    carries no information, so each row's likelihood is the bivariate
    copula's of the other two, whatever their censoring: one, two and
    three censored coordinates against the bivariate code."""
    x, c, xl, xr = _every_pair_of_codes(103)
    margins = _margins()
    nu = [NU] if family is StudentT else []
    wide = margins[2].qf(np.array([1e-10, 1 - 1e-10]))
    N = x.shape[0]
    for third in range(3):
        keep = [d for d in range(3) if d != third]
        full = [np.empty((N, 3)) for _ in range(4)]
        for arr, src, fill in zip(
            full, (x, c, xl, xr), (1.0, 2, wide[0], wide[1])
        ):
            arr[:, keep] = src
            arr[:, third] = fill
        m3 = [None] * 3
        m3[keep[0]], m3[keep[1]] = margins[0], margins[1]
        m3[third] = margins[2]
        data3 = MultivariateSurpyvalData(
            full[0], c=full[1].astype(int), xl=full[2], xr=full[3]
        )
        many = for_axes(family.name, 3, "unstructured")
        R = np.eye(3)
        R[keep[0], keep[1]] = R[keep[1], keep[0]] = 0.45
        R[keep[0], third] = R[third, keep[0]] = 0.3
        R[keep[1], third] = R[third, keep[1]] = -0.2
        params = [R[i, j] for i, j in many.pairs] + nu
        ours = many._rows_loglik(params, _dims(many, m3, data3))
        data2 = MultivariateSurpyvalData(x, c=c, xl=xl, xr=xr)
        dims2 = _dims(family, margins[:2], data2)
        bivariate = family._pair_loglik([0.45] + nu, dims2[0], dims2[1])
        np.testing.assert_allclose(ours, bivariate, rtol=0, atol=1e-7)


@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_a_censored_rows_likelihood_is_the_integral_of_the_density(family):
    model = _model(family)
    point = np.array([8.0, 15.0, 22.0])
    copula = model.copula
    for censored, code in ((2, 1), (0, -1), (1, 2)):
        c = np.zeros((1, 3), int)
        c[0, censored] = code
        lo, hi = {
            1: (point[censored], np.inf),
            -1: (0.0, point[censored]),
        }.get(code, (0.7 * point[censored], 1.4 * point[censored]))
        data = MultivariateSurpyvalData(
            point[None, :],
            c=c,
            xl=np.where(c == 2, 0.7 * point, point),
            xr=np.where(c == 2, 1.4 * point, point),
        )
        ours = copula._rows_loglik(
            model.params, _dims(copula, model.margins, data)
        )

        def density(t, censored=censored):
            at = point.copy()
            at[censored] = t
            return float(model.pdf(at))

        upper = (
            hi if np.isfinite(hi) else model.margins[censored].qf(1 - 1e-13)
        )
        integral = integrate.quad(
            density, lo, upper, epsabs=0, epsrel=1e-11, limit=400
        )[0]
        np.testing.assert_allclose(np.exp(ours[0]), integral, rtol=1e-7)


def test_unit_variance_normal_scores_give_the_closed_form_estimate():
    """With known margins, the Gaussian copula's likelihood is that of a
    centred normal sample of the normal scores with a correlation
    matrix. Where the scores' mean squares are exactly 1, the
    unconstrained maximum -- their matrix of mean products -- is a
    correlation matrix, so it is the estimate."""
    rng = np.random.default_rng(104)
    L = np.linalg.cholesky(
        np.array([[1, 0.6, 0.3], [0.6, 1, -0.2], [0.3, -0.2, 1]])
    )
    z = rng.standard_normal((500, 3)) @ L.T
    z = z / np.sqrt(np.mean(z**2, axis=0))
    S = z.T @ z / len(z)
    margins = _margins()
    X = np.column_stack([m.qf(ndtr(z[:, d])) for d, m in enumerate(margins)])
    model = Gaussian.fit(X, margins=margins)
    assert model.maximum == "verified"
    np.testing.assert_allclose(model.correlation, S, atol=2e-5)
    # exchangeable: the mean of the pairs' products (the likelihood's
    # stationary point is not that in general, so only the unstructured
    # estimate has this closed form); it is at least a higher likelihood
    # than at that mean
    exchangeable = Gaussian.fit(X, margins=margins, structure="exchangeable")
    assert exchangeable.parameter_names == ["rho"]
    assert exchangeable.neg_ll() >= model.neg_ll()


# -- the model's functions ---------------------------------------------------
def test_gaussian_density_is_the_multivariate_normals():
    margins = [sp.Normal.from_params([0.0, 1.0])] * 3
    model = Gaussian.from_params(RHO, margins)
    rng = np.random.default_rng(105)
    pts = rng.normal(0.0, 1.2, (20, 3))
    ref = multivariate_normal(mean=np.zeros(3), cov=model.correlation).pdf(pts)
    np.testing.assert_allclose(model.pdf(pts), ref, rtol=1e-11)


def test_student_t_copula_density_is_the_multivariate_ts():
    model = _model(StudentT)
    rng = np.random.default_rng(106)
    z = rng.standard_t(NU, (20, 3))
    U = student_t.cdf(z, NU)
    ref = multivariate_t(shape=model.correlation, df=NU).pdf(z) / np.prod(
        student_t.pdf(z, NU), axis=1
    )
    np.testing.assert_allclose(
        model.copula.pdf(U, *model.params), ref, rtol=1e-10
    )


@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_joint_functions_agree_with_the_pairs(family):
    """Inclusion-exclusion: the joint survival from the joint CDF, the
    margins and the bivariate copulas of each pair."""
    model = _model(family)
    nu = [NU] if family is StudentT else []
    pts = np.array([[5.0, 10.0, 15.0], [10.0, 20.0, 30.0], [20.0, 12.0, 40.0]])
    F = np.column_stack([m.ff(pts[:, d]) for d, m in enumerate(model.margins)])
    pairs = 0.0
    for k, (i, j) in enumerate(model.copula.pairs):
        pair = family.from_params(
            [RHO[k]] + nu, [model.margins[i], model.margins[j]]
        )
        pairs = pairs + pair.cdf(pts[:, [i, j]])
        # ... and the pairs' dependence measures are the bivariate ones
        assert model.kendall_tau()[i, j] == pytest.approx(pair.kendall_tau())
        assert model.spearman_rho()[i, j] == pytest.approx(pair.spearman_rho())
        assert model.tail_dependence()[0][i, j] == pytest.approx(
            pair.tail_dependence()[0]
        )
    expected = 1.0 - F.sum(axis=1) + pairs - model.cdf(pts)
    np.testing.assert_allclose(model.sf(pts), expected, atol=1e-8)
    np.testing.assert_array_equal(model.ff(pts), model.cdf(pts))
    assert model.sf(pts[0]).shape == ()
    assert np.isnan(model.sf([5.0, np.nan, 15.0]))


@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_conditional_cdf_is_the_ratio_of_integrals_of_the_density(family):
    model = _model(family)
    point = np.array([8.0, 15.0, 22.0])

    def density(t):
        return float(model.pdf([t, point[1], point[2]]))

    top = model.margins[0].qf(1 - 1e-13)
    part = integrate.quad(density, 0.0, point[0], epsabs=0, epsrel=1e-11)[0]
    rest = integrate.quad(
        density, point[0], top, epsabs=0, epsrel=1e-11, limit=400
    )[0]
    np.testing.assert_allclose(
        model.conditional_cdf(point, of=0), part / (part + rest), rtol=1e-7
    )
    # given one series only: the bivariate copula's h-function
    nu = [NU] if family is StudentT else []
    pair = family.from_params(
        [RHO[1]] + nu, [model.margins[2], model.margins[0]]
    )
    np.testing.assert_allclose(
        model.conditional_cdf(point, of=0, given=[2]),
        pair.conditional_cdf([point[2], point[0]], given_dim=0),
        rtol=1e-9,
    )
    for bad in (
        {"of": 3},
        {"of": 0, "given": [0]},
        {"of": 0, "given": [1, 1]},
    ):
        with pytest.raises(ValueError, match="series"):
            model.conditional_cdf(point, **bad)


@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_draws_have_the_models_rank_correlations(family):
    model = _model(family)
    X = model.random(4000, random_state=107)
    assert X.shape == (4000, 3)
    assert model.random((2, 5), random_state=1).shape == (2, 5, 3)
    np.testing.assert_array_equal(X, model.random(4000, random_state=107))
    tau = model.kendall_tau()
    for i, j in model.copula.pairs:
        # (the standard error of Kendall's tau at n = 4000 is below 0.011)
        assert abs(kendalltau(X[:, i], X[:, j]).statistic - tau[i, j]) < 0.044


# -- fits ---------------------------------------------------------------------
def _censored_sample(model, size, seed):
    """Draws with right, left and interval censoring in every series."""
    rng = np.random.default_rng(seed)
    X = model.random(size, random_state=seed)
    c = np.zeros(X.shape, int)
    xl, xr = X.copy(), X.copy()
    for d, margin in enumerate(model.margins):
        late = X[:, d] > margin.qf(0.75)  # a test that ends
        c[late, d] = 1
        X[late, d] = margin.qf(0.75)
        early = X[:, d] < margin.qf(0.08)  # failed before the first look
        c[early, d] = -1
        X[early, d] = margin.qf(0.08)
        seen = (c[:, d] == 0) & (rng.uniform(size=size) < 0.15)
        c[seen, d] = 2  # found between two inspections
        xl[seen, d] = 0.9 * X[seen, d]
        xr[seen, d] = 1.1 * X[seen, d]
    return X, c, xl, xr


@pytest.mark.parametrize("structure", ["unstructured", "exchangeable"])
def test_gaussian_fit_recovers_the_correlations_from_censored_data(structure):
    truth = (
        _model()
        if structure == "unstructured"
        else Gaussian.from_params([0.4], _margins(), structure=structure)
    )
    X, c, xl, xr = _censored_sample(truth, 400, 108)
    assert (np.sum(c != 0, axis=1) == 3).any()  # rows censored throughout
    model = Gaussian.fit(
        X, c=c, xl=xl, xr=xr, margins=[sp.Weibull] * 3, structure=structure
    )
    assert model.maximum == "verified"
    se = model.standard_errors()
    assert np.all(se > 0) and np.all(se < 0.12)
    assert np.all(np.abs(model.params - truth.params) < 3.0 * se)
    for name in model.parameter_names:
        low, high = model.param_cb(name)
        assert (
            -1
            < low
            < model.params[model.parameter_names.index(name)]
            < high
            < 1
        )
    joint = Gaussian.fit(
        X,
        c=c,
        xl=xl,
        xr=xr,
        margins=[sp.Weibull] * 3,
        structure=structure,
        how="MLE",
    )
    assert joint.neg_ll() <= model.neg_ll() + 1e-6
    assert joint.k == len(model.params) + 6


def test_student_t_fit_recovers_its_parameters():
    truth = _model(StudentT)
    X = truth.random(600, random_state=109)
    model = StudentT.fit(X, margins=[sp.Weibull] * 3)
    assert model.maximum == "verified"
    se = model.standard_errors()
    assert np.all(np.abs(model.params - truth.params) < 3.0 * se)
    assert model.parameter_names == ["rho_0_1", "rho_0_2", "rho_1_2", "nu"]
    low, high = model.param_cb("nu")
    assert 0 < low < model.params[3] < high


def test_fit_from_df_and_data_object_agree_with_the_arrays():
    import pandas as pd

    X = _model().random(120, random_state=110)
    arrays = Gaussian.fit(X, margins=[sp.Weibull] * 3)
    df = pd.DataFrame(X, columns=["pump", "motor", "seal"])
    frame = Gaussian.fit_from_df(
        df, x_cols=["pump", "motor", "seal"], margins=[sp.Weibull] * 3
    )
    data = Gaussian.fit(
        MultivariateSurpyvalData(X, names=("pump", "motor", "seal")),
        margins=[sp.Weibull] * 3,
    )
    np.testing.assert_array_equal(frame.params, arrays.params)
    np.testing.assert_array_equal(data.params, arrays.params)


# -- refusals -----------------------------------------------------------------
# (truncation: test_elliptical_nd_truncation.py)


def test_rows_with_more_censored_series_than_the_rule_takes_are_refused():
    D = 16
    X = np.random.default_rng(115).uniform(1.0, 10.0, (30, D))
    c = np.zeros(X.shape, int)
    c[:2] = 1
    with pytest.raises(
        ValueError, match=r"up to 15 censored.*2 row\(s\) have 16.*15 of"
    ):
        Gaussian.fit(X, c=c, margins=[sp.Weibull] * D)
    c[:2, 0] = 0
    with pytest.raises(ValueError, match=r"up to 14 censored.*have 15"):
        StudentT.fit(X, c=c, margins=[sp.Weibull] * D)


@pytest.mark.parametrize("family", [Gaussian, StudentT])
def test_four_censored_series_with_one_censored_to_everything(family):
    """A fourth series interval censored over (almost) its whole range
    carries no information, so a row with it and three other censored
    series has the three-series copula's likelihood: the lattice rule of
    four censored coordinates against the three-coordinate quadrature."""
    rng = np.random.default_rng(114)
    margins = _margins(4)
    nu = [NU] if family is StudentT else []
    N = 24
    c = rng.choice([1, -1, 2], (N, 3))
    x = rng.uniform(3.0, 30.0, (N, 3))
    xl = x * rng.uniform(0.5, 0.9, x.shape)
    xr = x * rng.uniform(1.1, 1.8, x.shape)
    wide = margins[3].qf(np.array([1e-10, 1 - 1e-10]))
    data4 = MultivariateSurpyvalData(
        np.c_[x, np.full(N, 30.0)],
        c=np.c_[c, np.full(N, 2)],
        xl=np.c_[xl, np.full(N, wide[0])],
        xr=np.c_[xr, np.full(N, wide[1])],
    )
    R = np.array(
        [
            [1.0, 0.5, 0.2, 0.3],
            [0.5, 1.0, 0.4, -0.2],
            [0.2, 0.4, 1.0, 0.1],
            [0.3, -0.2, 0.1, 1.0],
        ]
    )
    four = for_axes(family.name, 4, "unstructured")
    three = for_axes(family.name, 3, "unstructured")
    ours = four._rows_loglik(
        [R[i, j] for i, j in four.pairs] + nu, _dims(four, margins, data4)
    )
    data3 = MultivariateSurpyvalData(x, c=c, xl=xl, xr=xr)
    expected = three._rows_loglik(
        [R[i, j] for i, j in three.pairs] + nu,
        _dims(three, margins[:3], data3),
    )
    np.testing.assert_allclose(ours, expected, rtol=0, atol=5e-5)


def test_arguments_are_checked():
    margins = _margins()
    X = _model().random(20, random_state=113)
    with pytest.raises(ValueError, match="structure"):
        Gaussian.fit(X, margins=[sp.Weibull] * 3, structure="ar1")
    with pytest.raises(ValueError, match="not rotated"):
        Gaussian.fit(X, margins=[sp.Weibull] * 3, rotation=90)
    with pytest.raises(ValueError, match="needs 3 margins"):
        Gaussian.fit(X, margins=[sp.Weibull] * 2)
    with pytest.raises(ValueError, match="positive-definite"):
        Gaussian.from_params([0.9, 0.9, -0.9], margins)
    with pytest.raises(ValueError, match="outside its bounds"):
        Gaussian.from_params([-0.6], margins, structure="exchangeable")
    with pytest.raises(ValueError, match="takes 4 parameter"):
        StudentT.from_params(RHO, margins)
    with pytest.raises(ValueError, match="3 columns"):
        _model().sf([[1.0, 2.0]])
    with pytest.raises(ValueError, match="3 columns"):
        Gaussian.fit(X, margins=[sp.Weibull] * 3).cb([[1.0, 2.0]])
    with pytest.raises(ValueError, match="positive-definite"):
        Gaussian.fit(X, margins=[sp.Weibull] * 3, init=[0.9, 0.9, -0.9])


def test_a_correlation_matrix_is_read_as_the_pairs_correlations():
    margins = _margins()
    R = np.array([[1, 0.5, 0.2], [0.5, 1, 0.4], [0.2, 0.4, 1]])
    np.testing.assert_array_equal(Gaussian.from_params(R, margins).params, RHO)
    t = StudentT.from_params((R, NU), margins)
    np.testing.assert_array_equal(t.params, RHO + [NU])
    np.testing.assert_array_equal(t.correlation, R)
    with pytest.raises(ValueError, match="diagonal of 1"):
        Gaussian.from_params(2 * R, margins)
    with pytest.raises(ValueError, match="one correlation"):
        Gaussian.from_params(R, margins, structure="exchangeable")


# -- saving -------------------------------------------------------------------
@pytest.mark.parametrize(
    "family, structure",
    [
        (Gaussian, "unstructured"),
        (Gaussian, "exchangeable"),
        (StudentT, "unstructured"),
    ],
)
def test_models_round_trip_through_json_and_pickle(family, structure):
    X = _model(family).random(150, random_state=114)
    model = family.fit(X, margins=[sp.Weibull] * 3, structure=structure)
    saved = json.loads(json.dumps(model.to_dict()))
    assert saved["copula"] == f"Multivariate{family.name}"
    assert (saved["axes"], saved["structure"]) == (3, structure)
    pts = np.array([[5.0, 10.0, 15.0], [10.0, 20.0, 30.0]])
    for restored in (sp.from_dict(saved), pickle.loads(pickle.dumps(model))):
        assert type(restored) is type(model)
        assert restored.copula.structure == structure
        np.testing.assert_array_equal(restored.params, model.params)
        np.testing.assert_array_equal(restored.sf(pts), model.sf(pts))
        np.testing.assert_allclose(restored.aic(), model.aic())
        np.testing.assert_allclose(
            restored.standard_errors(), model.standard_errors()
        )
        np.testing.assert_allclose(restored.cb(pts), model.cb(pts))
