"""Fitting and building copula models: the starting value and ``init``,
the margins under IFM and MLE, and the checks on data and parameters.
"""

import warnings

import numpy as np
import pytest

import surpyval as surv
from surpyval import LogNormal, Weibull
from surpyval.multivariate import (
    Clayton,
    Copula,
    Frank,
    Gaussian,
    Gumbel,
    Independence,
)
from surpyval.multivariate.parametric.data import MultivariateSurpyvalData

MARGINS = [Weibull.from_params([10.0, 2.0]), LogNormal.from_params([2.5, 0.5])]


class AliMikhailHaq(Copula):
    name = "Ali-Mikhail-Haq"
    bounds = ((-1, 1),)
    parameter_names = ["theta"]

    def cdf(self, u, v, theta):
        return u * v / (1 - theta * (1 - u) * (1 - v))


AMH = AliMikhailHaq()


@pytest.fixture(scope="module")
def amh_data():
    return AMH.from_params(0.6, margins=MARGINS).random(400, random_state=0)


# -- starting value ---------------------------------------------------------


def test_default_start_is_strictly_inside_the_bounds(amh_data):
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the old start warned (arctanh(1))
        model = AMH.fit(amh_data, margins=[Weibull, LogNormal])
    theta = model.params[0]
    # the old fit returned exactly the start, theta = 1
    assert -1 < theta < 1
    assert theta != 1.0
    assert theta == pytest.approx(0.6, abs=0.3)


@pytest.mark.parametrize(
    "bounds, expected",
    [
        (((0, None),), [1.0]),  # 1 inside: the historical default is kept
        (((None, None),), [1.0]),
        (((-1, 1),), [0.0]),
        (((1, None),), [2.0]),
        (((None, 1),), [0.0]),
        (((2, 3),), [2.5]),
        (((0, None), (-1, 1)), [1.0, 0.0]),
    ],
)
def test_base_init_theta(bounds, expected):
    class Fam(Copula):
        pass

    fam = Fam()
    fam.bounds = bounds
    fam.parameter_names = ["p%d" % i for i in range(len(bounds))]
    np.testing.assert_array_equal(fam._init_theta([]), expected)


def test_public_init_argument(amh_data):
    a = AMH.fit(amh_data, margins=[Weibull, LogNormal], init=0.3)
    b = AMH.fit(amh_data, margins=[Weibull, LogNormal], init=[0.3])
    assert a.params[0] == b.params[0]
    assert a.params[0] == pytest.approx(
        AMH.fit(amh_data, margins=[Weibull, LogNormal]).params[0], abs=1e-3
    )
    # also used by the built-in families, and by MLE (through its IFM start)
    data = Clayton.from_params(2.0, margins=MARGINS).random(
        300, random_state=1
    )
    ifm = Clayton.fit(data, margins=[Weibull, LogNormal], init=5.0)
    ref = Clayton.fit(data, margins=[Weibull, LogNormal])
    assert ifm.params[0] == pytest.approx(ref.params[0], rel=1e-3)
    mle = Clayton.fit(data, margins=[Weibull, LogNormal], how="MLE", init=5.0)
    assert mle.params[0] == pytest.approx(ref.params[0], rel=0.2)


@pytest.mark.parametrize("bad", [1.0, -1.0, 2.0, np.nan, [0.1, 0.2]])
def test_init_outside_bounds_raises(amh_data, bad):
    with pytest.raises(ValueError, match="init"):
        AMH.fit(amh_data, margins=[Weibull, LogNormal], init=bad)


def test_subclass_start_on_bound_raises_not_silent(amh_data):
    class Stuck(AliMikhailHaq):
        def _init_theta(self, dims):
            return np.array([1.0])  # on the bound

    with pytest.raises(ValueError, match="strictly inside"):
        Stuck().fit(amh_data, margins=[Weibull, LogNormal])


def test_independence_accepts_empty_init():
    data = Independence.from_params([], margins=MARGINS).random(
        100, random_state=0
    )
    m = Independence.fit(data, margins=[Weibull, LogNormal], init=[])
    assert m.params.size == 0
    with pytest.raises(ValueError, match="init"):
        Independence.fit(data, margins=[Weibull, LogNormal], init=[0.5])


# ---------------------------------------------------------------------------
# Clayton fitted near independence; IFM margins use the counts
# and the truncation; the data checks.
# ---------------------------------------------------------------------------


def _negatively_dependent(n=500, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    z[:, 1] = -0.6 * z[:, 0] + 0.8 * z[:, 1]
    return np.exp(z + 2)


def test_clayton_near_zero_is_the_independence_copula():
    u = np.array([0.1, 0.4, 0.8])
    v = np.array([0.3, 0.9, 0.2])
    for theta in (1e-21, 1e-12):
        assert np.allclose(Clayton.cdf(u, v, theta), u * v)
        assert np.allclose(Clayton.pdf(u, v, theta), 1.0)
        assert np.allclose(Clayton.du(u, v, theta), v)
    # unchanged away from zero
    theta = 2.0
    direct = (u**-theta + v**-theta - 1) ** (-1 / theta)
    assert np.allclose(Clayton.cdf(u, v, theta), direct, rtol=1e-12)


def test_clayton_on_negative_dependence_is_no_better_than_independence():
    x = _negatively_dependent()
    margins = [surv.LogNormal, surv.LogNormal]
    clayton = Clayton.fit(x, margins=margins)
    dims = [
        Clayton._prepare_dim(clayton.margins[d], *clayton.data.dimension(d))
        for d in range(2)
    ]
    ll_i = -Independence.neg_ll([], dims, clayton.data.n)
    # at theta ~ 1e-21 the old closed form gave C = 1 and a density of
    # 1 / (u v): a likelihood ~1000 units above independence
    for theta in (1e-21, 1e-12, clayton.params[0]):
        ll_c = -Clayton.neg_ll([theta], dims, clayton.data.n)
        assert ll_c == pytest.approx(ll_i, abs=1e-3)
    assert Frank.fit(x, margins=margins).params[0] < 0


def test_ifm_margins_use_the_counts():
    x = _negatively_dependent(60)
    n = np.random.default_rng(1).integers(1, 6, size=60)
    weighted = Clayton.fit(x, n=n, margins=[surv.Weibull, surv.Weibull])
    expanded = Clayton.fit(
        np.repeat(x, n, axis=0), margins=[surv.Weibull, surv.Weibull]
    )
    for a, b in zip(weighted.margins, expanded.margins):
        assert np.allclose(a.params, b.params, rtol=1e-4)
    assert weighted.params[0] == pytest.approx(expanded.params[0], rel=1e-3)


def test_ifm_margins_use_the_truncation():
    rng = np.random.default_rng(3)
    x = rng.weibull(2.0, size=(4000, 2)) * 10
    keep = (x[:, 0] > 6) & (x[:, 1] > 6)
    x = x[keep]
    t = np.empty((len(x), 2, 2))
    t[..., 0] = 6.0
    t[..., 1] = np.inf
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        model = Independence.fit(x, t=t, margins=[surv.Weibull, surv.Weibull])
    for margin in model.margins:
        assert margin.params == pytest.approx([10.0, 2.0], rel=0.08)


def test_data_checks():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    with pytest.raises(ValueError, match="need xl and xr"):
        MultivariateSurpyvalData(x, c=np.array([[2, 0], [0, 0], [0, 0]]))
    # one row of codes applies to every row
    data = MultivariateSurpyvalData(x, c=np.array([0, 1]))
    assert (data.c == [[0, 1]] * 3).all()
    data = MultivariateSurpyvalData(x, c=[0, 1])
    assert (data.c == [[0, 1]] * 3).all()


# ---------------------------------------------------------------------------
# ``from_params`` validates the parameters and the margins;
# ``how="MLE"`` re-fits an already-fitted margin with its own
# configuration (offset, limited failure, zero inflation,
# fixed parameters), and refuses a non-parametric margin.
# ---------------------------------------------------------------------------


WEIBULL_MARGINS = [
    surv.Weibull.from_params([10, 2]),
    surv.Weibull.from_params([20, 3]),
]


@pytest.mark.parametrize(
    "family, params, match",
    [
        (Clayton, [2.0, 3.0], "takes 1 parameter"),
        (Clayton, [-2.0], "outside its bounds"),
        (Gumbel, [0.5], "outside its bounds"),
        (Gaussian, [1.5], "outside its bounds"),
        (Frank, [np.nan], "outside its bounds"),
    ],
)
def test_from_params_validates_parameters(family, params, match):
    with pytest.raises(ValueError, match=match):
        family.from_params(params, WEIBULL_MARGINS)


def test_from_params_validates_margins():
    with pytest.raises(ValueError, match="needs 2 margins"):
        Clayton.from_params([2.0], WEIBULL_MARGINS[:1])
    with pytest.raises(ValueError, match="not a fitted univariate model"):
        Clayton.from_params([2.0], [WEIBULL_MARGINS[0], 3.0])


def test_from_params_accepts_gumbel_independence():
    model = Gumbel.from_params([1.0], WEIBULL_MARGINS)
    assert model.kendall_tau() == 0.0


def _clayton_sample(shift=0.0):
    M = [
        surv.Weibull.from_params([10, 2]),
        surv.LogNormal.from_params([2.5, 0.5]),
    ]
    X = Clayton.from_params([2.0], M).random(300, random_state=1)
    return np.column_stack([X[:, 0] + shift, X[:, 1]])


def test_mle_keeps_an_offset_margin():
    X = _clayton_sample(shift=5.0)
    margins = [
        surv.Weibull.fit(X[:, 0], offset=True),
        surv.LogNormal.fit(X[:, 1]),
    ]
    ifm = Clayton.fit(X, margins=margins)
    mle = Clayton.fit(X, margins=margins, how="MLE")
    assert mle.margins[0].offset
    assert 0 < mle.margins[0].gamma < X[:, 0].min()
    assert mle.neg_ll() <= ifm.neg_ll()
    assert mle.k == 1 + 3 + 2


def test_mle_keeps_limited_failure_zero_inflation_and_fixed():
    X = _clayton_sample()
    c = np.zeros(X.shape, dtype=int)
    over = X[:, 0] > 12
    c[over, 0] = 1
    Xc = X.copy()
    Xc[over, 0] = 12
    lfp = [surv.Weibull.fit(Xc[:, 0], c=c[:, 0], lfp=True), surv.LogNormal]
    mle = Clayton.fit(Xc, c=c, margins=lfp, how="MLE")
    assert mle.margins[0].lfp and 0 < mle.margins[0].p < 1

    Xz = X.copy()
    Xz[:20, 0] = 0
    zi = [surv.Weibull.fit(Xz[:, 0], zi=True), surv.LogNormal]
    mle = Clayton.fit(Xz, margins=zi, how="MLE")
    assert mle.margins[0].zi and 0 < mle.margins[0].f0 < 1

    fixed = [surv.Weibull.fit(X[:, 0], fixed={"beta": 2.0}), surv.LogNormal]
    mle = Clayton.fit(X, margins=fixed, how="MLE")
    assert mle.margins[0].params[1] == 2.0
    assert mle.k == 1 + 1 + 2


def test_mle_refuses_a_non_parametric_margin():
    X = _clayton_sample()
    km = surv.KaplanMeier.fit(X[:, 0])
    with pytest.raises(ValueError, match="how='IFM'"):
        Clayton.fit(X, margins=[km, surv.LogNormal], how="MLE")
