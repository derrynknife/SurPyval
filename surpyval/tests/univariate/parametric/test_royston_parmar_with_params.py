"""``RoystonParmarModel.with_params``: the same spline (knots, scale,
unit) at other coefficients.

References
----------
flexsurv 2.3.2's ``psurvspline``, ``Hsurvspline``, ``hsurvspline``,
``dsurvspline``, ``qsurvspline`` and ``mean_survspline`` at fixed gamma and
knots, printed by ``scripts/reference/reference_r_royston_parmar_given.R``
(R 4.6.2) and typed in below. The same function at the same coefficients:
``rtol=1e-12`` (largest seen 9.5e-15), except ``Hf`` on the normal scale,
``-log`` of a survival near 1 (largest seen 7.4e-11 at ``Hf = 4.8e-7``;
``rtol=1e-9``), and the mean, a quadrature in both (largest seen 7.8e-8;
``rtol=1e-6``).

Draws. The spread of ``sf(t)`` over ``with_params`` of draws from the
covariance is checked against the Wald bounds of ``cb`` (seed 20261105,
4000 draws, fixed before the results were seen). The linear predictor is
linear in the coefficients, so the 2.5% and 97.5% points of the drawn
``sf(t)`` are the ``cb`` bounds up to Monte Carlo error (about 0.001
here; seen at most 0.0013, tolerance 0.003), and the standard deviation
over the draws is the delta-method standard error (ratios seen 1.000 to
1.028, tolerance 0.05). 8 draws in 4000 were left out as not a survival
distribution (tolerance: under 1%).
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as surv
from surpyval import RoystonParmar, Weibull
from surpyval.datasets import load_rossi_static
from surpyval.tests.conformance.registry import CASE_BY_NAME
from surpyval.tests.conformance.registry_families import (
    fitted,
    predictions,
)
from surpyval.univariate.parametric.royston_parmar import (
    _lowest_slope,
    rcs_basis,
    rcs_deriv,
)

SCALES = ("hazard", "odds", "normal")
T = np.array([0.5, 2.0, 5.0, 13.0, 30.0, 60.0, 150.0])
P = np.array([0.01, 0.1, 0.5, 0.9, 0.99])


def _rossi():
    df = load_rossi_static()
    return df["week"].values, 1 - df["arrest"].values


@pytest.fixture(scope="module", params=SCALES)
def rossi_fit(request):
    x, c = _rossi()
    return RoystonParmar.fit(x, c=c, df=3, scale=request.param, unit="weeks")


def test_covariance_names_name_every_row(rossi_fit):
    names = rossi_fit.covariance_names
    assert names == ["gamma_0", "gamma_1", "gamma_2", "gamma_3"]
    assert names == rossi_fit.parameter_names
    assert len(names) == rossi_fit.covariance().shape[0]
    assert_allclose(rossi_fit._estimates(), rossi_fit.params, rtol=0)


def test_round_trip_reproduces_every_function(rossi_fit):
    m = rossi_fit
    r = m.with_params(m._estimates())
    t = np.array([0.0, 1.0, 5.0, 20.0, 52.0, 80.0, np.inf])
    for f in ("sf", "ff", "hf", "Hf", "df"):
        assert_allclose(getattr(r, f)(t), getattr(m, f)(t), rtol=1e-12)
    assert_allclose(r.qf([0.05, 0.2]), m.qf([0.05, 0.2]), rtol=1e-12)
    assert_allclose(r.mean(), m.mean(), rtol=1e-12)
    assert_allclose(
        r.random(20, random_state=4), m.random(20, random_state=4), rtol=1e-12
    )


def test_structure_kept_and_fit_left_behind(rossi_fit):
    m = rossi_fit
    before = m.params.copy()
    r = m.with_params(m.params + [0.01, 0.0, 0.0, 0.0])
    assert type(r) is type(m)
    assert r is not m
    assert_allclose(m.params, before, rtol=0)  # not mutated
    assert r.scale == m.scale
    assert_allclose(r.knots, m.knots, rtol=0)
    assert r.unit == m.unit == "week"
    assert r.maximum == "not applicable"
    assert r.n == 0 and r.n_events == 0


def test_data_based_methods_refuse_plainly(rossi_fit):
    r = rossi_fit.with_params(rossi_fit._estimates())
    for method in ("neg_ll", "aic", "aic_c", "bic"):
        with pytest.raises(ValueError, match="data"):
            getattr(r, method)()
    for call in (r.covariance, r.standard_errors, lambda: r.cb([10.0])):
        with pytest.raises(ValueError, match="built from coefficients"):
            call()
    assert "given coefficients" in repr(r)
    assert np.isnan(r.summary()["se(coef)"]).all()


def test_saved_and_restored(rossi_fit):
    r = rossi_fit.with_params(rossi_fit._estimates())
    back = surv.from_dict(r.to_dict())
    assert_allclose(back.sf(T), r.sf(T), rtol=0)
    assert back.maximum == "not applicable"
    assert back.unit == "week"
    with pytest.raises(ValueError, match="data"):
        back.aic()


@pytest.mark.parametrize(
    "bad",
    [[1.0, 2.0], [[-7.0, 1.6, 0.04, 0.0]], np.zeros(5), "abc"],
)
def test_wrong_shape_refused_naming_the_names(rossi_fit, bad):
    with pytest.raises(ValueError, match="gamma_0"):
        rossi_fit.with_params(bad)


def test_non_finite_refused(rossi_fit):
    with pytest.raises(ValueError, match="finite"):
        rossi_fit.with_params([np.nan, 1.0, 0.0, 0.0])


def test_falling_spline_refused():
    x, c = _rossi()
    m = RoystonParmar.fit(x, c=c, df=2)
    # Falling below the first knot (gamma_1 < 0)
    with pytest.raises(ValueError, match="not a survival distribution"):
        m.with_params([-7.0, -0.5, 0.0])
    # Flat below the first knot: the survival would not start at 1
    with pytest.raises(ValueError, match="stop rising"):
        m.with_params([-7.0, 0.0, 0.0])
    # Rising beyond both boundary knots but falling between the knots,
    # at log time 1.2 (two internal knots; with one, the spline's lowest
    # slope is always beyond a boundary knot)
    m = RoystonParmar.fit(x, c=c, knots=[0.0, 1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match=r"-0\.2 at time 3\.32.*decrease"):
        m.with_params([0.0, 1.0, 2.0, -3.0])
    m.with_params([0.0, 1.0, 1.0, -1.5])  # lowest slope 0.4: accepted


@pytest.mark.parametrize(
    "gamma",
    [[0, 1, 2, -3], [0, 1, 1, -1.5], [0, 0.5, 1, -1.2], [-2, 0.3, 0.5, -0.6]],
)
def test_lowest_slope_is_exact(gamma):
    knots = np.array([0.0, 1.0, 2.0, 3.0])
    gamma = np.asarray(gamma, dtype=float)
    lx = np.linspace(-1.0, 4.0, 50001)
    on_grid = rcs_deriv(lx, knots) @ gamma
    lowest, where = _lowest_slope(knots, gamma)
    assert lowest <= on_grid.min() + 1e-12
    assert on_grid.min() - lowest < 1e-8
    assert_allclose(rcs_deriv(np.array([where]), knots) @ gamma, lowest)


def test_with_refit_params_is_the_refit():
    x, c = _rossi()
    m = RoystonParmar.fit(x, c=c, df=3)
    rng = np.random.default_rng(20261105)
    i = rng.integers(0, x.size, x.size)
    refit = RoystonParmar.fit(x[i], c=c[i], knots=m.knots)
    built = m.with_params(refit.params)
    t = np.array([5.0, 20.0, 52.0])
    for f in ("sf", "hf", "Hf", "df"):
        assert_allclose(getattr(built, f)(t), getattr(refit, f)(t), rtol=0)
    assert_allclose(built.qf([0.1, 0.2]), refit.qf([0.1, 0.2]), rtol=0)


# flexsurv 2.3.2 at knots log(2, 8, 20, 60), gamma (-4, 1.3, 0.05, -0.02)
FLEXSURV = {
    "hazard": {
        "sf": [0.99258913609187049, 0.95590342301042852, 0.86433416596931123,
               0.64140887378791356, 0.34004832351126268, 0.10948539121515832,
               0.003350696798143149],
        "Hf": [0.0074384607890848973, 0.04509839298941596, 0.145795818784652,
               0.44408815678929775, 1.0786675434968516, 2.2119641521861597,
               5.6985869553283353],
        "hf": [0.019339998051620733, 0.029313955443120376,
               0.036205898703135724, 0.037300703249822563,
               0.037489090142276769, 0.038074966317611474,
               0.039236351306627497],
        "df": [0.019196671958076686, 0.028021310350053945,
               0.031293995258744184, 0.023925002062965859,
               0.012748102252843821, 0.0041686525827876698,
               0.00013146911669393603],
        "qf": [0.63024083720692148, 3.8610611204698935, 19.687004915436972,
               62.378539619647803, 122.04145098133164],
        "mean": 27.708497802945264,
    },
    "odds": {
        "sf": [0.99261646137347315, 0.95684770611844894, 0.87275584672730089,
               0.69247849952827134, 0.48107741092533907, 0.31133597780640543,
               0.14928521592222643],
        "Hf": [0.0074109318705292155, 0.044111036955112114,
               0.13609943381885722, 0.3674780890152351, 0.73172708435070455,
               1.1668826354326214, 1.9018966019641255],
        "hf": [0.019197200428969642, 0.02804899102300815,
               0.031598909779178098, 0.025829935017786444,
               0.018035154423593162, 0.01185410686843952,
               0.0058574071768102177],
        "df": [0.019055457158081161, 0.02683861271930232, 0.02757813325998618,
               0.017886674644029509, 0.0086763053957408746,
               0.0036906099529072441, 0.00087442429513451237],
        "qf": [0.63268380481186814, 4.0263223963525387, 27.900449326369927,
               233.48818481917553, 2380.0874433654003],
        "mean": 849.93353187515152,
    },
    "normal": {
        "sf": [0.99999952347134125, 0.99902882555682904, 0.97291959010148177,
               0.79152733890882865, 0.4698183369726816, 0.21363239152262548,
               0.040910351867458283],
        "Hf": [4.7652877225957657e-07, 0.00097164633862378337,
               0.027453841422326236, 0.23379085962735199,
               0.75540917607441349, 1.5434985376879691, 3.1963721460644159],
        "hf": [6.3074773505028559e-06, 0.0021326983696646333,
               0.015949343745483218, 0.030451709131133816,
               0.029427418076572384, 0.023455680804962114,
               0.014770607902029893],
        "df": [6.307474344809137e-06, 0.0021306271475130217,
               0.01551742897924316, 0.02410336029379203, 0.013825540622135065,
               0.0050108931851553969, 0.00060427076656830283],
        "qf": [3.637280334331038, 8.5480574241140594, 27.900449326369909,
               96.209825541577715, 264.58301280409012],
        "mean": 44.621300731550797,
    },
}  # fmt: skip


@pytest.mark.parametrize("scale", SCALES)
def test_matches_flexsurv_at_given_coefficients(scale):
    ref = FLEXSURV[scale]
    x = Weibull.random(300, 20, 1.5, random_state=3)
    base = RoystonParmar.fit(
        x, df=3, scale=scale, knots=np.log([2, 8, 20, 60])
    )
    m = base.with_params([-4.0, 1.3, 0.05, -0.02])
    for f in ("sf", "hf", "df"):
        assert_allclose(getattr(m, f)(T), ref[f], rtol=1e-12)
    hf_rtol = 1e-9 if scale == "normal" else 1e-12
    assert_allclose(m.Hf(T), ref["Hf"], rtol=hf_rtol)
    assert_allclose(m.qf(P), ref["qf"], rtol=1e-12)
    assert_allclose(m.mean(), ref["mean"], rtol=1e-6)


def test_draws_agree_with_the_wald_bounds():
    x, c = _rossi()
    m = RoystonParmar.fit(x, c=c, df=3)
    t = np.array([10.0, 20.0, 35.0, 52.0])
    rng = np.random.default_rng(20261105)
    draws = rng.multivariate_normal(m._estimates(), m.covariance(), 4000)
    S, left_out = [], 0
    for d in draws:
        try:
            S.append(m.with_params(d).sf(t))
        except ValueError:
            left_out += 1
    S = np.array(S)
    assert left_out < 40
    q = np.quantile(S, [0.025, 0.975], axis=0).T
    assert_allclose(q, m.cb(t), atol=3e-3)
    # The delta method on sf = exp(-exp(eta)): dS/deta = -S exp(eta)
    B = rcs_basis(np.log(t), m.knots)
    se_eta = np.sqrt(np.einsum("ij,jk,ik->i", B, m.covariance(), B))
    se_sf = m.sf(t) * np.exp(B @ m.params) * se_eta
    assert_allclose(S.std(axis=0) / se_sf, 1.0, atol=0.05)


def test_conformance_round_trip_and_wrong_length():
    # What the generic conformance check of with_params asks of every
    # model that has it and _estimates.
    case = CASE_BY_NAME["RoystonParmar"]
    model = fitted(case)
    rebuilt = model.with_params(model._estimates())
    for f, want in predictions(case, model).items():
        got = predictions(case, rebuilt)[f]
        assert_allclose(got, want, rtol=1e-12, atol=1e-300, err_msg=f)
    with pytest.raises(ValueError):
        model.with_params(np.append(model._estimates(), 0.0))
