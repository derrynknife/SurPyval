"""Fits with sampling weights ``w`` against R ``survival``, ``survey`` and
lifelines.

The data are ``_helpers.weighted_reference_data()`` (forty rows with
weights from 0.34 to 2.96, ties among the events, late entry on a third of
the rows). The values were computed once and are stored here:

- R 4.5.3, ``survival`` 3.8.12 and ``survey`` 4.5:
  ``survfit(Surv(x, d) ~ 1, weights = w, robust = TRUE, stype, ctype)``
  (``stype = 1, ctype = 1`` Kaplan-Meier, ``2, 1`` Nelson-Aalen, ``2, 2``
  Fleming-Harrington) at ``times = c(1, 2.2, 3.7, 6.1, 8.3)``, with
  ``std.err / surv`` (the standard error of ``log S``) and, for
  Nelson-Aalen, ``std.chaz`` (of the cumulative hazard; R's ``std.err``
  for ``stype = 2, ctype = 1`` is not on that scale);
  ``survfit(Surv(tl, x, d) ~ 1, ..., id = id)`` for the late entries;
  ``survreg(Surv(x, d) ~ 1, weights = w, robust = TRUE, dist = ...)`` and
  ``~ z1 + z2`` (``coef``, ``log(scale)``, ``sqrt(diag(vcov))``);
  ``coxph(Surv(x, d) ~ z1 + z2, weights = w, robust = TRUE, ties = ...)``,
  with ``Surv(tl, x, d)`` and ``id = id``, and with ``+ cluster(cl)``
  (``cl = (id - 1) %/% 4``), and ``survreg(Surv(x, d) ~ z1 + z2 +
  cluster(cl), weights = w, robust = TRUE, dist = ...)`` (R 4.4.2,
  ``survival`` 3.8.3; its unclustered values equal 3.8.12's above);
  ``svycoxph`` on ``svydesign(ids = ~1,
  weights = ~w)``, whose standard errors are R ``survival``'s times
  ``sqrt(n / (n - 1))``, the survey package's finite-sample factor, which
  SurPyval (as ``survival``) does not apply.
- lifelines 0.30.3 for the point estimates (``KaplanMeierFitter``,
  ``WeibullFitter`` and ``CoxPHFitter`` with weights). Its robust standard
  errors with weights follow other conventions and are not compared.

R's ``print(survfit(...))`` gives ``se(rmean)`` from the classical
formula even with ``robust = TRUE``, so the robust variance of the
restricted mean is checked against an independent numerical
infinitesimal jackknife instead.

Tolerances: the non-parametric estimates and their variances are closed
forms, equal to rounding (``rtol=1e-10``). The fits are optimisations; the
coefficients agree to the optimisers' tolerance (``atol=2e-5`` for
``survreg``, whose convergence criterion is on the log-likelihood,
``1e-7`` for Cox), and the standard errors, smooth functions of them,
to ``rtol=2e-4`` and ``1e-6``.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data

TIMES = np.array([1, 2.2, 3.7, 6.1, 8.3])
CLOSED = dict(rtol=1e-10, atol=1e-12)

R_SURVFIT = {
    "KaplanMeier": (
        [0.923161189358372, 0.776525821596244, 0.538054872467689,
         0.441984942284838, 0.267143908812717],
        [0.0425276576222045, 0.0916447161673752, 0.1635459209094220,
         0.2038634103678010, 0.3471761338708973],
    ),
    "NelsonAalen": (
        [0.924324874876702, 0.780088308368704, 0.547374712686876,
         0.455424805354952, 0.286873256063502],
        # std.chaz
        [0.0411306096341895, 0.0880588734666975, 0.1537705223373366,
         0.1864578438579789, 0.2981738039526620],
    ),
    "FlemingHarrington": (
        [0.923904286727219, 0.779733351029604, 0.545463478547742,
         0.451090426213569, 0.284143019496056],
        [0.0416622759000732, 0.0883041754537210, 0.1557894238812997,
         0.1924831942107194, 0.3017495188139375],
    ),
}  # fmt: skip
R_SURVFIT_TRUNCATED = (
    [0.912915846230892, 0.767907853703625, 0.532083481064552,
     0.437079745399302, 0.264179116703005],
    [0.0483340686320875, 0.0957344234784129, 0.1650857868245073,
     0.2059623153393602, 0.3476031445164834],
)  # fmt: skip

# survreg: (coef, log(scale)) and their robust standard errors
R_SURVREG = {
    "Weibull": (
        [1.916120039703859, -0.191526394484084],
        [0.193783473870293, 0.152523715519059],
    ),
    "LogNormal": (
        [1.565354321790535, 0.137748009886272],
        [0.207456687313002, 0.214686817699261],
    ),
    "LogLogistic": (
        [1.539003522486248, -0.482600282812611],
        [0.197381295182942, 0.187186096923434],
    ),
    "Exponential": ([1.96080680925178], [0.20832261991811]),
}
R_AFT = (
    [2.157448137754637, -0.656109818799668, -0.660322621484025,
     -0.477022092803571],
    [0.227220553503763, 0.306052636226662, 0.137563313236780,
     0.161286464480136],
)  # fmt: skip
R_COX = {
    "efron": (
        [1.06791948026886, 1.01251989144677],
        [0.472898791484031, 0.239872432814770],
        -125.646463108494,
    ),
    "breslow": (
        [1.07811152152835, 1.01076581411262],
        [0.470415731017950, 0.236148382983504],
        -125.985886210508,
    ),
}
R_COX_TRUNCATED = (
    [1.08336516101190, 1.01671555418292],
    [0.480453994661017, 0.241024675928827],
)
R_COX_CLUSTER_SE = [0.678374922893751, 0.138051802381629]
# survreg(... + cluster(cl), weights = w, robust = TRUE): the standard
# errors of (Intercept), z1, z2 and Log(scale)
R_AFT_CLUSTER_SE = {
    "Weibull": [0.251976433689535, 0.432836073664428, 0.125109406853725,
                0.131727088477561],
    "LogNormal": [0.288480976099690, 0.475653620203211, 0.115085366985270,
                  0.237485354537948],
}  # fmt: skip
R_LOGNORMAL_AFT_SE = [0.279629845800538, 0.404547454386976,
                      0.132959957833004, 0.244599283596568]  # fmt: skip
SURVEY_COX_SE = [0.478923222782169, 0.242928255789650]

LIFELINES_KM = [
    0.9231611893583727,
    0.7765258215962443,
    0.5380548724676895,
    0.44198494228483776,
    0.26714390881271705,
]
LIFELINES_WEIBULL = (6.794503165274475, 1.2110931520967734)
LIFELINES_COX = [1.0679194758604589, 1.012519884427659]


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


def _log_sf_se(model, x):
    idx = np.searchsorted(model.x, x, side="right") - 1
    return np.sqrt(model.greenwood[idx])


@pytest.mark.parametrize("name", list(R_SURVFIT))
def test_survfit_robust(data, name):
    model = getattr(sp, name).fit(data["x"], data["c"], w=data["w"])
    surv, se = R_SURVFIT[name]
    assert_allclose(model.sf(TIMES), surv, **CLOSED)
    assert_allclose(_log_sf_se(model, TIMES), se, **CLOSED)
    # The bounds are the usual log(-log) ones on this variance
    z = 1.959963984540054
    s = np.asarray(surv)
    v = np.asarray(se) ** 2 / np.log(s) ** 2
    lower = np.exp(-np.exp(np.log(-np.log(s)) + z * np.sqrt(v)))
    upper = np.exp(-np.exp(np.log(-np.log(s)) - z * np.sqrt(v)))
    assert_allclose(model.cb(TIMES), np.column_stack([lower, upper]), **CLOSED)


def test_survfit_robust_late_entry(data):
    model = sp.KaplanMeier.fit(
        data["x"], data["c"], tl=data["tl"], w=data["w"]
    )
    surv, se = R_SURVFIT_TRUNCATED
    assert_allclose(model.sf(TIMES), surv, **CLOSED)
    assert_allclose(_log_sf_se(model, TIMES), se, **CLOSED)


def test_kaplan_meier_matches_lifelines(data):
    model = sp.KaplanMeier.fit(data["x"], data["c"], w=data["w"])
    assert_allclose(model.sf(TIMES), LIFELINES_KM, **CLOSED)


def test_rmst_robust_variance_is_the_jackknife(data):
    # The derivative of the restricted mean in each unit's weight, taken
    # numerically, and the infinitesimal jackknife sum_i w_i^2 U_i^2.
    x, c, w = data["x"], data["c"], data["w"]
    model = sp.KaplanMeier.fit(x, c, w=w)
    res = model.rmst(tau=6.0)

    def area(weights):
        return sp.KaplanMeier.fit(x, c, w=weights).mean(tau=6.0)

    h = 1e-6
    U = np.array(
        [
            (area(w + h * np.eye(len(w))[i]) - res["rmst"]) / h
            for i in range(len(w))
        ]
    )
    assert_allclose(res["rmst"], 4.104576926253483, rtol=1e-12)
    assert_allclose(res["se"], np.sqrt(np.sum(w**2 * U**2)), rtol=1e-5)


def _survreg_scale(name, model):
    """SurPyval's parameters and covariance on survreg's scale:
    ``(log scale-parameter or mu, log sigma)``."""
    p = np.asarray(model.params, dtype=float)
    cov = model.covariance()
    if name == "LogNormal":
        coef = np.array([p[0], np.log(p[1])])
        J = np.diag([1.0, 1.0 / p[1]])
    elif name == "Exponential":
        coef = np.array([-np.log(p[0])])
        J = np.array([[-1.0 / p[0]]])
    else:  # Weibull (alpha, beta), LogLogistic (alpha, beta)
        coef = np.array([np.log(p[0]), -np.log(p[1])])
        J = np.diag([1.0 / p[0], -1.0 / p[1]])
    return coef, np.sqrt(np.diag(J @ cov @ J.T))


@pytest.mark.parametrize("name", list(R_SURVREG))
def test_survreg_robust(data, name):
    model = getattr(sp, name).fit(data["x"], data["c"], w=data["w"])
    coef, se = _survreg_scale(name, model)
    ref_coef, ref_se = R_SURVREG[name]
    assert_allclose(coef, ref_coef, rtol=0, atol=2e-5)
    assert_allclose(se, ref_se, rtol=2e-4)
    # The Wald bounds of the parameters use it: param_cb and
    # standard_errors agree with covariance()
    assert_allclose(
        model.standard_errors(), np.sqrt(np.diag(model.covariance()))
    )


def test_weibull_matches_lifelines(data):
    model = sp.Weibull.fit(data["x"], data["c"], w=data["w"])
    assert_allclose(model.params, LIFELINES_WEIBULL, rtol=2e-5)


def test_weibull_aft_robust(data):
    model = sp.WeibullAFT.fit(data["x"], data["Z"], c=data["c"], w=data["w"])
    alpha, beta, b1, b2 = model.params
    cov = model.covariance()
    # survreg: log T = mu - (coef' Z) ..., SurPyval's AFT coefficients are
    # minus R's; the intercept is log alpha and log(scale) = -log beta.
    coef = [np.log(alpha), -b1, -b2, -np.log(beta)]
    J = np.zeros((4, 4))
    J[0, 0], J[1, 2], J[2, 3], J[3, 1] = 1 / alpha, -1, -1, -1 / beta
    se = np.sqrt(np.diag(J @ cov @ J.T))
    ref_coef, ref_se = R_AFT
    assert_allclose(coef, ref_coef, rtol=0, atol=2e-5)
    assert_allclose(se, ref_se, rtol=2e-4)


@pytest.mark.parametrize("ties", list(R_COX))
def test_coxph_robust(data, ties):
    model = sp.CoxPH.fit(
        data["x"], data["Z"], c=data["c"], w=data["w"], tie_method=ties
    )
    coef, se, loglik = R_COX[ties]
    assert_allclose(model.params, coef, rtol=0, atol=1e-7)
    assert_allclose(model.standard_errors(), se, rtol=1e-6)
    assert_allclose(model.log_likelihood, loglik, rtol=0, atol=1e-7)
    if ties != "efron":
        return
    # survey's svycoxph (Efron's ties): the same, times sqrt(n / (n - 1))
    n = len(data["x"])
    assert_allclose(
        model.standard_errors() * np.sqrt(n / (n - 1)),
        SURVEY_COX_SE,
        rtol=1e-6,
    )


def test_coxph_robust_late_entry(data):
    model = sp.CoxPH.fit(
        data["x"], data["Z"], c=data["c"], tl=data["tl"], w=data["w"]
    )
    coef, se = R_COX_TRUNCATED
    assert_allclose(model.params, coef, rtol=0, atol=1e-7)
    assert_allclose(model.standard_errors(), se, rtol=1e-6)


def test_coxph_cluster(data):
    model = sp.CoxPH.fit(data["x"], data["Z"], c=data["c"], w=data["w"])
    cluster = np.arange(len(data["x"])) // 4
    se = np.sqrt(np.diag(model.robust_covariance(cluster)))
    assert_allclose(se, R_COX_CLUSTER_SE, rtol=1e-6)


def test_coxph_matches_lifelines(data):
    model = sp.CoxPH.fit(data["x"], data["Z"], c=data["c"], w=data["w"])
    assert_allclose(model.params, LIFELINES_COX, rtol=0, atol=1e-6)


def _aft_se_on_survreg_scale(name, model, cov):
    """The standard errors of ``cov`` (of ``model.params``) on survreg's
    scale: (Intercept), z1, z2, Log(scale)."""
    p = np.asarray(model.params, dtype=float)
    J = np.zeros((4, 4))
    if name == "Weibull":  # alpha, beta, b1, b2
        J[0, 0], J[3, 1] = 1 / p[0], -1 / p[1]
    else:  # LogNormal: mu, sigma, b1, b2
        J[0, 0], J[3, 1] = 1.0, 1 / p[1]
    # SurPyval's AFT coefficients are minus R's
    J[1, 2], J[2, 3] = -1, -1
    return np.sqrt(np.diag(J @ cov @ J.T))


@pytest.mark.parametrize("name", list(R_AFT_CLUSTER_SE))
def test_aft_robust_covariance_of_weighted_fit(data, name):
    # robust_covariance() of a weighted fit is the weighted sandwich, the
    # same as covariance(); with cluster= the weighted scores are summed
    # within each cluster (R's survreg with weights and cluster()). It
    # used to wrap the sandwich in a second one and drop the weights.
    model = getattr(sp, name + "AFT").fit(
        data["x"], data["Z"], c=data["c"], w=data["w"]
    )
    assert_allclose(model.robust_covariance(), model.covariance(), rtol=0)
    ref = R_AFT[1] if name == "Weibull" else R_LOGNORMAL_AFT_SE
    se = _aft_se_on_survreg_scale(name, model, model.robust_covariance())
    assert_allclose(se, ref, rtol=2e-4)
    cluster = np.arange(len(data["x"])) // 4
    se = _aft_se_on_survreg_scale(
        name, model, model.robust_covariance(cluster)
    )
    assert_allclose(se, R_AFT_CLUSTER_SE[name], rtol=2e-4)
    table = model.summary(robust=True, cluster=cluster)
    assert_allclose(
        table["se(coef)"].to_numpy(),
        model.robust_standard_errors(cluster),
        rtol=1e-12,
    )
    # Each row its own cluster is no clustering at all
    assert_allclose(
        model.robust_covariance(np.arange(len(data["x"]))),
        model.covariance(),
        rtol=1e-10,
        atol=1e-14,
    )


def test_coxph_robust_summary_of_weighted_fit(data):
    # summary(robust=True) of a weighted Cox fit used to fail on a
    # missing attribute ('jac'): its robust standard errors are the
    # weighted Lin-Wei ones, R's coxph(..., weights = w, robust = TRUE),
    # and with cluster= R's + cluster(cl).
    model = sp.CoxPH.fit(data["x"], data["Z"], c=data["c"], w=data["w"])
    table = model.summary(robust=True)
    assert_allclose(table["se(coef)"].to_numpy(), R_COX["efron"][1], rtol=1e-6)
    cluster = np.arange(len(data["x"])) // 4
    table = model.summary(robust=True, cluster=cluster)
    assert_allclose(table["se(coef)"].to_numpy(), R_COX_CLUSTER_SE, rtol=1e-6)
    assert_allclose(
        model.robust_summary(cluster)["se"], R_COX_CLUSTER_SE, rtol=1e-6
    )


# survreg(Surv(x, d) ~ 1, weights = w, robust = TRUE, dist = ...): the
# square roots of the diagonal of ``naive.var``, the inverse Hessian of
# the weighted log-likelihood (with the weights as given), on survreg's
# scale (R 4.6.2, survival 3.8.12).
R_SURVREG_NAIVE_SE = {
    "Weibull": [0.133746698013079, 0.132597679788907],
    "LogNormal": [0.161171132799090, 0.120215406198856],
    "LogLogistic": [0.142254899019476, 0.135595613723975],
    "Exponential": [0.159414430369081],
}

# SurPyval 0.26's bounds on the weighted fits, whose ``hess_inv`` held the
# robust covariance: param_cb of each parameter, cb at 2, 5 and 8,
# quantile_cb(0.1) and mean_cb().
V026_WEIGHTED_BOUNDS = {
    "Weibull": dict(
        param_cb=[[4.647415062036453, 9.933642319980377],
                  [0.8981539418993779, 1.6330803458670047]],
        cb=[[0.686008649206206, 0.8718070188363984],
            [0.3459949752794354, 0.6387339802890533],
            [0.13809395319114495, 0.47226938012620034]],
        quantile_cb=[0.6156086968527006, 1.8241934326183613],
        mean_cb=[4.2159158370194385, 9.64398582529258],
    ),
    "LogNormal": dict(
        param_cb=[[1.1587467983892026, 1.97196240789849],
                  [0.7534969992643635, 1.7480952184241658]],
        cb=[[0.6433246766443574, 0.8754586480314682],
            [0.34521472845661333, 0.6260657421087849],
            [0.18403703130471916, 0.5016836800014941]],
        quantile_cb=[0.5910899244937168, 2.0438948725636603],
        mean_cb=[4.208855421561086, 20.301666340773412],
    ),
}  # fmt: skip

# A weighted Weibull fit to these data saved by SurPyval 0.26: its
# "hess_inv" is the robust covariance.
V026_WEIGHTED_DICT = {
    "parameterization": "parametric", "distribution": "Weibull",
    "how": "MLE", "param_names": ["alpha", "beta"],
    "params": [6.794538905529906, 1.2110976633116133],
    "lfp": False, "p": 1.0, "zi": False, "f0": 0.0, "offset": False,
    "gamma": 0.0,
    "hess_inv": [[1.7336126462853145, -0.09095765031994801],
                 [-0.09095765031994801, 0.03412187150774266]],
    "covariance": [[1.7336126462853145, -0.09095765031994801],
                   [-0.09095765031994801, 0.03412187150774266]],
    "_neg_ll": 115.55693707673387, "weighted": True,
    "effective_n": 33.366591977720795, "maximum": "verified",
    "data_summary": "40 units: 24 events at 21 unique times, 16 right "
    "censored",
    "ic_n": 24.0, "schema": 3,
}  # fmt: skip


def _on_survreg_scale(name, model, cov):
    """``cov`` (of SurPyval's parameters) on survreg's scale, as
    ``_survreg_scale`` takes ``covariance()`` there: standard errors."""
    p = np.asarray(model.params, dtype=float)
    if name == "LogNormal":
        J = np.diag([1.0, 1.0 / p[1]])
    elif name == "Exponential":
        J = np.array([[-1.0 / p[0]]])
    else:
        J = np.diag([1.0 / p[0], -1.0 / p[1]])
    return np.sqrt(np.diag(J @ np.asarray(cov) @ J.T))


@pytest.mark.parametrize("name", list(R_SURVREG_NAIVE_SE))
def test_hess_inv_of_a_weighted_fit_is_the_inverse_hessian(data, name):
    # hess_inv is survreg's naive.var, the inverse Hessian; covariance()
    # is its var, the robust sandwich; they differ.
    model = getattr(sp, name).fit(data["x"], data["c"], w=data["w"])
    se = _on_survreg_scale(name, model, model.hess_inv)
    assert_allclose(se, R_SURVREG_NAIVE_SE[name], rtol=2e-4)
    _, robust_se = _survreg_scale(name, model)
    assert_allclose(robust_se, R_SURVREG[name][1], rtol=2e-4)
    assert not np.allclose(model.hess_inv, model.covariance())


@pytest.mark.parametrize("name", list(V026_WEIGHTED_BOUNDS))
def test_weighted_bounds_are_unchanged(data, name):
    # The bounds use covariance(), the robust one, as they did when
    # hess_inv held it (0.26's values).
    model = getattr(sp, name).fit(data["x"], data["c"], w=data["w"])
    ref = V026_WEIGHTED_BOUNDS[name]
    tol = dict(rtol=1e-9)
    names = model.dist.parameter_names
    assert_allclose([model.param_cb(p) for p in names], ref["param_cb"], **tol)
    assert_allclose(model.cb([2.0, 5.0, 8.0]), ref["cb"], **tol)
    assert_allclose(model.quantile_cb(0.1), ref["quantile_cb"], **tol)
    assert_allclose(model.mean_cb(), ref["mean_cb"], **tol)
    # The plot's band is the cb's on the robust covariance
    plot = model.get_plot_data()
    assert len(plot["cbs"])
    assert_allclose(plot["cbs"], model.cb(plot["x_model"], on="ff"), **tol)
    # Saved and loaded, the bounds and hess_inv are kept
    restored = sp.from_dict(model.to_dict())
    assert_allclose(restored.hess_inv, model.hess_inv, rtol=1e-15)
    assert_allclose(restored.covariance(), model.covariance(), rtol=1e-15)
    assert_allclose(restored.cb([2.0, 5.0, 8.0]), ref["cb"], **tol)


def test_a_weighted_dict_saved_by_v026_keeps_its_bounds(data):
    model = sp.from_dict(V026_WEIGHTED_DICT)
    fresh = sp.Weibull.fit(data["x"], data["c"], w=data["w"])
    ref = V026_WEIGHTED_BOUNDS["Weibull"]
    assert_allclose(model.cb([2.0, 5.0, 8.0]), ref["cb"], rtol=1e-9)
    assert_allclose(
        [model.param_cb(p) for p in ("alpha", "beta")],
        ref["param_cb"],
        rtol=1e-9,
    )
    # Its "hess_inv" was the robust covariance, and the inverse Hessian
    # cannot be rebuilt without the data, so it has none
    assert getattr(model, "hess_inv", None) is None
    assert_allclose(model.covariance(), fresh.covariance(), rtol=1e-6)
    # A dict saved now keeps the robust block under "hess_inv", for a
    # 0.26 reader, and the inverse Hessian under its own key
    d = fresh.to_dict()
    n = len(fresh.params)
    assert_allclose(d["hess_inv"], fresh.covariance()[:n, :n])
    assert_allclose(d["inverse_hessian"], fresh.hess_inv)
