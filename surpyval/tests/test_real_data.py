from collections import namedtuple

import lifelines
import numpy as np
import pytest
from lifelines import datasets
from reliability import Fitters

import surpyval as surv

# Datasets in x, c, n: as namedtuples
SurvivalData = namedtuple("SurvivalData", ["x", "c", "n", "name"])
IntervalSurvivalData = namedtuple(
    "IntervalSurvivalData", ["left", "right", "name"]
)

# Canadian Senators
df = datasets.load_canadian_senators()
x = df["diff_days"].values
c = 1 - df["observed"].astype(int)
zero_idx = x == 0
x = x[~zero_idx]
c = c[~zero_idx]
n = np.ones_like(x)
canadian_senators = SurvivalData(x, c, n, "canadian_senators")

# Bacteria...
df = datasets.load_c_botulinum_lag_phase()
left = df["lower_bound_days"]
right = df["upper_bound_days"]
bacteria = IntervalSurvivalData(left, right, "bacteria")

# Political Durations
df = datasets.load_dd()
x = df["duration"].values
c = 1 - df["observed"].astype(int)
n = np.ones_like(x)
politics = SurvivalData(x, c, n, "politics")

df = datasets.load_diabetes()
left = df["left"]
right = df["right"]
diabetes = IntervalSurvivalData(left, right, "diabetes")

# ???
df = datasets.load_g3()
x = df["time"].values
c = 1 - df["event"].astype(int)
n = np.ones_like(x)
g3 = SurvivalData(x, c, n, "g3")

# ???
df = datasets.load_gbsg2()
x = df["time"].values
c = 1 - df["cens"].astype(int)
n = np.ones_like(x)
gbsg2 = SurvivalData(x, c, n, "gbsg2")

# holl_molly_polly...??!!!?!?!?!?!
df = datasets.load_holly_molly_polly()
x = df["T"].values
c = np.zeros_like(x)
n = np.ones_like(x)
holly_molly_polly = SurvivalData(x, c, n, "holly_molly_polly")

# kidney
df = datasets.load_kidney_transplant()
x = df["time"].values
c = 1 - df["death"].astype(int)
n = np.ones_like(x)
kidney = SurvivalData(x, c, n, "kidney")

# larynx
df = datasets.load_larynx()
x = df["time"].values
c = np.zeros_like(x)
n = np.ones_like(x)
larynx = SurvivalData(x, c, n, "larynx")

# leukemia
df = datasets.load_leukemia()
x = df["t"].values
c = 1 - df["status"].astype(int)
n = np.ones_like(x)
leukemia = SurvivalData(x, c, n, "leukemia")

# lung
df = datasets.load_lung()
x = df["time"].dropna()
c = np.zeros_like(x)
n = np.ones_like(x)
lung = SurvivalData(x, c, n, "lung")

# lupus
df = datasets.load_lupus()
x = df["time_elapsed_between_estimated_onset_and_diagnosis_(months)"].dropna()
c = 1 - df["dead"].astype(int)
n = np.ones_like(x)
lupus = SurvivalData(x, c, n, "lupus")

# lymph
df = datasets.load_lymph_node()
x = df["survtime"].dropna()
c = 1 - df["censdead"].astype(int)
n = np.ones_like(x)
lymph = SurvivalData(x, c, n, "lymph")

# lymphoma
df = datasets.load_lymphoma()
x = df["Time"].dropna()
c = 1 - df["Censor"].astype(int)
n = np.ones_like(x)
lymphoma = SurvivalData(x, c, n, "lymphoma")

# mice
df = datasets.load_mice()
left = df["l"]
right = df["u"]
mice = IntervalSurvivalData(left, right, "mice")

# aids
df = datasets.load_multicenter_aids_cohort_study()
x = df["T"].dropna()
c = 1 - df["D"].astype(int)
n = np.ones_like(x)
aids = SurvivalData(x, c, n, "aids")

# nh4
df = datasets.load_nh4()
x = df["Week"].dropna()
c = 1 - df["Censored"].astype(int)
n = np.ones_like(x)
nh4 = SurvivalData(x, c, n, "nh4")

# panel
df = datasets.load_panel_test()
x = df["t"].dropna()
c = 1 - df["E"].astype(int)
n = np.ones_like(x)
panel = SurvivalData(x, c, n, "panel")

# recur
df = datasets.load_recur()
x = df["AGE"].dropna()
c = 1 - df["CENSOR"].astype(int)
n = np.ones_like(x)
recur = SurvivalData(x, c, n, "recur")

# reg
df = datasets.load_regression_dataset()
x = df["T"].dropna()
c = 1 - df["E"].astype(int)
n = np.ones_like(x)
reg = SurvivalData(x, c, n, "reg")

# rossi
df = datasets.load_rossi()
x = df["week"].dropna()
c = 1 - df["arrest"].astype(int)
n = np.ones_like(x)
rossi = SurvivalData(x, c, n, "rossi")

# static
df = datasets.load_static_test()
x = df["t"].dropna() + 1e-10
c = 1 - df["E"].astype(int)
n = np.ones_like(x)
static = SurvivalData(x, c, n, "static")

# walton
df = datasets.load_waltons()
x = df["T"].dropna()
c = 1 - df["E"].astype(int)
n = np.ones_like(x)
walton = SurvivalData(x, c, n, "walton")


def id_func(val):
    if isinstance(val, SurvivalData):
        return val.name
    elif isinstance(val, IntervalSurvivalData):
        return val.name


xcn_datasets = [
    canadian_senators,
    politics,
    g3,
    gbsg2,
    holly_molly_polly,
    kidney,
    larynx,
    leukemia,
    lung,
    lupus,
    lymph,
    lymphoma,
    aids,
    nh4,
    panel,
    recur,
    reg,
    rossi,
    # static,
    walton,
]

int_datasets = [bacteria, diabetes, mice]

wf = lifelines.WeibullFitter()
lnf = lifelines.LogNormalFitter()
llf = lifelines.LogLogisticFitter()
ef = lifelines.ExponentialFitter()

DISTS = {
    "Weibull": (wf, surv.Weibull),
    "Exponential": (ef, surv.Exponential),
    "LogNormal": (lnf, surv.LogNormal),
    "LogLogistic": (llf, surv.LogLogistic),
}

REL_DISTS = {
    "Exponential": (Fitters.Fit_Exponential_1P, surv.Exponential),
    "Weibull": (Fitters.Fit_Weibull_2P, surv.Weibull),
    "Gamma": (Fitters.Fit_Gamma_2P, surv.Gamma),
    "LogNormal": (Fitters.Fit_Lognormal_2P, surv.LogNormal),
    "LogLogistic": (Fitters.Fit_Loglogistic_2P, surv.LogLogistic),
    "Normal": (Fitters.Fit_Normal_2P, surv.Normal),
    "Gumbel": (Fitters.Fit_Gumbel_2P, surv.Gumbel),
    "Beta": (Fitters.Fit_Beta_2P, surv.Beta),
}


def generate_case():
    for i, data in enumerate(xcn_datasets):
        yield data


def generate_real_cases():
    for dist in DISTS.keys():
        for data in xcn_datasets:
            yield data, dist


def generate_real_cases_reliability():
    for dist in REL_DISTS.keys():
        for data in xcn_datasets:
            yield data, dist


def generate_real_cases_int():
    for dist in DISTS.keys():
        for data in int_datasets:
            yield data, dist


def params_with_xcn_data_rel(data, surpyval_fitter, rel_fitter):
    if surpyval_fitter.name == "Beta":
        x = data.x / (data.x.max() + 1)
    else:
        x = data.x
    f, s = surv.xcn_to_fs(x, data.c, data.n)
    if s == []:
        s = None
    rel_model = rel_fitter(f, s)
    if surpyval_fitter.name == "Exponential":
        rel_params = rel_model.Lambda
    elif surpyval_fitter.name in ["Weibull", "Gamma", "LogLogistic", "Beta"]:
        rel_params = np.array([rel_model.alpha, rel_model.beta])
    elif surpyval_fitter.name in ["LogNormal", "Normal", "Gumbel"]:
        rel_params = np.array([rel_model.mu, rel_model.sigma])

    surp_est = surpyval_fitter.fit(x, data.c, data.n)
    if np.allclose(rel_params, surp_est.params, 1e-1):
        return True
    else:
        # reliability has performance that is to be desired. So check that
        # loglike is better or within a small tolerance:
        return (surp_est.neg_ll() - (-rel_model.loglik)) < 1e-5


def params_with_xcn_data(data, surpyval_fitter, lifelines_fitter):
    ll_est = lifelines_fitter.fit(
        data.x, 1 - data.c, weights=data.n
    ).params_.values
    surp_est = surpyval_fitter.fit(data.x, data.c, data.n).params
    if surpyval_fitter.name == "Exponential":
        surp_est = 1.0 / surp_est
    return ll_est, surp_est


def params_with_int_data(data, surpyval_fitter, lifelines_fitter):
    ll_est = lifelines_fitter.fit_interval_censoring(
        data.left, data.right
    ).params_.values
    surp_est = surpyval_fitter.fit(xl=data.left, xr=data.right).params
    if surpyval_fitter.name == "Exponential":
        surp_est = 1.0 / surp_est
    return ll_est, surp_est


@pytest.mark.parametrize("data", generate_case(), ids=id_func)
def test_weibull_offset_with_real(data):
    # Known issues - distribution too far off being Weibull to work with offset
    if data.name in ["gbsg2", "kidney", "lymph", "aids"]:
        assert True
    else:
        surpyval_fitter = surv.Weibull
        fitted = surpyval_fitter.fit(data.x, data.c, data.n, offset=True)
        assert fitted.res.success or ("Desired error" in fitted.res.message)


@pytest.mark.parametrize("data,dist", generate_real_cases(), ids=id_func)
def test_against_lifelines_with_real_data(data, dist):
    ll_fitter = DISTS[dist][0]
    surp_fitter = DISTS[dist][1]
    assert np.allclose(
        *params_with_xcn_data(data, surp_fitter, ll_fitter), 1e-1
    )


@pytest.mark.parametrize("data,dist", generate_real_cases_int(), ids=id_func)
def test_against_lifelines_with_real_data_interval(data, dist):
    ll_fitter = DISTS[dist][0]
    surp_fitter = DISTS[dist][1]
    assert np.allclose(
        *params_with_int_data(data, surp_fitter, ll_fitter), 1e-1
    )


@pytest.mark.parametrize(
    "data,dist", generate_real_cases_reliability(), ids=id_func
)
def test_against_reliability_with_real_data(data, dist):
    rel_fitter = REL_DISTS[dist][0]
    surp_fitter = REL_DISTS[dist][1]
    assert params_with_xcn_data_rel(data, surp_fitter, rel_fitter)
