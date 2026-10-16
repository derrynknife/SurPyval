"""The recurrent-event families of ``diagnose()`` (#873): the Poisson
processes, the imperfect-repair models and the proportional-intensity
models. Their statistics against references (R, the MIL-HDBK-189 worked
example and the package's own trend and repair tests), each advice rule
firing on data built to call for it, the refusals, the plots and the
report's round trip.
"""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import surpyval as sp
import surpyval.diagnostics.recurrent as rec
from surpyval.datasets import load_mettas_and_zhao
from surpyval.diagnostics import Diagnosis
from surpyval.recurrent import (
    ARA,
    ARI,
    HPP,
    CoxLewis,
    CrowAMSAA,
    Duane,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
    laplace,
)
from surpyval.recurrent.diagnostics import item_windows


@pytest.fixture(autouse=True)
def fewer_resamples(monkeypatch):
    # 39 bootstrap refits: p-values in steps of 0.025, so a test can still
    # fail at 0.05, in a fraction of the time of the report's 99.
    monkeypatch.setattr(rec, "N_BOOT", 39)


@pytest.fixture
def no_fit_test(monkeypatch):
    # For the tests of the other checks: four refits, a fit p-value of at
    # least 0.2 (never a failure), at a fraction of the time.
    monkeypatch.setattr(rec, "N_BOOT", 4)


def _quiet_fit(fitter, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fitter.fit(*args, **kwargs)


def _items(times, T):
    x, i, c = [], [], []
    for k, ts in enumerate(times):
        x += list(ts) + [T]
        i += [k] * (len(ts) + 1)
        c += [0] * len(ts) + [1]
    return np.array(x, dtype=float), np.array(i), np.array(c)


def _power_law(rng, m, T, beta, eta, frailty=None):
    """``m`` items observed to ``T`` under the power law ``(t / eta) **
    beta``, each item's intensity multiplied by its ``frailty``."""
    times = []
    for k in range(m):
        f = 1.0 if frailty is None else frailty[k]
        N = rng.poisson(f * (T / eta) ** beta)
        times.append(np.sort(T * rng.random(N) ** (1 / beta)))
    return _items(times, T)


def _renewal(rng, m, T, shape, scale):
    """``m`` items observed to ``T``, each a renewal process with Weibull
    times between failures."""
    times = []
    for _ in range(m):
        t, ts = 0.0, []
        while True:
            t += scale * rng.weibull(shape)
            if t > T:
                break
            ts.append(t)
        times.append(ts)
    return _items(times, T)


def _bathtub(rng, m, T):
    """An intensity that falls and then rises, which no monotone intensity
    (power law, log-linear) describes."""

    def intensity(t):
        return 2.0 / (1 + t) + 3 * (t / T) ** 4

    times = []
    for _ in range(m):
        n = rng.poisson(5.0 * T)
        cand = np.sort(rng.uniform(0, T, n))
        times.append(cand[rng.uniform(0, 5.0, n) < intensity(cand)])
    return _items(times, T)


def _rows(d, check, verdict=None):
    table = d.table[d.table["check"] == check]
    if verdict is not None:
        table = table[table["verdict"] == verdict]
    return table


def _advice(d, check, text):
    return [a for a in d.advice if a.check == check and text in a]


@pytest.fixture(scope="module")
def mettas():
    df = load_mettas_and_zhao()
    return df["x"].values, df["i"].values, df["c"].values


# The MIL-HDBK-189 test-fix-find-test example (MIL-HDBK-00189A section
# 7.5, as surpyval/tests/recurrent/test_growth_projection.py has it): one
# system to T = 400, 56 failures.
HDBK = [
    0.7, 3.7, 13.2, 15, 17.6, 25.3, 47.5, 54, 54.5, 56.4, 63.6, 72.2, 99.2,
    99.6, 100.3, 102.5, 112, 112.2, 120.9, 121.9, 125.5, 133.4, 151, 163,
    164.7, 174.5, 177.4, 191.6, 192.7, 213, 244.8, 249, 250.8, 260.1, 263.5,
    273.1, 274.7, 282.8, 285, 304, 315.4, 317.1, 320.6, 324.5, 324.9, 342,
    350.2, 355.2, 364.6, 364.9, 366.3, 373, 379.4, 389, 394.9, 395.2,
]  # fmt: skip


# -- references ---------------------------------------------------------------
# R 4.3.3, base R (no R package has these tests): the script is
#
#   trend <- function(t, sys, Tq) {     # time truncated at T_q
#     num <- 0; v <- 0; mil <- 0; N <- 0
#     for (q in unique(sys)) {
#       tt <- sort(t[sys == q]); T <- Tq[[as.character(q)]]
#       n <- length(tt); N <- N + n
#       num <- num + sum(tt) - n * T / 2; v <- v + n * T^2 / 12
#       mil <- mil + 2 * sum(log(T / tt))
#     }
#     c(laplace = num / sqrt(v), mil = mil, mil_df = 2 * N)
#   }
#   lr_ft <- function(t, sys) {         # failure truncated at t_n
#     num <- 0; v <- 0; gaps <- c()
#     for (q in unique(sys)) {
#       tt <- sort(t[sys == q]); n <- length(tt); T <- tt[n]
#       gaps <- c(gaps, diff(c(0, tt)))
#       if (n < 2) next
#       num <- num + sum(tt[-n]) - (n - 1) * T / 2
#       v <- v + (n - 1) * T^2 / 12
#     }
#     z <- num / sqrt(v) / (sd(gaps) / mean(gaps))
#     c(lr = z, lr_p = 2 * pnorm(-abs(z)), cv = sd(gaps) / mean(gaps))
#   }
#
# on the handbook data (one system, T = 400) and on Mettas and Zhao's six
# systems (load_mettas_and_zhao).
R_REFERENCE = {
    "hdbk": {
        "laplace": 0.166763345471531,
        "mil": 120.845090553601295,
        "mil_df": 112,
        "lr": 0.0962657303168498,
        "lr_p": 0.9233095278912804,
        "cv": 0.9932741979111707,
    },
    "mettas": {
        "laplace": 0.734874178884267,
        "laplace_p": 0.462416125264746,
        "mil": 42.594660890942649,
        "mil_df": 54,
        "mil_p": 0.262491581661124,
        "lr": 1.9304394767131754,
        "lr_p": 0.0535524075468083,
        "cv": 0.8957328647795638,
    },
}


def _trend_row(d, test):
    t = _rows(d, "trend")
    return t[t["test"] == "{} trend test".format(test)].iloc[0]


def test_trend_statistics_match_r_on_the_handbook_example():
    x = np.r_[HDBK, 400.0]
    c = np.r_[np.zeros(len(HDBK)), 1]
    d = HPP.fit(x, c=c).diagnose()
    ref = R_REFERENCE["hdbk"]
    assert _trend_row(d, "Laplace")["statistic"] == pytest.approx(
        ref["laplace"], rel=1e-12
    )
    mil = _trend_row(d, "MIL-HDBK-189C")
    assert mil["statistic"] == pytest.approx(ref["mil"], rel=1e-12)
    assert mil["df"] == ref["mil_df"]
    # The handbook's Crow-AMSAA MLE is beta = 0.9268 (its 0.91026 is
    # 55/56 of it), and the statistic is 2 N / beta.
    assert mil["statistic"] == pytest.approx(2 * 56 / 0.9268, rel=1e-4)
    lr = _trend_row(d, "Lewis-Robinson")
    assert lr["statistic"] == pytest.approx(ref["lr"], rel=1e-12)
    assert lr["p"] == pytest.approx(ref["lr_p"], rel=1e-12)
    assert "CV 0.993" in lr["detail"]
    assert d.ok


def test_trend_statistics_match_r_and_the_package_on_six_systems(mettas):
    model = HPP.fit(*mettas)
    d = model.diagnose()
    ref = R_REFERENCE["mettas"]
    row = _trend_row(d, "Laplace")
    assert row["statistic"] == pytest.approx(ref["laplace"], rel=1e-12)
    assert row["p"] == pytest.approx(ref["laplace_p"], rel=1e-12)
    row = _trend_row(d, "MIL-HDBK-189C")
    assert row["statistic"] == pytest.approx(ref["mil"], rel=1e-12)
    assert row["p"] == pytest.approx(ref["mil_p"], rel=1e-12)
    row = _trend_row(d, "Lewis-Robinson")
    assert row["statistic"] == pytest.approx(ref["lr"], rel=1e-12)
    assert row["p"] == pytest.approx(ref["lr_p"], rel=1e-12)
    # ... and they are the model's own trend tests
    for test, key in (
        ("Laplace", "laplace"),
        ("MIL-HDBK-189C", "mil_hdbk_189c"),
    ):
        own = model.trend_test(key)
        row = _trend_row(d, test)
        assert row["statistic"] == own.statistic and row["p"] == own.p_value


def test_lewis_robinson_is_laplace_with_the_spread_of_the_gaps(mettas):
    x, i, c = mettas
    out = rec.lewis_robinson(item_windows(HPP.fit(x, i, c).data))
    # Failure truncated at each system's last failure, as laplace(T=None)
    events = x[c == 0]
    z = laplace(events, i[c == 0]).statistic
    assert out["statistic"] == pytest.approx(z / out["cv"], rel=1e-12)
    assert out["n_gaps"] == events.size
    assert out["cv"] == pytest.approx(R_REFERENCE["mettas"]["cv"], rel=1e-12)


def test_lewis_robinson_holds_its_size_under_a_renewal_process():
    # The time-truncated form rejected 30% of regular renewal processes
    # (Weibull shape 3) at the 5% level; the failure-truncated one does not.
    rng = np.random.default_rng(2)
    for shape in (0.6, 3.0):
        rejected = 0
        for _ in range(100):
            x, i, c = _renewal(rng, 10, 100.0, shape, 5.0)
            windows = item_windows(HPP.fit(x, i, c).data)
            rejected += rec.lewis_robinson(windows)["p_value"] < 0.05
        assert rejected <= 12, (shape, rejected)


def test_lewis_robinson_with_delayed_entry_starts_at_the_first_failure():
    # An item that entered at 5 is renewed first at its first failure, 7.
    windows = [
        (1, np.array([7.0, 9.0, 12.0, 14.0]), 5.0, 20.0, True),
        (2, np.array([1.0, 4.0, 5.0]), 0.0, 6.0, True),
    ]
    out = rec.lewis_robinson(windows)
    gaps = np.r_[np.diff([7.0, 9, 12, 14]), np.diff([0.0, 1, 4, 5])]
    cv = gaps.std(ddof=1) / gaps.mean()
    z = laplace(
        [9.0, 12.0, 1.0, 4.0], [1, 1, 2, 2], T={1: 14, 2: 5}, tl={1: 7, 2: 0}
    )
    assert out["cv"] == pytest.approx(cv)
    assert out["statistic"] == pytest.approx(z.statistic / cv)


def test_repair_rows_are_the_models_repair_test(mettas, no_fit_test):
    model = GeneralizedRenewal.fit(*mettas)
    d = model.diagnose()
    test = model.repair_test()
    rows = _rows(d, "repair")
    assert list(rows["test"]) == [
        "likelihood-ratio test of perfect repair (q = 0)",
        "likelihood-ratio test of minimal repair (q = 1)",
    ]
    np.testing.assert_allclose(
        rows["statistic"], [test.perfect.statistic, test.minimal.statistic]
    )
    np.testing.assert_allclose(
        rows["p"], [test.perfect.p_value, test.minimal.p_value]
    )
    # q = 0 is on the edge of q's range: the halved chi-squared(1) tail
    from scipy.stats import chi2

    perfect = rows.iloc[0]
    assert perfect["p"] == pytest.approx(chi2.sf(perfect["statistic"], 1) / 2)
    assert "(on the edge of its range)" in perfect["detail"]


def test_cramer_von_mises_row_is_the_models_test_with_the_reports_seed(
    mettas,
):
    model = CrowAMSAA.fit(*mettas)
    row = _rows(model.diagnose(), "fit").iloc[0]
    own = model.cramer_von_mises(n_boot=rec.N_BOOT, random_state=rec.SEED)
    assert row["statistic"] == own.statistic and row["p"] == own.p_value
    assert model.diagnose().to_dict() == model.diagnose().to_dict()


def test_dispersion_test_by_hand(mettas):
    model = HPP.fit(*mettas)
    row = _rows(model.diagnose(), "between_items").iloc[0]
    x, i, c = mettas
    rate = (c == 0).sum() / sum(x[i == k].max() for k in np.unique(i))
    observed = np.array([((i == k) & (c == 0)).sum() for k in np.unique(i)])
    expected = rate * np.array([x[i == k].max() for k in np.unique(i)])
    stat = ((observed - expected) ** 2 / expected).sum()
    assert row["statistic"] == pytest.approx(stat, rel=1e-10)
    assert row["df"] == 5


def test_dispersion_test_holds_its_size():
    from surpyval.diagnostics.registry import Context

    rng = np.random.default_rng(1)
    rejected = 0
    for _ in range(150):
        x, i, c = _power_law(rng, 15, 100.0, 1.8, 20.0)
        model = CrowAMSAA.fit(x, i, c)
        p = rec._dispersion(Context(model, 0.05, [], []))["p_value"]
        rejected += p < 0.05
    assert rejected <= 15


# -- the Poisson processes ----------------------------------------------------
def test_hpp_on_a_power_law_fails_its_trend_and_is_advised_an_nhpp():
    rng = np.random.default_rng(1)
    d = HPP.fit(*_power_law(rng, 8, 100.0, 2.0, 25.0)).diagnose()
    assert d.family == "Recurrent events: Poisson process"
    assert d.model == "ParametricRecurrenceModel"
    trend = _rows(d, "trend")
    assert list(trend["verdict"]) == ["fail"] * 3
    assert set(trend["detail"]) == {
        "the rate of failures increases with time",
        "not a renewal process: the rate of failures increases with time",
    }
    (advice,) = _advice(d, "trend", "")
    assert advice.startswith("The rate of failures increases with time")
    assert "CrowAMSAA.fit" in advice and "GeneralizedRenewal.fit" in advice
    assert advice.features[:2] == ("CrowAMSAA.fit", "CoxLewis.fit")
    assert not d.ok


def test_hpp_on_a_renewal_process_is_advised_a_renewal_process():
    # Weibull(0.6) times between failures: no trend, but far more spread
    # than exponential. Laplace and MIL-HDBK-189 reject the HPP;
    # Lewis-Robinson does not reject a renewal process.
    rng = np.random.default_rng(1)
    x, i, c = _renewal(rng, 8, 100.0, 0.6, 6.0)
    d = HPP.fit(x, i, c).diagnose()
    assert list(_rows(d, "trend")["verdict"]) == ["fail", "fail", "ok"]
    (advice,) = _advice(d, "trend", "")
    assert advice.startswith(
        "A test against a constant rate (Laplace and MIL-HDBK-189C) finds"
    )
    assert advice.features == ("GeneralizedRenewal.fit", "repair_test")
    # Fitted with a power law instead, the trend rows are a screen: the
    # trend is found (ok), the renewal process is not ruled out (warn)
    d = CrowAMSAA.fit(x, i, c).diagnose()
    assert list(_rows(d, "trend")["verdict"]) == ["ok", "ok", "warn"]
    (advice,) = _advice(d, "trend", "")
    assert advice.startswith("Lewis-Robinson does not reject a renewal")


@pytest.mark.parametrize("fitter", [CrowAMSAA, Duane, CoxLewis])
def test_an_nhpp_with_no_trend_is_advised_an_hpp(fitter):
    rng = np.random.default_rng(3)
    d = _quiet_fit(fitter, *_power_law(rng, 8, 100.0, 1.0, 5.0)).diagnose()
    trend = _rows(d, "trend")
    assert list(trend["verdict"]) == ["warn"] * 3
    assert trend["detail"].iloc[0] == (
        "no evidence of a trend; consistent with a homogeneous Poisson "
        "process"
    )
    (advice,) = _advice(d, "trend", "")
    assert "HPP.fit" in advice and advice.features == ("HPP.fit", "aic")
    # A screen never fails
    assert "fail" not in set(trend["verdict"])


def test_a_bathtub_intensity_fails_the_fit_of_a_power_law():
    rng = np.random.default_rng(4)
    d = CrowAMSAA.fit(*_bathtub(rng, 6, 30.0)).diagnose()
    row = _rows(d, "fit").iloc[0]
    assert row["verdict"] == "fail"
    assert row["detail"].startswith("the event times depart")
    (advice,) = _advice(d, "fit", "")
    assert "Crow-AMSAA (CrowAMSAA.fit)" in advice
    assert "CoxLewis.fit" in advice and "GeneralizedRenewal.fit" in advice


def test_items_that_differ_are_advised_proportional_intensity():
    rng = np.random.default_rng(5)
    frailty = rng.gamma(2.0, 0.5, 12)
    d = CrowAMSAA.fit(
        *_power_law(rng, 12, 100.0, 1.5, 10.0, frailty=frailty)
    ).diagnose()
    row = _rows(d, "between_items").iloc[0]
    assert row["verdict"] == "fail" and row["df"] == 11
    assert row["detail"].startswith("the items' counts vary more")
    (advice,) = _advice(d, "between_items", "")
    assert "ProportionalIntensityNHPP.fit(x, Z, i=..., c=..., " in advice
    assert "baseline=CrowAMSAA)" in advice
    assert advice.features == (
        "ProportionalIntensityNHPP.fit(baseline=)",
        "ProportionalIntensityHPP.fit",
    )
    d = HPP.fit(*_power_law(rng, 12, 100.0, 1.0, 5.0, frailty=frailty))
    (advice,) = _advice(d.diagnose(), "between_items", "")
    assert "ProportionalIntensityHPP.fit(x, Z, i=..., c=...)" in advice


def test_poisson_data_that_meet_the_assumptions_pass():
    rng = np.random.default_rng(13)
    d = CrowAMSAA.fit(*_power_law(rng, 10, 100.0, 1.8, 15.0)).diagnose()
    assert d.ok and not d.advice, d
    assert set(d.table["verdict"]) == {"ok"}
    assert any(n.startswith("Model: Crow-AMSAA") for n in d.notes)
    assert any("minimal repair" in n for n in d.notes)


# -- the imperfect-repair models ----------------------------------------------
@pytest.fixture(scope="module")
def kijima_half():
    sim = GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.5)
    data = sim.time_terminated_simulation_data(100.0, items=10, random_state=1)
    return _quiet_fit(GeneralizedRenewal, data)


def test_imperfect_repair_that_is_neither_end_passes(kijima_half):
    d = kijima_half.diagnose()
    assert d.family == "Recurrent events: imperfect repair"
    assert set(d.table["check"]) == {"trend", "fit", "repair"}
    assert d.ok and not d.advice, d
    assert list(_rows(d, "repair")["verdict"]) == ["ok", "ok"]
    assert "minimal repair rejected" in _rows(d, "repair")["detail"].iloc[1]


def test_a_renewal_process_is_advised_the_distribution_of_its_gaps():
    rng = np.random.default_rng(9)
    model = _quiet_fit(
        GeneralizedRenewal, *_renewal(rng, 10, 100.0, 2.0, 10.0)
    )
    d = model.diagnose()
    assert list(_rows(d, "repair")["verdict"]) == ["warn", "ok"]
    (advice,) = _advice(d, "repair", "perfect repair")
    assert "(q = 0, as good as new)" in advice
    assert "the Weibull fitted to the times between failures" in advice
    assert advice.features == ("EventHistoryData.to_gaps",)
    (trend,) = _advice(d, "trend", "")
    assert trend.startswith("No test finds a trend")
    # The advice's fit is the perfect-repair restricted fit
    from surpyval import Weibull

    gaps = Weibull.fit(model.data.to_gaps())
    np.testing.assert_allclose(
        gaps.params, model.repair_test().perfect.params[1:], rtol=1e-4
    )
    assert d.ok  # a screen never fails


@pytest.mark.parametrize(
    "fitter, name", [(GeneralizedRenewal, "q"), (ARA, "rho")]
)
def test_minimal_repair_is_advised_an_nhpp(fitter, name, no_fit_test):
    rng = np.random.default_rng(10)
    d = _quiet_fit(fitter, *_power_law(rng, 10, 100.0, 1.8, 15.0)).diagnose()
    rows = _rows(d, "repair")
    assert list(rows["verdict"]) == ["ok", "warn"]
    (advice,) = _advice(d, "repair", "minimal repair")
    value = 1 if name == "q" else 0
    assert "({} = {}, as bad as old)".format(name, value) in advice
    assert advice.features == ("CrowAMSAA.fit",)


@pytest.mark.parametrize("baseline", [CrowAMSAA, CoxLewis])
def test_ari_minimal_repair_names_its_baseline(baseline, no_fit_test):
    rng = np.random.default_rng(10)
    x, i, c = _power_law(rng, 10, 100.0, 1.8, 15.0)
    d = _quiet_fit(ARI, x, i, c, baseline=baseline).diagnose()
    rows = _rows(d, "repair")
    assert (
        rows["test"]
        .iloc[0]
        .startswith("likelihood-ratio test of maximal repair (rho = 1)")
    )
    advice = _advice(d, "repair", "minimal repair")
    name = {CrowAMSAA: "CrowAMSAA", CoxLewis: "CoxLewis"}[baseline]
    if rows["verdict"].iloc[1] == "warn":
        (advice,) = advice
        assert advice.features == ("{}.fit".format(name),)
        assert "{}.fit".format(name) in advice
    # Maximal repair is not a renewal process: never advised as one
    assert not _advice(d, "repair", "perfect repair")


def test_g1_has_the_perfect_repair_row_only(mettas, no_fit_test):
    d = _quiet_fit(GeneralizedOneRenewal, *mettas).diagnose()
    rows = _rows(d, "repair")
    assert list(rows["test"]) == [
        "likelihood-ratio test of perfect repair (q = 0)"
    ]
    assert any(
        "G1 renewal process has no minimal repair" in n for n in d.notes
    )


def test_a_bathtub_intensity_fails_the_fit_of_a_kijima_model():
    rng = np.random.default_rng(4)
    model = _quiet_fit(GeneralizedRenewal, *_bathtub(rng, 6, 30.0))
    d = model.diagnose()
    assert _rows(d, "fit")["verdict"].iloc[0] == "fail", d
    (advice,) = _advice(d, "fit", "")
    assert "dist=" in advice and "kijima='ii'" in advice
    assert "GeneralizedRenewal.fit(dist=)" in advice.features


# -- the proportional-intensity models ----------------------------------------
@pytest.fixture(scope="module")
def groups():
    return np.repeat([0.0, 1.0], 8)


def test_an_omitted_covariate_fails_between_items(groups):
    rng = np.random.default_rng(6)
    w = rng.normal(size=groups.size)
    x, i, c = _power_law(
        rng, groups.size, 100.0, 1.6, 12.0, frailty=np.exp(0.8 * groups + w)
    )
    Z = pd.DataFrame({"z": groups[i]})
    model = _quiet_fit(
        ProportionalIntensityNHPP, x, Z, i=i, c=c, baseline=CrowAMSAA
    )
    d = model.diagnose()
    assert d.family == "Recurrent events: proportional intensity"
    assert d.columns == ["z"]
    row = _rows(d, "between_items").iloc[0]
    assert row["verdict"] == "fail" and row["df"] == groups.size - 2
    assert row["covariates"] == ("z",)
    (advice,) = _advice(d, "between_items", "")
    assert advice.startswith("The items' event counts vary more than the")


def test_pi_hpp_on_a_power_law_is_advised_an_nhpp_baseline(groups):
    rng = np.random.default_rng(7)
    x, i, c = _power_law(
        rng, groups.size, 100.0, 2.0, 20.0, frailty=np.exp(0.8 * groups)
    )
    d = ProportionalIntensityHPP.fit(
        x, groups[i][:, None], i=i, c=c
    ).diagnose()
    assert list(_rows(d, "trend")["verdict"]) == ["fail"] * 3
    (advice,) = _advice(d, "trend", "")
    assert "ProportionalIntensityNHPP.fit(x, Z, ..., baseline=CrowAMSAA)" in (
        advice
    )
    assert advice.features[0] == "ProportionalIntensityNHPP.fit(baseline=)"
    assert _rows(d, "fit")["verdict"].iloc[0] == "fail"
    (fit,) = _advice(d, "fit", "")
    assert "another baseline intensity" in fit


def test_pi_nhpp_with_no_trend_is_advised_a_constant_baseline(groups):
    rng = np.random.default_rng(8)
    x, i, c = _power_law(
        rng, groups.size, 100.0, 1.0, 8.0, frailty=np.exp(0.8 * groups)
    )
    model = ProportionalIntensityNHPP.fit(
        x, groups[i][:, None], i=i, c=c, baseline=CrowAMSAA
    )
    d = model.diagnose()
    (advice,) = _advice(d, "trend", "")
    assert advice.features == ("ProportionalIntensityHPP.fit", "aic")
    assert d.columns == ["coef_0"]
    # Every check is model-wide: columns= is accepted and has no effect
    sub = model.diagnose(columns=["coef_0"])
    assert sub.to_dict()["table"] == d.to_dict()["table"]
    with pytest.raises(ValueError, match="'columns'"):
        model.diagnose(columns=["nope"])


# -- what cannot be computed --------------------------------------------------
def test_too_few_events_warn_with_the_reason():
    d = HPP.fit([3.0, 8.0], c=[0, 0]).diagnose()
    trend = _rows(d, "trend")
    assert trend["p"].isna().all() and (trend["verdict"] == "warn").all()
    assert all(
        t.startswith("could not be computed: ") for t in trend["detail"]
    )
    assert d.ok
    assert any("Between items: not tested" in n for n in d.notes)
    assert any("failure-truncated item" in n for n in d.notes)
    # a screen's unavailable test warns too
    d = CrowAMSAA.fit([3.0, 8.0], c=[0, 0]).diagnose()
    assert (_rows(d, "trend")["verdict"] == "warn").all()


def test_a_failed_cramer_von_mises_warns(monkeypatch, mettas):
    model = CrowAMSAA.fit(*mettas)

    def broken(**kwargs):
        raise ValueError("All bootstrap refits failed; cannot compute it.")

    monkeypatch.setattr(model, "cramer_von_mises", broken)
    row = _rows(model.diagnose(), "fit").iloc[0]
    assert np.isnan(row["p"]) and row["verdict"] == "warn"
    assert row["detail"] == (
        "could not be computed: All bootstrap refits failed; cannot "
        "compute it."
    )


def test_too_few_items_for_the_dispersion_test(mettas):
    x, i, c = mettas
    keep = i <= 2
    d = HPP.fit(x[keep], i[keep], c[keep]).diagnose()
    row = _rows(d, "between_items").iloc[0]
    assert row["df"] == 1 and np.isfinite(row["p"])
    keep = i == 1
    d = HPP.fit(x[keep], i[keep], c[keep]).diagnose()
    assert _rows(d, "between_items").empty
    assert any("the data have 1" in n for n in d.notes)


def test_no_degrees_of_freedom_left_warns():
    x = np.array([2.0, 5.0, 10.0, 3.0, 10.0])
    i = np.array([1, 1, 1, 2, 2])
    c = np.array([0, 0, 1, 0, 1])
    Z = np.array([0.0, 0.0, 0.0, 1.0, 1.0])[:, None]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
        d = model.diagnose()
    row = _rows(d, "between_items").iloc[0]
    assert np.isnan(row["p"]) and row["verdict"] == "warn"
    assert "no degrees of freedom" in row["detail"]


# -- refusals -----------------------------------------------------------------
def test_a_model_without_data_refuses(mettas):
    model = CrowAMSAA.fit(*mettas)
    for bare in (
        CrowAMSAA.from_params(model.params),
        sp.from_dict(model.to_dict()),
        GeneralizedRenewal.fit_from_parameters([10.0, 2.0], 0.5),
        sp.from_dict(GeneralizedRenewal.fit(*mettas).to_dict()),
    ):
        with pytest.raises(ValueError, match="needs a model fitted from data"):
            bare.diagnose()


def test_a_least_squares_fit_refuses(mettas):
    model = CrowAMSAA.fit(*mettas, how="MSE")
    with pytest.raises(ValueError, match="how='MLE'"):
        model.diagnose()


def test_a_fit_with_no_finite_maximum_refuses(mettas):
    model = CrowAMSAA.fit(*mettas)
    model.maximum = "no finite maximum"
    with pytest.raises(ValueError, match="finite maximum"):
        model.diagnose()


def test_inexact_times_and_right_truncation_refuse():
    model = CrowAMSAA.fit([[1, 2], [2, 3], [3, 4], [5, 6]], c=[2, 2, 2, 2])
    with pytest.raises(ValueError, match="exact event times"):
        model.diagnose()
    rng = np.random.default_rng(0)
    t = np.sort(rng.uniform(0, 100, 15))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = CrowAMSAA.fit(t, tr=100.0)
    with pytest.raises(ValueError, match="right truncation"):
        model.diagnose()


# -- the report ---------------------------------------------------------------
def test_to_dict_round_trips(kijima_half, mettas):
    for d in (kijima_half.diagnose(), CrowAMSAA.fit(*mettas).diagnose()):
        out = d.to_dict()
        back = Diagnosis.from_dict(
            json.loads(json.dumps(out, allow_nan=False))
        )
        assert back.to_dict() == out
        pd.testing.assert_frame_equal(back.table, d.table)
        assert [a.features for a in back.advice] == [
            a.features for a in d.advice
        ]


def test_plot_draws_every_check(kijima_half, mettas, no_fit_test):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = CrowAMSAA.fit(*mettas).diagnose()
    fig = d.plot()
    titles = [ax.get_title() for ax in fig.axes]
    assert titles[0].startswith("trend: Laplace p = 0.462")
    assert titles[1].startswith("fit (Cramer-von Mises): p = ")
    assert titles[2].startswith("between items: p = ")
    plt.close(fig)
    fig = kijima_half.diagnose().plot()
    assert [ax.get_title() for ax in fig.axes][2] == "repair: profile of q"
    plt.close(fig)
    fig = kijima_half.diagnose().plot("repair")
    (ax,) = fig.axes
    labels = ax.get_legend_handles_labels()[1]
    assert labels == [
        "95% limit",
        "estimate",
        "perfect repair",
        "minimal repair",
    ]
    plt.close(fig)
    with pytest.raises(ValueError, match="'which'"):
        d.plot("repair")


@pytest.mark.parametrize("fitter", [HPP, CrowAMSAA, Duane, CoxLewis])
def test_every_intensity_model_is_diagnosed(fitter, mettas):
    d = _quiet_fit(fitter, *mettas).diagnose()
    assert set(d.table["check"]) == {"trend", "fit", "between_items"}
    assert np.isfinite(_rows(d, "fit")["p"]).all()
