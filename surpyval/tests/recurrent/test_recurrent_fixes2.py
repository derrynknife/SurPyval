"""Regression tests for the second round of recurrent-event fixes."""

import warnings

import matplotlib
import numpy as np
import pytest

from surpyval.datasets import load_rossi_static
from surpyval.recurrent import (
    ARI,
    HPP,
    CauseSpecificMCF,
    CauseSpecificNHPP,
    CrowAMSAA,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
    NonParametricCounting,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)
from surpyval.recurrent.renewal.renewal_model import RenewalModel

matplotlib.use("Agg")

# Three items; item 1 has far more events than the others, so the robust
# (Lawless-Nadeau) variance is well above the per-step one.
X = [1, 2, 3, 4, 5, 6, 7, 3, 6, 9, 2, 9]
I = [1] * 7 + [2] * 3 + [3] * 2
C = [0] * 6 + [1] + [0, 0, 1] + [0, 1]

FLEET_X = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44, 60]
FLEET_I = [1] * 6 + [2] * 5
FLEET_C = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]


def _lawless_nadeau_by_hand(x, i, c, e, cause):
    """Brute-force Lawless-Nadeau variance of one cause's MCF: each item's
    deviations n_k - d/r, weighted by 1/r while at risk, summed over time
    and then squared."""
    x, i, c, e = map(np.asarray, (x, i, c, e))
    grid = np.unique(x)
    items = np.unique(i)
    exit_ = {k: x[i == k].max() for k in items}
    r = np.array([sum(t <= exit_[k] for k in items) for t in grid])
    counts = {
        k: np.array(
            [
                np.sum((x == t) & (i == k) & (c == 0) & (e == cause))
                for t in grid
            ]
        )
        for k in items
    }
    d = sum(counts.values())
    total = np.zeros(len(grid))
    for k in items:
        at_risk = grid <= exit_[k]
        total += np.cumsum(at_risk * (counts[k] - d / r) / r) ** 2
    return total


# -- 1: cause-specific MCF variance ------------------------------------------


def test_cause_specific_mcf_single_cause_matches_overall_mcf():
    # With every event of one cause, the cause-specific MCF is the overall
    # MCF, variance included. It used to carry the per-step variance
    # (0.556 at the end, against the robust 1.556).
    e = ["A" if ci == 0 else None for ci in C]
    overall = NonParametricCounting.fit(X, I, C)
    cs = CauseSpecificMCF.fit(X, I, C, e=e).models["A"]
    np.testing.assert_allclose(cs.mcf_hat, overall.mcf_hat)
    np.testing.assert_allclose(cs.var, overall.var)
    assert cs.var[-1] == pytest.approx(14 / 9)


def test_cause_specific_mcf_robust_variance_per_cause():
    # Other causes' events are non-events for a cause; the risk set is
    # shared.
    e = ["A", "B", "A", "A", "B", "A", None, "B", "A", None, "A", None]
    model = CauseSpecificMCF.fit(X, I, C, e=e)
    for cause in ("A", "B"):
        np.testing.assert_allclose(
            model.models[cause].var,
            _lawless_nadeau_by_hand(X, I, C, e, cause),
        )


# -- 2: per-step variance with ties ------------------------------------------


def test_from_xrd_per_step_variance_with_ties():
    # Two events among three items at risk: d (r - d) / r^3 = 2/27. The
    # old formula centred on 1/r and gave 1/9.
    model = NonParametricCounting.from_xrd([1.0], [3], [2])
    np.testing.assert_allclose(model.var, [2 / 27])


def test_from_xrd_single_event_steps_unchanged():
    r = np.array([5, 4, 3])
    model = NonParametricCounting.from_xrd([1.0, 2.0, 3.0], r, [1, 1, 1])
    np.testing.assert_allclose(model.var, np.cumsum((r - 1) / r**3))


def test_from_xrd_more_events_than_at_risk_gives_nan_variance():
    # The triple cannot say how three events shared out over two items.
    model = NonParametricCounting.from_xrd([1.0, 2.0], [4, 2], [1, 3])
    assert model.var[0] == pytest.approx(3 / 64)
    assert np.isnan(model.var[1])


# -- 3: data-requiring methods on models without data ------------------------


def _rossi():
    data = load_rossi_static()
    x = data["week"].values
    c = data["arrest"].values
    i = np.arange(len(x))
    Z = data[["fin", "age"]].values
    return x, Z, i, c


def _data_less_models():
    x, Z, i, c = _rossi()
    pi_nhpp = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, dist=CrowAMSAA)
    pi_hpp = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    ca = CrowAMSAA.fit(FLEET_X, FLEET_I, FLEET_C)
    g1 = GeneralizedOneRenewal.fit(FLEET_X, FLEET_I, FLEET_C)
    return {
        "HPP.from_params": HPP.from_params([0.5]),
        "CrowAMSAA restored": type(ca).from_dict(ca.to_dict()),
        "PI-NHPP restored": type(pi_nhpp).from_dict(pi_nhpp.to_dict()),
        "PI-HPP restored": type(pi_hpp).from_dict(pi_hpp.to_dict()),
        "G1 restored": RenewalModel.from_dict(g1.to_dict()),
        "GRP fit_from_parameters": GeneralizedRenewal.fit_from_parameters(
            [10, 2], 0.3
        ),
    }


@pytest.mark.parametrize(
    "method", ["residuals", "trend_test", "cramer_von_mises", "plot"]
)
def test_data_less_models_raise_informative_error(method):
    for name, model in _data_less_models().items():
        with pytest.raises(ValueError, match="requires a model fitted"):
            getattr(model, method)()


def test_restored_cause_specific_nhpp_plot_raises_informative_error():
    e = ["A", "B", "A", "B", "A", None, "A", "B", "A", "A", None]
    model = CauseSpecificNHPP.fit(FLEET_X, FLEET_I, FLEET_C, e=e)
    restored = CauseSpecificNHPP.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="requires a model fitted"):
        restored.plot()


# -- 4: right truncation in the non-parametric MCF ---------------------------


def test_mcf_right_truncation_equals_censoring_row_at_tr():
    x = [1, 2, 3, 2, 5, 4]
    i = [1, 1, 1, 2, 2, 3]
    tr = [6, 6, 6, 6, 6, 8]
    truncated = NonParametricCounting.fit(x, i, tr=tr)
    censored = NonParametricCounting.fit(
        x + [6, 6, 8], i + [1, 2, 3], [0] * 6 + [1, 1, 1]
    )
    np.testing.assert_array_equal(truncated.x, censored.x)
    np.testing.assert_array_equal(truncated.r, censored.r)
    np.testing.assert_allclose(truncated.mcf_hat, censored.mcf_hat)
    np.testing.assert_allclose(truncated.var, censored.var)
    # item 3 is watched to 8, past its only event at 4
    assert truncated.mcf(7) == pytest.approx(censored.mcf(7))
    assert np.isnan(truncated.mcf(9))


def test_cause_specific_mcf_right_truncation_equals_censoring_row_at_tr():
    x = [1, 2, 3, 2, 5, 4]
    i = [1, 1, 1, 2, 2, 3]
    e = ["A", "B", "A", "B", "A", "A"]
    truncated = CauseSpecificMCF.fit(x, i, e=e, tr=[6, 6, 6, 6, 6, 8])
    censored = CauseSpecificMCF.fit(
        x + [6, 6, 8],
        i + [1, 2, 3],
        [0] * 6 + [1, 1, 1],
        e=e + [None] * 3,
    )
    for cause in ("A", "B"):
        np.testing.assert_allclose(
            truncated.models[cause].mcf_hat, censored.models[cause].mcf_hat
        )
        np.testing.assert_allclose(
            truncated.models[cause].var, censored.models[cause].var
        )


# -- 5: warning-free renewal fits --------------------------------------------


def test_g1_weibull_fit_emits_no_warnings():
    x = np.array([3, 6, 11, 5, 16, 9, 19, 22, 37, 23, 31, 45]).cumsum()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = GeneralizedOneRenewal.fit(x)
        GeneralizedOneRenewal.fit(FLEET_X, FLEET_I, FLEET_C)
    assert model.q == pytest.approx(0.2163, abs=1e-3)


def test_ari_fit_skips_infeasible_start_without_warnings():
    # The rho = 0.9 start drives the intensity negative (zero likelihood);
    # Nelder-Mead from it used to warn about inf - inf.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = ARI.fit(FLEET_X, FLEET_I, FLEET_C, m=1)
    assert 0 < model.rho < 1


def test_infeasible_user_init_is_reported():
    with pytest.raises(ValueError, match="zero likelihood"):
        ARI.fit(FLEET_X, FLEET_I, FLEET_C, m=1, init=[0.9, 7.8, 0.74])


# -- 6: one BIC sample size for every recurrent model ------------------------


def test_bic_counts_observed_events_only():
    n_events = sum(ci == 0 for ci in FLEET_C)
    models = [
        HPP.fit(FLEET_X, FLEET_I, FLEET_C),
        CrowAMSAA.fit(FLEET_X, FLEET_I, FLEET_C),
        GeneralizedOneRenewal.fit(FLEET_X, FLEET_I, FLEET_C),
        ARI.fit(FLEET_X, FLEET_I, FLEET_C, m=1),
    ]
    for model in models:
        k = model._mle.size
        expected = k * np.log(n_events) - 2 * model.log_likelihood
        assert model.bic == pytest.approx(expected)


# -- 7: renewal goodness-of-fit bootstrap keeps each item's scheme -----------


def test_renewal_cvm_resimulates_time_truncated_items_to_their_window(
    monkeypatch,
):
    # Item 1 is time truncated at 60, item 2 failure truncated at 44.
    x = [3, 9, 20, 35, 56, 60, 4, 11, 25, 44]
    i = [1] * 6 + [2] * 4
    c = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    model = GeneralizedOneRenewal.fit(x, i, c)

    simulated = []
    fitter = model._fitter
    original_refit = fitter._refit

    def spy(fitted, data):
        simulated.append(data)
        return original_refit(fitted, data)

    monkeypatch.setattr(fitter, "_refit", spy)
    model.cramer_von_mises(n_boot=5, seed=3)

    assert simulated
    for data in simulated:
        x1, c1, _ = data.get_events_for_item(1)
        x2, c2, _ = data.get_events_for_item(2)
        # time truncated: ends in a c=1 row at 60, random event count
        assert c1[-1] == 1 and x1[-1] == 60
        assert np.all(x1[:-1] < 60)
        # failure truncated: the observed four events, all exact
        assert len(x2) == 4 and np.all(c2 == 0)


def test_bic_counts_interval_events():
    # Only interval counts: the five events they hold are observed events,
    # so BIC's sample size is 5 (it used to count exact events only, and
    # was NaN here rather than log(0) = -inf).
    model = HPP.fit([[0, 10], [10, 20]], c=[2, 2], n=[2, 3])
    assert model.bic == pytest.approx(np.log(5) - 2 * model.log_likelihood)
    assert np.isfinite(model.aic)
