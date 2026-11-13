"""Bootstrap refits with no finite maximum, on every bootstrap path.

A resample can leave the likelihood with no finite maximum: no failures,
fewer distinct failure times than parameters, a covariate group with no
failures. Every bootstrap that refits resamples treats such refits the
same way:

* a refit that returns with ``maximum == "no finite maximum"`` is kept
  at the estimate it reached, so that it counts toward the end of the
  bounds it ran to (the rule since #617);
* a resample the fit refuses because its likelihood has no maximum at
  all (``NoFiniteMaximumError``) is left out, as it has no estimate;
* both are counted, in one ``BootstrapRefitWarning`` that says how many
  of how many, what was done with them and what to do, whenever there is
  at least one (not only above 2% of the refits, as before), with the
  counts on the warning;
* data whose refits all reach a verified maximum give the same bounds as
  before, and no warning.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.utils.no_maximum import (
    BootstrapRefitWarning,
    NoFiniteMaximumError,
    warn_bootstrap_refits,
)


def _caught(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = fn(*args, **kwargs)
    return out, [
        w.message
        for w in caught
        if isinstance(w.message, BootstrapRefitWarning)
    ]


# -- the resamples themselves (hand-checked) ---------------------------------


@pytest.mark.parametrize(
    "x, c",
    [
        # no failures: the likelihood keeps rising as the scale runs out
        ([1.0, 2.0, 3.0, 4.0], [1, 1, 1, 1]),
        # one distinct failure time for two parameters: beta runs to inf
        ([2.0, 2.0, 3.0, 4.0], [0, 0, 1, 1]),
        # only left-censored: the distribution runs down to 0
        ([1.0, 2.0, 3.0], [-1, -1, -1]),
    ],
)
def test_resamples_with_no_maximum_are_refused_as_such(x, c):
    with pytest.raises(NoFiniteMaximumError):
        sp.Weibull.fit(x, c)
    # still a ValueError, so that code catching that is unchanged
    assert issubclass(NoFiniteMaximumError, ValueError)


def test_a_one_parameter_fit_to_one_failure_has_a_maximum():
    # The exponential's maximum, d / total time = 1 / 10, exists.
    model = sp.Exponential.fit([1.0, 2.0, 3.0, 4.0], [0, 1, 1, 1])
    assert model.maximum == "verified"
    assert model.params[0] == pytest.approx(0.1)


# -- the warning -------------------------------------------------------------


def test_warning_counts_and_wording():
    _, msgs = _caught(
        warn_bootstrap_refits,
        200,
        no_maximum=1,
        no_estimate=3,
        unverified=1,
        failed=2,
    )
    assert len(msgs) == 1
    w = msgs[0]
    assert (w.n_boot, w.no_maximum, w.no_estimate, w.unverified, w.failed) == (
        200,
        1,
        3,
        1,
        2,
    )
    text = str(w)
    assert text.startswith("7 of the 200 bootstrap refits")
    assert "1 with no finite maximum, kept" in text
    assert "3 on resamples whose likelihood has no maximum at all" in text
    assert "left out" in text and "narrows the bounds" in text
    assert "1 at an unverified maximum" in text
    assert "2 failed for another reason" in text


def test_one_refit_with_no_maximum_is_reported():
    # Below the 2% that unverified refits are allowed, still reported.
    _, msgs = _caught(warn_bootstrap_refits, 1000, no_maximum=1)
    assert len(msgs) == 1 and str(msgs[0]).startswith("1 of the 1000")
    _, msgs = _caught(warn_bootstrap_refits, 1000, no_estimate=1)
    assert len(msgs) == 1
    # unverified or failed refits alone keep the 2% allowance
    _, msgs = _caught(warn_bootstrap_refits, 1000, unverified=20)
    assert msgs == []
    _, msgs = _caught(warn_bootstrap_refits, 1000, unverified=21)
    assert len(msgs) == 1


# -- univariate parametric ---------------------------------------------------

#: 3 failures among 10 units: a simulated resample often has fewer than 2
#: distinct failures, where the Weibull has no maximum.
SPARSE = ([1.0, 2.0, 3.0] + [10.0] * 7, [0, 0, 0] + [1] * 7)


def test_parametric_bounds_report_resamples_with_no_maximum():
    model = sp.Weibull.fit(*SPARSE)
    cb, msgs = _caught(
        model.cb, [2.0, 5.0], method="bootstrap", n_boot=200, random_state=1
    )
    assert len(msgs) == 1
    w = msgs[0]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the same warning, said again
        boot = model.bootstrap(200, random_state=1)
    # the same refits, the same counts, on the warning and the result
    assert w.no_estimate == boot.no_estimate > 0
    assert w.no_maximum == boot.no_maximum
    assert len(boot) == 200 - boot.failed
    assert str(w).startswith(
        f"{w.no_estimate + w.no_maximum + w.unverified + w.failed} of the 200"
    )
    assert boot.to_dict()["no_estimate"] == boot.no_estimate
    assert np.all(np.isfinite(cb))


def test_parametric_refits_refused_for_no_maximum_are_counted_apart(
    monkeypatch,
):
    model = sp.Weibull.fit([3.1, 4.7, 5.2, 6.9, 8.4, 9.0, 11.3, 12.8])
    fit = sp.Weibull.fit
    calls = {"n": 0}

    def sometimes(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 5 == 0:
            raise NoFiniteMaximumError("no failures")
        if calls["n"] % 7 == 0:
            raise RuntimeError("something else")
        return fit(*args, **kwargs)

    monkeypatch.setattr(model.dist, "fit", sometimes)
    _, msgs = _caught(
        model.cb, 5.0, method="bootstrap", n_boot=40, random_state=0
    )
    # calls 5, 10, ..., 40 refused for no maximum (8); 7, 14, 21, 28
    # failed otherwise (35 is a multiple of 5)
    assert len(msgs) == 1
    assert (msgs[0].no_estimate, msgs[0].failed) == (8, 4)
    assert str(msgs[0]).startswith("12 of the 40")


def test_parametric_kept_refit_with_no_maximum_is_reported_and_kept(
    monkeypatch,
):
    # One refit labelled "no finite maximum" (1 of 50, under 2%): the
    # bounds are those of the same refits unlabelled, and it is reported.
    x = [3.1, 4.7, 5.2, 6.9, 8.4, 9.0, 11.3, 12.8]
    plain = sp.Weibull.fit(x)
    before, msgs = _caught(
        plain.cb, 5.0, method="bootstrap", n_boot=50, random_state=0
    )
    assert msgs == []
    model = sp.Weibull.fit(x)
    fit = sp.Weibull.fit
    calls = {"n": 0}

    def labelled(*args, **kwargs):
        calls["n"] += 1
        out = fit(*args, **kwargs)
        if calls["n"] == 3:
            out.maximum = "no finite maximum"
        return out

    monkeypatch.setattr(model.dist, "fit", labelled)
    after, msgs = _caught(
        model.cb, 5.0, method="bootstrap", n_boot=50, random_state=0
    )
    np.testing.assert_array_equal(after, before)
    assert len(msgs) == 1 and msgs[0].no_maximum == 1
    assert "1 with no finite maximum, kept" in str(msgs[0])


def test_parametric_normal_data_bounds_unchanged_and_quiet():
    x = sp.Weibull.random(40, 10, 2, random_state=3)
    model = sp.Weibull.fit(x)
    cb, msgs = _caught(
        model.cb, [5.0, 10.0], method="bootstrap", n_boot=100, random_state=4
    )
    assert msgs == []
    fits = model._bootstrap_refits[(100, 4)]
    assert fits.no_maximum == fits.no_estimate == fits.failed == 0
    assert len(fits.models) == 100


# -- competing risks (a cause's net bounds are its model's) ------------------


def test_competing_risks_cause_bounds_report_no_maximum():
    x = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    e = ["a", "b", "a", "a", "b", "a", "a", "a", "a", "a"]
    model = sp.ParametricCompetingRisks.fit(x, e, dist=sp.Weibull)
    _, msgs = _caught(
        model.cb,
        5.0,
        event="b",
        on="sf",
        method="bootstrap",
        n_boot=100,
        random_state=0,
    )
    # cause b has 2 failures among 10 units: many resamples have fewer
    # than 2 distinct b failures
    assert len(msgs) == 1 and msgs[0].no_estimate > 0


# -- regressions -------------------------------------------------------------


def _ph_sparse_group():
    # Two failures in the Z = 1 group: resamples with none there run the
    # coefficient off (no finite maximum).
    rng = np.random.default_rng(2)
    Z = np.repeat([0.0, 1.0], 20)
    t = 10 * rng.weibull(1.5, 40) * np.exp(-1.5 * Z)
    x, c = np.minimum(t, 10.0), (t > 10.0).astype(int)
    c[Z == 1] = 1
    c[np.flatnonzero(Z == 1)[:2]] = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.WeibullPH.fit(x, Z, c=c)


def test_regression_separated_refits_are_kept_and_counted():
    model = _ph_sparse_group()
    (lo, hi), msgs = _caught(
        model.param_cb, "coef_0", method="bootstrap", n_boot=40, random_state=0
    )
    fits = model._bootstrap_refits[(40, 0)]
    assert len(msgs) == 1
    assert msgs[0].no_maximum == fits.no_maximum > 0
    assert fits.params.shape[0] == 40  # kept
    # kept at the estimate they ran off to: the lower bound is far out
    assert lo < -5 < model.params[2] < hi


def test_regression_resamples_refused_for_no_maximum(monkeypatch):
    x = sp.Weibull.random(30, 10, 2, random_state=1)
    Z = np.tile([0.0, 1.0], 15)
    model = sp.WeibullPH.fit(x, Z)
    fit = model.model.fit
    calls = {"n": 0}

    def sometimes(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 4 == 0:
            raise NoFiniteMaximumError("no failures")
        return fit(*args, **kwargs)

    monkeypatch.setattr(model.model, "fit", sometimes)
    _, msgs = _caught(
        model.cb, 5.0, [1.0], method="bootstrap", n_boot=20, random_state=0
    )
    assert len(msgs) == 1
    assert (msgs[0].no_estimate, msgs[0].failed) == (5, 0)
    assert model._bootstrap_refits[(20, 0)].params.shape[0] == 15


# -- degradation -------------------------------------------------------------


def _analysis():
    rng = np.random.default_rng(7)
    times = np.arange(1.0, 7.0)
    x, y, i = [], [], []
    for u in range(12):
        slope = np.exp(rng.normal(0.0, 0.3))
        x.append(times)
        y.append(slope * times * np.exp(rng.normal(0.0, 0.03, times.size)))
        i.append(np.full(times.size, u))
    x, y, i = map(np.concatenate, (x, y, i))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.DegradationAnalysis.fit(x, y, i, threshold=10.0)


def test_degradation_bootstrap_counts_refits_without_a_maximum(monkeypatch):
    from surpyval.degradation import degradation_analysis as da_module

    model = _analysis()
    t = np.array([6.0, 10.0])
    before, msgs = _caught(
        model.cb, t, method="bootstrap", n_boot=20, random_state=3
    )
    assert msgs == []
    fitter = da_module.DegradationAnalysis
    fit = fitter.fit
    calls = {"n": 0}

    def sometimes(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise NoFiniteMaximumError("no pseudo failures")
        out = fit(*args, **kwargs)
        if calls["n"] == 6:
            out.life_model.maximum = "no finite maximum"
        return out

    monkeypatch.setattr(fitter, "fit", sometimes)
    _, msgs = _caught(
        model.cb, t, method="bootstrap", n_boot=20, random_state=3
    )
    assert len(msgs) == 1
    assert (msgs[0].no_estimate, msgs[0].no_maximum) == (1, 1)
    assert str(msgs[0]).startswith("2 of the 20")


def test_destructive_noise_free_refits_warn_once():
    # Noise-free readings: every resample of them is noise free too, so
    # every refit has no finite maximum (sigma runs to 0). They used to
    # warn once each, 30 times; now once, with the count.
    t = np.repeat([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 3)
    y = np.exp(1 + 0.05 * t)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.DestructiveDegradation.fit(t, y, 10.0)
    assert model.maximum == "no finite maximum"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model.cb([30.0], n_boot=30, random_state=0)
    msgs = [w.message for w in caught]
    assert len(msgs) == 1, [str(m) for m in msgs]
    assert isinstance(msgs[0], BootstrapRefitWarning)
    assert msgs[0].no_maximum == 30


def test_destructive_normal_data_quiet():
    rng = np.random.default_rng(1)
    x = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
    y = np.exp(4.0 - 0.02 * x + rng.normal(0, 0.1, 24))
    model = sp.DestructiveDegradation.fit(x, y, threshold=20)
    _, msgs = _caught(model.cb, [30.0], n_boot=40, random_state=0)
    assert msgs == []


# -- goodness-of-fit bootstraps (p-values, said in the report's notes) -------


def test_univariate_goodness_of_fit_counts_no_maximum():
    from surpyval.diagnostics.univariate import bootstrap, no_maximum_note

    model = sp.Weibull.fit(*SPARSE)
    boot = bootstrap(model, 100, random_state=1)
    assert boot["no_estimate"] > 0
    assert boot["failed"] >= boot["no_estimate"]
    note = no_maximum_note(boot)
    total = boot["no_estimate"] + boot["no_maximum"]
    assert note.startswith(f"{total} of the 100 bootstrap refits had no")
    assert no_maximum_note({"n_boot": 10, "failed": 0}) is None
