"""Comparing the links of a joint model in one call:
``JointPHFitter.compare_links`` and ``LinkComparison``.

The checks: each candidate is the fit of its own link to the same data
(the same parameters and log-likelihood as fitting it alone, though the
later fits start from the value fit's path estimates); the table's
ranking, differences, weights, tests and words; the choice of the best
(the simpler of links within 2); the verdict; ``to_dict``; the
likelihood-ratio tests of the nested pairs against the separate fits'
log-likelihoods; the options (``links``, ``metric``,
``marker_reference``); what is shown but not ranked (a fit that is not a
verified maximum, a link that could not be fitted); warnings said once;
and the refusals. The fleet is ``_fleet.simulate_links`` with the slope
link (seed 5, as in ``test_joint_links.py``). On the Mayo PBC trial the
comparison's table is that of the four links fitted one at a time.
"""

import copy
import functools
import warnings

import numpy as np
import pytest

from surpyval import JointPH, Weibull, WeibullJointPH
from surpyval.degradation import LloydLipowPath, LogarithmicPath
from surpyval.tests.univariate.joint._fleet import simulate_links
from surpyval.univariate.joint import JointPHFitter, LinkComparison
from surpyval.univariate.joint.compare import DEFAULT_LINKS, _candidates
from surpyval.utils.no_maximum import warn_unverified

LINKS = {
    "value": "value",
    "slope": "slope",
    "cumulative": "cumulative",
    "value + slope": ["value", "slope"],
    "value + cumulative": ["value", "cumulative"],
}


@functools.cache
def _data():
    return simulate_links(300, seed=5, alpha=3.0, beta=2.0, slope=1.0)


@functools.cache
def _alone(label):
    return JointPH(Weibull, link=LINKS[label]).fit(_data())


@functools.cache
def _comparison():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return WeibullJointPH.compare_links(_data())


def test_each_candidate_is_its_links_own_fit():
    comparison = _comparison()
    assert isinstance(comparison, LinkComparison)
    assert list(comparison.models) == list(LINKS)
    assert comparison.failed == {}
    for label, model in comparison.models.items():
        alone = _alone(label)
        assert model.link == alone.link
        assert model.parameter_names == alone.parameter_names
        assert model.maximum == "verified"
        # the same maximum, whichever start the search had
        gap = (model.params - alone.params) / alone.standard_errors()
        assert np.max(np.abs(gap)) < 1e-4
        assert model.log_likelihood == pytest.approx(
            alone.log_likelihood, abs=1e-6
        )


def test_the_table_ranks_the_links():
    comparison = _comparison()
    table = comparison.table
    assert list(table.columns) == [
        "k",
        "log_likelihood",
        "aic",
        "aic_c",
        "bic",
        "delta",
        "weight",
        "lr_against",
        "lr_p_value",
        "maximum",
        "status",
        "reads_as",
        "reason",
    ]
    assert table.index.name == "link"
    assert set(table.index) == set(LINKS)
    # best first, by AIC
    assert np.all(np.diff(table["aic"].to_numpy()) >= 0)
    assert table["status"].iloc[0] == "best"
    assert set(table["status"].iloc[1:]) == {"ranked"}
    assert (table["maximum"] == "verified").all()
    for label in LINKS:
        assert table.loc[label, "aic"] == pytest.approx(_alone(label).aic())
        assert table.loc[label, "bic"] == pytest.approx(_alone(label).bic())
        assert table.loc[label, "k"] == _alone(label).k
    delta = table["aic"] - table["aic"].min()
    np.testing.assert_allclose(table["delta"], delta)
    weight = np.exp(-delta / 2) / np.exp(-delta / 2).sum()
    np.testing.assert_allclose(table["weight"], weight)
    assert table["weight"].sum() == pytest.approx(1.0)


def test_the_link_the_fleet_was_simulated_with_is_best():
    comparison = _comparison()
    assert comparison.best_link == ("slope",)
    assert comparison.best_label == "slope"
    assert comparison.best is comparison.models["slope"]


def test_the_nested_pairs_are_tested_by_the_likelihood_ratio():
    tests = _comparison().likelihood_ratio_tests()
    assert list(tests.columns) == [
        "simpler",
        "richer",
        "statistic",
        "df",
        "p_value",
        "note",
    ]
    pairs = set(zip(tests["simpler"], tests["richer"]))
    assert pairs == {
        ("value", "value + slope"),
        ("slope", "value + slope"),
        ("value", "value + cumulative"),
        ("cumulative", "value + cumulative"),
    }
    from scipy.stats import chi2

    for _, row in tests.iterrows():
        gain = (
            _alone(row["richer"]).log_likelihood
            - _alone(row["simpler"]).log_likelihood
        )
        assert row["df"] == 1
        assert row["statistic"] == pytest.approx(max(2 * gain, 0.0))
        assert row["p_value"] == pytest.approx(chi2.sf(row["statistic"], 1))
        assert row["note"] == ""
    # the slope adds to the value; the value adds nothing to the slope
    by_pair = tests.set_index(["simpler", "richer"])["p_value"]
    assert by_pair[("value", "value + slope")] < 0.01
    assert by_pair[("slope", "value + slope")] > 0.05


def test_no_nested_pair_gives_an_empty_table():
    comparison = LinkComparison(
        {k: _alone(k) for k in ("value", "slope")}, {}, "aic"
    )
    tests = comparison.likelihood_ratio_tests()
    assert tests.empty
    assert list(tests.columns)[:2] == ["simpler", "richer"]


def test_the_printout_says_the_ranking():
    text = repr(_comparison())
    assert text.startswith("Joint model link comparison")
    assert "Ranked by           : aic (lower is better)" in text
    assert "Best                : slope" in text
    for label in LINKS:
        assert label in text


def test_ranked_by_bic():
    comparison = LinkComparison(
        {k: _alone(k) for k in LINKS}, {}, metric="bic"
    )
    table = comparison.table
    assert np.all(np.diff(table["bic"].to_numpy()) >= 0)
    np.testing.assert_allclose(
        table["delta"], table["bic"] - table["bic"].min()
    )
    assert "bic (lower is better)" in repr(comparison)


def test_links_and_metric_are_passed_on():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comparison = WeibullJointPH.compare_links(
            _data(),
            links=["slope", ["slope", "value"]],
            metric="bic",
            marker_reference=4.0,
        )
    assert list(comparison.models) == ["slope", "value + slope"]
    assert comparison.metric == "bic"
    # the reference goes to the links that follow the value only
    assert comparison.models["value + slope"].marker_reference == 4.0
    assert comparison.models["slope"].marker_reference is None
    np.testing.assert_array_equal(
        comparison.models["slope"].params, _alone("slope").params
    )


def test_the_fitters_own_path_and_varying_are_kept():
    fitter = JointPH(Weibull, varying="a")
    candidates, left_out, reasons = _candidates(fitter, None)
    assert [c.link for c in candidates] == list(DEFAULT_LINKS)
    assert left_out == []
    assert reasons == []
    for candidate in candidates:
        assert candidate.path_model is fitter.path_model
        assert candidate.varying == ("a",)


def test_a_path_whose_slope_is_unbounded_at_0_leaves_out_the_slope():
    fitter = JointPH(Weibull, LogarithmicPath)
    candidates, left_out, reasons = _candidates(fitter, None)
    assert [c.link for c in candidates] == [
        ("value",),
        ("cumulative",),
        ("value", "cumulative"),
    ]
    assert left_out == ["slope", "value + slope"]
    assert len(reasons) == 1 and "unbounded at age 0" in reasons[0]
    with pytest.raises(ValueError, match="unbounded at age 0"):
        _candidates(fitter, ["value", "slope"])


def test_the_comparison_says_which_links_it_left_out(monkeypatch):
    fitted = _alone("value")  # fitted before the stand-in replaces fit

    def stand_in(self, *args, **kwargs):
        return fitted

    monkeypatch.setattr(JointPHFitter, "fit", stand_in)
    with pytest.warns(UserWarning) as caught:
        JointPH(Weibull, LogarithmicPath).compare_links(_data())
    messages = [str(w.message) for w in caught]
    left = [m for m in messages if "left out of the comparison" in m]
    assert len(left) == 1
    assert "slope b / t, unbounded at age 0" in left[0]
    assert "2 link(s) were left out" in left[0]
    assert "(slope, value + slope)" in left[0]


def test_a_path_that_can_take_only_the_value_has_nothing_to_compare():
    with pytest.raises(ValueError, match="nothing to compare") as refusal:
        _candidates(JointPH(Weibull, LloydLipowPath), None)
    message = str(refusal.value)
    assert "only the value link" in message
    assert "no finite area" in message
    assert "unbounded at age 0" in message
    with pytest.raises(ValueError, match="no finite area"):
        _candidates(JointPH(Weibull, LloydLipowPath), ["value", "cumulative"])


@pytest.mark.parametrize(
    "links, match",
    [
        ("value", "'links' is a list of the links to compare"),
        (["value"], "two or more links"),
        (3, "two or more links"),
        (["value", ["value"]], "twice"),
        (["slope", ["value", "slope"], ["slope", "value"]], "twice"),
        (["value", "level"], "link names what of the marker's path"),
    ],
)
def test_links_that_cannot_be_compared_are_refused(links, match):
    with pytest.raises(ValueError, match=match):
        WeibullJointPH.compare_links(_data(), links=links)


def test_an_unknown_metric_is_refused():
    with pytest.raises(ValueError, match="'metric' must be one of"):
        WeibullJointPH.compare_links(_data(), metric="neg_ll")


def test_an_error_in_the_data_is_raised_as_the_fit_raises_it():
    with pytest.raises(ValueError, match="needs the items' marker readings"):
        WeibullJointPH.compare_links([1.0, 2.0, 3.0])


def _patched_fit(monkeypatch, change):
    """Make every fit go through ``change(fitter, model)``."""
    original = JointPHFitter.fit

    def fit(self, *args, **kwargs):
        return change(self, original(self, *args, **kwargs))

    monkeypatch.setattr(JointPHFitter, "fit", fit)


def test_a_fit_that_is_not_a_verified_maximum_is_set_aside(monkeypatch):
    def change(fitter, model):
        if fitter.link == ("slope",):
            warn_unverified("The joint fit", "a test")
            model.maximum = "unverified"
        return model

    _patched_fit(monkeypatch, change)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        comparison = WeibullJointPH.compare_links(
            _data(), links=["value", "slope", ["value", "slope"]]
        )
    messages = [str(w.message) for w in caught]
    assert len(messages) == 1, messages
    assert "set aside 1 fit(s)" in messages[0]
    assert "(slope)" in messages[0]
    assert "Best: value + slope" in messages[0]
    table = comparison.table
    assert table.loc["slope", "status"] == "set aside"
    assert table.loc["slope", "reason"] == "the fit is not a verified maximum"
    assert np.isnan(table.loc["slope", "weight"])
    assert table.index[-1] == "slope"
    assert comparison.best_label == "value + slope"
    tests = comparison.likelihood_ratio_tests()
    assert list(zip(tests["simpler"], tests["richer"])) == [
        ("value", "value + slope")
    ]


def test_no_verified_fit_has_no_best():
    models = {}
    for label in ("value", "slope"):
        models[label] = copy.copy(_alone(label))
        models[label].maximum = "unverified"
    comparison = LinkComparison(models, {}, "aic")
    assert comparison.best is None
    assert comparison.best_link is None
    assert set(comparison.table["status"]) == {"set aside"}
    assert "Best                : none" in repr(comparison)


def test_a_link_that_cannot_be_fitted_is_listed(monkeypatch):
    def change(fitter, model):
        if fitter.link == ("cumulative",):
            raise ValueError("The joint fit could not start: a test.")
        return model

    _patched_fit(monkeypatch, change)
    with pytest.warns(UserWarning, match="could not fit 1 of 3 links"):
        comparison = WeibullJointPH.compare_links(
            _data(), links=["value", "slope", "cumulative"]
        )
    assert comparison.failed == {
        "cumulative": "The joint fit could not start: a test."
    }
    assert list(comparison.models) == ["value", "slope"]
    row = comparison.table.loc["cumulative"]
    assert row["status"] == "failed"
    assert np.isnan(row["aic"])
    assert "Not fitted          : cumulative" in repr(comparison)


def test_when_no_link_fits_the_reason_is_raised(monkeypatch):
    def same(fitter, model):
        raise ValueError("The same for every link.")

    _patched_fit(monkeypatch, same)
    with pytest.raises(ValueError, match="^The same for every link.$"):
        WeibullJointPH.compare_links(_data(), links=["value", "slope"])

    def different(fitter, model):
        raise ValueError(f"Only {fitter.link[0]}.")

    monkeypatch.undo()
    _patched_fit(monkeypatch, different)
    with pytest.raises(ValueError, match="No link could be fitted"):
        WeibullJointPH.compare_links(_data(), links=["value", "slope"])


def test_a_warning_every_fit_gives_is_given_once(monkeypatch):
    def change(fitter, model):
        warnings.warn("Something about the data.", UserWarning)
        return model

    _patched_fit(monkeypatch, change)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        WeibullJointPH.compare_links(_data(), links=["value", "slope"])
    assert [str(w.message) for w in caught] == ["Something about the data."]


def test_a_richer_fit_below_the_simpler_one_is_not_tested():
    richer = copy.copy(_alone("value + slope"))
    richer._neg_ll = -_alone("value").log_likelihood + 1.0
    comparison = LinkComparison(
        {"value": _alone("value"), "value + slope": richer}, {}, "aic"
    )
    row = comparison.likelihood_ratio_tests().iloc[0]
    assert np.isnan(row["statistic"]) and np.isnan(row["p_value"])
    assert "did not reach its maximum" in row["note"]


# -- the best, the words and the verdict ------------------------------------


def _with_aic(label, aic):
    """The link's fit with its log-likelihood moved to give this AIC."""
    model = copy.copy(_alone(label))
    model._neg_ll = (aic - 2 * model.k) / 2
    for cached in ("_aic", "_aic_c", "_bic"):
        model.__dict__.pop(cached, None)
    assert model.aic() == pytest.approx(aic)
    return model


def _crafted(aics, metric="aic"):
    return LinkComparison(
        {label: _with_aic(label, aic) for label, aic in aics.items()},
        {},
        metric,
    )


def test_within_2_the_simpler_link_is_best():
    # one effect more is not worth 1.5 of AIC
    comparison = _crafted({"value": 100.0, "value + slope": 98.5})
    assert comparison.best_label == "value"
    table = comparison.table
    assert list(table.index) == ["value", "value + slope"]
    assert table.loc["value + slope", "delta"] == pytest.approx(-1.5)
    assert table.loc["value + slope", "reads_as"].startswith(
        "as good: no real preference; the rate of change adds"
    )
    assert comparison.verdict == (
        "Risk follows the marker's level. Adding the rate of change does "
        "not clearly improve the fit."
    )
    # but it is worth 2.5
    comparison = _crafted({"value": 100.0, "value + slope": 97.5})
    assert comparison.best_label == "value + slope"
    assert comparison.table.loc["value", "delta"] == pytest.approx(2.5)
    assert comparison.table.loc["value", "reads_as"] == "worse"


def test_on_a_tie_the_level_comes_first():
    comparison = _crafted({"slope": 99.0, "cumulative": 98.5, "value": 100.0})
    assert comparison.best_label == "value"
    comparison = _crafted({"slope": 99.0, "cumulative": 98.5, "value": 101.5})
    assert comparison.best_label == "slope"
    assert comparison.table["delta"].tolist() == pytest.approx(
        [0.0, -0.5, 2.5]
    )
    # the weights are the Akaike weights whichever link is best
    delta = comparison.table["delta"].to_numpy()
    weight = np.exp(-delta / 2) / np.exp(-delta / 2).sum()
    np.testing.assert_allclose(comparison.table["weight"], weight)


def test_the_rows_in_words():
    comparison = _crafted(
        {
            "slope": 100.0,
            "value + slope": 101.6,
            "value": 105.0,
            "cumulative": 126.5,
        }
    )
    reads = comparison.table["reads_as"]
    assert reads["slope"] == "best"
    assert reads["value"] == "worse"
    assert reads["cumulative"] == "clearly worse"
    # the richer link is tested against the best, nested in it
    row = comparison.table.loc["value + slope"]
    assert row["lr_against"] == "slope"
    from scipy.stats import chi2

    gain = (
        comparison.models["value + slope"].log_likelihood
        - comparison.models["slope"].log_likelihood
    )
    p = chi2.sf(2 * gain, 1)
    assert row["lr_p_value"] == pytest.approx(p)
    assert reads["value + slope"] == (
        "as good: no real preference; the level adds nothing "
        f"(test p = {p:.2g})"
    )
    for label in ("slope", "value", "cumulative"):
        assert comparison.table.loc[label, "lr_against"] == ""
        assert np.isnan(comparison.table.loc[label, "lr_p_value"])


def test_the_verdict_of_the_fleet():
    assert _comparison().verdict == (
        "Risk follows how fast the marker is changing. Adding the level "
        "does not improve the fit. The remaining 3 links fit clearly worse "
        "(AIC 13.4 or more higher)."
    )
    assert "Verdict             : Risk follows" in repr(_comparison())


def test_the_verdict_of_a_richer_best_and_of_a_tie():
    comparison = _crafted(
        {"value + slope": 100.0, "value": 104.0, "slope": 140.0}
    )
    p = comparison.table.loc["value + slope", "lr_p_value"]
    assert comparison.table.loc["value + slope", "lr_against"] == "value"
    assert comparison.verdict == (
        "Risk follows the marker's level and how fast the marker is "
        "changing. The rate of change adds to what the level alone "
        f"explains (test p = {p:.2g}). The other 2 links fit worse (AIC "
        "4.0 or more higher)."
    )
    comparison = _crafted({"value": 100.0, "slope": 99.2}, metric="aic")
    assert comparison.verdict == (
        "Risk follows the marker's level. The data cannot tell this from "
        "a risk that follows how fast the marker is changing (AIC differs "
        "by 0.8: no real preference)."
    )
    comparison = _crafted({"value": 100.0, "cumulative": 112.0})
    assert comparison.verdict == (
        "Risk follows the marker's level. A risk that follows the "
        "marker's accumulated exposure (the area under its path) fits "
        "clearly worse (AIC 12.0 higher)."
    )


def test_the_verdict_names_what_was_not_compared():
    models = {k: _alone(k) for k in ("value", "slope")}
    models["cumulative"] = copy.copy(_alone("cumulative"))
    models["cumulative"].maximum = "unverified"
    comparison = LinkComparison(
        models, {"value + slope": "could not start"}, "bic"
    )
    assert comparison.verdict.endswith(
        "Not compared: cumulative, value + slope (see the table's reason)."
    )
    assert "BIC" in comparison.verdict
    reads = comparison.table["reads_as"]
    assert reads["cumulative"] == "set aside"
    assert reads["value + slope"] == "not fitted"
    unranked = {}
    for label in ("value", "slope"):
        unranked[label] = copy.copy(_alone(label))
        unranked[label].maximum = "unverified"
    assert LinkComparison(unranked, {}, "aic").verdict.startswith(
        "No fit reached a verified maximum"
    )


def test_to_dict_is_strict_json():
    import json

    comparison = _comparison()
    out = comparison.to_dict()
    assert json.loads(json.dumps(out, allow_nan=False)) == out
    assert out["metric"] == "aic"
    assert out["best"] == "slope"
    assert out["best_link"] == ["slope"]
    assert out["verdict"] == comparison.verdict
    table = comparison.table
    assert [row["link"] for row in out["table"]] == list(table.index)
    first = out["table"][0]
    assert first["aic"] == table["aic"].iloc[0]
    assert first["k"] == table["k"].iloc[0]
    assert first["lr_p_value"] is None
    assert out["table"][1]["lr_against"] == "slope"
    assert len(out["likelihood_ratio_tests"]) == 4
    # a failed link's numbers are None
    failed = LinkComparison(
        {k: _alone(k) for k in ("value", "slope")}, {"cumulative": "no"}
    ).to_dict()
    row = failed["table"][-1]
    assert row["link"] == "cumulative"
    assert row["aic"] is None and row["k"] is None
    json.dumps(failed, allow_nan=False)


# -- warm starts --------------------------------------------------------------


def _recording(monkeypatch, starts):
    original = JointPHFitter.fit

    def fit(self, *args, **kwargs):
        starts.append((self.link, self._path_start))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(JointPHFitter, "fit", fit)


def test_later_fits_start_from_the_value_fits_path(monkeypatch):
    starts = []
    _recording(monkeypatch, starts)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comparison = WeibullJointPH.compare_links(
            _data(), links=["slope", "value", ["value", "slope"]]
        )
    # the value link first, from its own start; then the others from it
    assert [link for link, _ in starts] == [
        ("value",),
        ("slope",),
        ("value", "slope"),
    ]
    assert starts[0][1] is None
    given = comparison.models["value"]._path_start
    for _, start in starts[1:]:
        assert start is given
    # the models in the order the links were given
    assert list(comparison.models) == ["slope", "value", "value + slope"]
    assert set(given) == {"mu", "Delta", "L", "sigma"}


def test_a_warm_start_that_is_not_verified_is_fitted_again(monkeypatch):
    original = JointPHFitter.fit
    calls = []

    def fit(self, *args, **kwargs):
        calls.append((self.link, self._path_start is not None))
        model = original(self, *args, **kwargs)
        if self._path_start is not None:
            model = copy.copy(model)
            model.maximum = "unverified"
        return model

    monkeypatch.setattr(JointPHFitter, "fit", fit)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        comparison = WeibullJointPH.compare_links(
            _data(), links=["value", "slope"]
        )
    assert calls == [
        (("value",), False),
        (("slope",), True),
        (("slope",), False),
    ]
    assert comparison.models["slope"].maximum == "verified"
    np.testing.assert_array_equal(
        comparison.models["slope"].params, _alone("slope").params
    )


def test_a_start_of_another_shape_is_not_used():
    from surpyval.univariate.joint._likelihood import Layout
    from surpyval.univariate.joint.joint_ph import _given_path_start

    layout = Layout(2, 0, (0, 1))
    start = dict(_alone("value")._path_start)
    assert _given_path_start(start, layout) is not None
    start["L"] = np.eye(3)
    assert _given_path_start(start, layout) is None
    assert _given_path_start(None, layout) is None


# -- the Mayo PBC trial -------------------------------------------------------


def test_the_pbc_table_is_the_four_links_fitted_alone():
    from surpyval.tests.reference.test_joint_links import _patients

    patients = _patients()
    options = dict(hazard_covariates=["treated"], path_covariates=[])
    links = ["value", "slope", ["value", "slope"], "cumulative"]
    comparison = WeibullJointPH.compare_links(
        patients["years"].to_numpy(),
        c=patients["censored"].to_numpy(),
        Z=patients[["treated"]],
        markers=list(patients["readings"]),
        links=links,
        **options,
    )
    table = comparison.table
    for link in links:
        alone = JointPH(Weibull, link=link).fit_from_df(
            patients,
            x_col="years",
            c_col="censored",
            markers_col="readings",
            Z_cols=["treated"],
            **options,
        )
        model = comparison.models[" + ".join(alone.link)]
        row = table.loc[" + ".join(alone.link)]
        assert alone.maximum == model.maximum == "verified"
        gap = (model.params - alone.params) / alone.standard_errors()
        assert np.max(np.abs(gap)) < 1e-3
        assert row["log_likelihood"] == pytest.approx(
            alone.log_likelihood, abs=1e-5
        )
        assert row["aic"] == pytest.approx(alone.aic(), abs=1e-4)
        assert row["bic"] == pytest.approx(alone.bic(), abs=1e-4)
        assert row["k"] == alone.k
    # both the level and its rise carry information
    assert comparison.best_label == "value + slope"
    assert table.loc["value + slope", "lr_against"] == "value"
    assert table.loc["value + slope", "lr_p_value"] < 0.01
    assert comparison.verdict.startswith(
        "Risk follows the marker's level and how fast the marker is "
        "changing. The rate of change adds to what the level alone "
        "explains"
    )
