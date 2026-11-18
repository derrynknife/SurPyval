"""Comparing the links of a joint model in one call:
``JointPHFitter.compare_links`` and ``LinkComparison``.

The checks: each candidate is the fit of its own link to the same data
(the same parameters and log-likelihood as fitting it alone); the table's
ranking, differences and weights; the likelihood-ratio tests of the
nested pairs against the separate fits' log-likelihoods; the options
(``links``, ``metric``, ``marker_reference``); what is shown but not ranked
(a fit that is not a verified maximum, a link that could not be fitted);
warnings said once; and the refusals. The fleet is
``_fleet.simulate_links`` with the slope link (seed 5, as in
``test_joint_links.py``).
"""

import copy
import functools
import warnings

import numpy as np
import pytest

from surpyval import JointPH, Weibull, WeibullJointPH
from surpyval.degradation import LloydLipowPath
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
        np.testing.assert_array_equal(model.params, alone.params)
        assert model.log_likelihood == alone.log_likelihood


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
        "maximum",
        "status",
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
    candidates, left_out = _candidates(fitter, None)
    assert [c.link for c in candidates] == list(DEFAULT_LINKS)
    assert left_out == []
    for candidate in candidates:
        assert candidate.path_model is fitter.path_model
        assert candidate.varying == ("a",)


def test_a_path_with_no_finite_area_leaves_out_the_cumulative():
    candidates, left_out = _candidates(JointPH(Weibull, LloydLipowPath), None)
    assert [c.link for c in candidates] == [
        ("value",),
        ("slope",),
        ("value", "slope"),
    ]
    assert left_out == ["cumulative", "value + cumulative"]
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
