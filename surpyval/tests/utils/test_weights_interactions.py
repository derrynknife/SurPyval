"""Sampling weights ``w`` where they meet the other parts of SurPyval:
the global likelihood-ratio test, the life table, the Bayesian and PyMC
fits, and the serialised data. Each of these used to drop the weights or
misread them; each now either uses them correctly or refuses, saying what
to use instead. (The robust covariances of weighted regression fits are
checked against R in ``reference/test_weights.py``.)"""

import json

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import surpyval as sp
from surpyval.tests._helpers import weighted_reference_data
from surpyval.utils.weights import WeightedRefusal


@pytest.fixture(scope="module")
def data():
    return weighted_reference_data()


@pytest.fixture(scope="module")
def weighted(data):
    return sp.SurpyvalData(data["x"], data["c"], w=data["w"])


# -- likelihood_ratio_test -------------------------------------------------


@pytest.mark.parametrize("name", ["CoxPH", "WeibullPH", "WeibullAFT"])
def test_likelihood_ratio_test_refuses_weighted_fits(data, name):
    # On Cox the statistic grew with the scale of the weights (33.0, then
    # 330 with the weights times 10); on WeibullPH the null model was
    # fitted without the weights, which gave a statistic of 0 with the
    # null model's log-likelihood above the fit's.
    model = getattr(sp, name).fit(
        data["x"], data["Z"], c=data["c"], w=data["w"]
    )
    with pytest.raises(WeightedRefusal) as err:
        model.likelihood_ratio_test()
    message = str(err.value)
    assert "not chi-squared" in message
    assert "Wald" in message and "robust covariance" in message


def test_likelihood_ratio_test_unweighted_is_unchanged(data):
    # The unweighted null model is the baseline fitted to the same rows,
    # its log-likelihood below the fit's, with or without late entry.
    for tl in (None, data["tl"]):
        model = sp.WeibullPH.fit(data["x"], data["Z"], c=data["c"], tl=tl)
        res = model.likelihood_ratio_test()
        null = sp.Weibull.fit(data["x"], data["c"], tl=tl)
        assert res["log_likelihood_null"] == pytest.approx(
            -null.neg_ll(), rel=1e-9
        )
        assert res["log_likelihood_null"] < res["log_likelihood"]


def test_likelihood_ratio_test_refuses_a_fit_below_its_null(data):
    # A fit that has not reached its maximum, below the model without
    # covariates nested in it, used to give a statistic of 0 (p = 1).
    model = sp.WeibullPH.fit(data["x"], data["Z"], c=data["c"])
    null = model.likelihood_ratio_test()["log_likelihood_null"]
    model.neg_ll = lambda: -null + 1.0  # one unit below the null model
    with pytest.raises(ValueError, match="did not reach its maximum"):
        model.likelihood_ratio_test()


# -- life_table ------------------------------------------------------------


def test_life_table_refuses_weighted_data(data, weighted):
    km = sp.KaplanMeier.fit(data["x"], data["c"], w=data["w"])
    for given in (weighted, km, sp.from_dict(km.to_dict())):
        with pytest.raises(WeightedRefusal, match="life_table") as err:
            sp.life_table(given)
        assert "KaplanMeier.fit" in str(err.value)
    # The arrays without the weights still give the table of the units
    table = sp.life_table(data["x"], data["c"])
    assert table.equals(sp.life_table(sp.SurpyvalData(data["x"], data["c"])))


# -- Bayesian and PyMC fits ------------------------------------------------


def test_fit_bayes_refuses_weighted_data(data, weighted):
    # It used to fit the weighted likelihood as if the weights were
    # units seen.
    with pytest.raises(WeightedRefusal, match="fit_bayes") as err:
        sp.Weibull.fit_bayes(weighted)
    assert "Weibull.fit(data)" in str(err.value)
    with pytest.raises(WeightedRefusal, match="fit_bayes"):
        sp.Exponential.fit_bayes(weighted)


def test_bayesian_entry_points_refuse_weighted_data(data, weighted):
    from surpyval.bayes import BayesianParametric, log_likelihood_fn

    with pytest.raises(WeightedRefusal, match="log_likelihood_fn"):
        log_likelihood_fn(sp.Weibull, weighted)
    with pytest.raises(WeightedRefusal, match="prior.from_data"):
        sp.prior.from_data(weighted, dist=sp.Weibull)
    model = sp.Weibull.fit_bayes(data["x"], data["c"])
    with pytest.raises(WeightedRefusal, match="update"):
        model.update(weighted)
    draws = np.column_stack([np.full(10, 7.0), np.linspace(1, 1.2, 10)])
    with pytest.raises(WeightedRefusal, match="from_draws"):
        BayesianParametric.from_draws(
            sp.Weibull, draws, ["alpha", "beta"], data=weighted
        )


def test_pymc_helpers_refuse_weighted_data(weighted):
    from surpyval.pymc import to_surpyval
    from surpyval.pymc.likelihood import as_surpyval_data

    with pytest.raises(WeightedRefusal, match="log_likelihood"):
        as_surpyval_data("log_likelihood", weighted)
    with pytest.raises(WeightedRefusal, match="to_surpyval"):
        to_surpyval(None, sp.Weibull, data=weighted)


def test_unweighted_bayesian_fit_is_unchanged(data):
    # The same data without weights still fits
    model = sp.Weibull.fit_bayes(sp.SurpyvalData(data["x"], data["c"]))
    again = sp.Weibull.fit_bayes(data["x"], data["c"])
    assert_array_equal(model.params, again.params)


# -- serialised data -------------------------------------------------------


def test_weighted_data_dict_is_schema_3_and_round_trips(data):
    given = sp.SurpyvalData(
        data["x"], data["c"], tl=data["tl"], Z=data["Z"], w=data["w"]
    )
    d = json.loads(json.dumps(given.to_dict(), allow_nan=False))
    # A weighted model's dict is schema 3 too
    model = sp.Weibull.fit(data["x"], data["c"], w=data["w"])
    assert d["schema"] == model.to_dict()["schema"] == 3
    for again in (sp.SurpyvalData.from_dict(d), sp.from_dict(d)):
        assert_array_equal(again.w, given.w)
        assert_array_equal(again.x, given.x)
        assert_array_equal(again.t, given.t)
        assert_array_equal(again.n_o, given.n_o)
        assert_array_equal(again.Z, given.Z)
    # The fit of the restored data is the fit of the data
    assert_array_equal(
        sp.WeibullPH.fit(sp.SurpyvalData.from_dict(d)).params,
        sp.WeibullPH.fit(given).params,
    )


def test_unweighted_data_dict_is_unchanged():
    given = sp.SurpyvalData([1, 2, 2, 5], c=[0, 0, 0, 1], tl=0.5)
    d = given.to_dict()
    assert "schema" not in d and "w" not in d
    assert sorted(d) == ["c", "class", "n", "non_finite", "t", "x"]
    assert d == sp.SurpyvalData.from_dict(d).to_dict()


def test_data_dict_readers_by_schema(data):
    given = sp.SurpyvalData(data["x"], data["c"], w=data["w"])
    d = given.to_dict()
    # A weighted dict written before the stamp keeps its weights
    unstamped = {k: v for k, v in d.items() if k != "schema"}
    assert_array_equal(sp.SurpyvalData.from_dict(unstamped).w, given.w)
    # A newer layout asks for an upgrade, rather than being guessed at
    for read in (sp.SurpyvalData.from_dict, sp.from_dict):
        with pytest.raises(ValueError, match="Upgrade surpyval"):
            read({**d, "schema": 4})
