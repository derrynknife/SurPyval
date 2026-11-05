"""``with_params``, ``covariance_names`` and ``_estimates`` on the
regression models: the parametric families (PH, AFT, PO, AH, accelerated
life, Royston-Parmar, distributional) and the semi-parametric
proportional odds and Lin-Ying additive hazards models.

The contract (shared with the univariate and process models): a flat
vector in the order of ``covariance_names`` gives a new model of the same
kind, with the structure (formula and its encoding, centre, fixed
parameters, aliased coefficients) kept and no data, covariance or
likelihood; ``with_params(_estimates())`` predicts as the fit does; the
parameters of a refit, given to ``with_params``, predict as the refit
does; and draws from the covariance, made models by ``with_params``,
spread as the delta method says they do. Seeds are fixed.
"""

import pickle

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

import surpyval as sp
from surpyval.datasets import load_rossi_static

T = np.array([5.0, 20.0, 40.0, 52.0])


@pytest.fixture(scope="module")
def rossi():
    df = load_rossi_static()
    df["censored"] = 1 - df["arrest"]
    return df


def _arrays(df, cols=("fin", "age", "prio")):
    return (
        df["week"].values.astype(float),
        df[list(cols)].values.astype(float),
        df["censored"].values,
    )


PARAMETRIC = [
    sp.WeibullPH,
    sp.LogNormalAFT,
    sp.WeibullPO,
    sp.ExponentialAH,
    sp.WeibullDistributional,
]


@pytest.mark.parametrize("fitter", PARAMETRIC, ids=lambda f: type(f).__name__)
def test_round_trip_and_structure(rossi, fitter):
    x, Z, c = _arrays(rossi, ("fin", "prio"))
    model = fitter.fit(x, Z, c=c)
    assert model.covariance_names == model.parameter_names
    assert model.covariance().shape == (len(model.covariance_names),) * 2
    same = model.with_params(model._estimates())
    assert type(same) is type(model)
    for f in ("sf", "ff", "df", "hf", "Hf"):
        assert_allclose(
            getattr(same, f)(T, Z[:4]), getattr(model, f)(T, Z[:4]), rtol=1e-12
        )
    assert_allclose(same.qf([0.1, 0.3], Z[:2]), model.qf([0.1, 0.3], Z[:2]))
    # A new model: the original is untouched
    assert same is not model and model.data is not None
    assert not hasattr(same, "data")


@pytest.mark.parametrize("fitter", PARAMETRIC, ids=lambda f: type(f).__name__)
def test_wrong_length_names_the_parameters(rossi, fitter):
    x, Z, c = _arrays(rossi, ("fin", "prio"))
    model = fitter.fit(x, Z, c=c)
    with pytest.raises(ValueError, match="covariance_names"):
        model.with_params(model._estimates()[:-1])
    with pytest.raises(ValueError, match=model.covariance_names[-1]):
        model.with_params(np.append(model._estimates(), 0.0))


def test_rebuilt_model_refuses_what_needs_the_fit(rossi):
    x, Z, c = _arrays(rossi)
    model = sp.WeibullPH.fit(x, Z, c=c)
    other = model.with_params(model._estimates() * 1.01)
    with pytest.raises(ValueError, match="from_params models carry"):
        other.covariance()
    with pytest.raises(ValueError, match="from_params models carry"):
        other.cb(T, Z[0])
    with pytest.raises(ValueError, match="with_params"):
        other.plot()
    with pytest.raises(ValueError, match="fit with data"):
        other.aic()
    assert other.maximum == "not applicable"
    assert "given parameters" in repr(other)
    # It saves and loads, without a covariance, and pickles
    loaded = sp.ParametricRegressionModel.from_dict(other.to_dict())
    assert "covariance" not in other.to_dict()
    assert_allclose(loaded.sf(T, Z[0]), other.sf(T, Z[0]), rtol=1e-15)
    unpickled = pickle.loads(pickle.dumps(other))
    assert_allclose(unpickled.sf(T, Z[0]), other.sf(T, Z[0]), rtol=1e-15)


def test_parameters_outside_their_support_are_refused(rossi):
    x, Z, c = _arrays(rossi)
    model = sp.WeibullPH.fit(x, Z, c=c)
    draw = model._estimates()
    draw[0] = -4.6
    with pytest.raises(ValueError, match=r"alpha = -4\.6 .*center=True"):
        model.with_params(draw)
    draw = model._estimates()
    draw[2] = np.nan
    with pytest.raises(ValueError, match="coef_0 = nan"):
        model.with_params(draw)


def test_refit_parameters_predict_as_the_refit(rossi):
    # Two fits of one structure to different data: the second's estimate,
    # given to the first, predicts as the second.
    x, Z, c = _arrays(rossi)
    first = sp.WeibullAFT.fit(x[:300], Z[:300], c=c[:300])
    refit = sp.WeibullAFT.fit(x[100:], Z[100:], c=c[100:])
    moved = first.with_params(refit._estimates())
    for f in ("sf", "hf", "Hf"):
        assert_allclose(
            getattr(moved, f)(T, Z[:4]),
            getattr(refit, f)(T, Z[:4]),
            rtol=1e-12,
        )
    assert_allclose(moved.mean(Z[0]), refit.mean(Z[0]), rtol=1e-9)


def test_formula_and_categories_survive(rossi):
    model = sp.WeibullPH.fit_from_df(
        rossi, x_col="week", c_col="censored", formula="fin + age + C(race)"
    )
    draw = model._estimates() + 0.01
    other = model.with_params(draw)
    assert other.covariance_names == ["alpha", "beta", "fin", "age"] + [
        "C(race)[T.1]"
    ]
    rows = rossi.iloc[:5]
    # The same raw columns, encoded as the fit encoded them
    refit = sp.WeibullPH.fit_from_df(
        rossi, x_col="week", c_col="censored", formula="fin + age + C(race)"
    )
    assert_allclose(
        model.with_params(refit._estimates()).sf(np.full(5, 30.0), rows),
        refit.sf(np.full(5, 30.0), rows),
        rtol=1e-12,
    )
    by_hand = sp.WeibullPH.fit(
        rossi["week"].values,
        rossi[["fin", "age", "race"]].values.astype(float),
        c=rossi["censored"].values,
    ).with_params(draw)
    assert_allclose(
        other.sf(np.full(5, 30.0), rows),
        by_hand.sf(np.full(5, 30.0), rows[["fin", "age", "race"]].values),
        rtol=1e-12,
    )


def test_centre_fixed_and_aliased_are_kept(rossi):
    x, Z, c = _arrays(rossi, ("fin", "age"))
    centred = sp.WeibullPH.fit(x, Z, c=c, center=True)
    plain = sp.WeibullPH.fit(x, Z, c=c)
    rebuilt = centred.with_params(centred._estimates())
    assert_allclose(rebuilt.center, centred.center)
    assert_allclose(rebuilt.sf(T, [1, 30]), plain.sf(T, [1, 30]), rtol=1e-6)
    assert_allclose(rebuilt.sf(T, [1, 30]), centred.sf(T, [1, 30]), rtol=1e-12)

    fixed = sp.WeibullPH.fit(x, Z, c=c, fixed={"beta": 1.2})
    draw = fixed._estimates()
    draw[1] = 9.0  # a fixed parameter's entry is ignored
    assert fixed.with_params(draw).params[1] == 1.2
    assert fixed.with_params(draw).fixed == {"beta": 1.2}

    Z3 = np.column_stack([Z, 2 * Z[:, 1]])  # the third column is aliased
    with pytest.warns(UserWarning):
        aliased = sp.WeibullPH.fit(x, Z3, c=c)
    draw = np.nan_to_num(aliased._estimates(), nan=5.0)
    other = aliased.with_params(draw)
    assert np.isnan(other.params[-1])
    assert_allclose(
        other.sf(T, [1, 30, 60]),
        plain.with_params(draw[:-1]).sf(T, [1, 30]),
        rtol=1e-12,
    )


def test_accelerated_life_and_spline_baselines():
    from surpyval.life_models import Power

    stress = np.repeat([20.0, 30.0, 40.0], 40)
    x = sp.Weibull.random(120, 10, 3, random_state=1) * (100.0 / stress)
    model = sp.AcceleratedLife(sp.Weibull, Power).fit(x, Z=stress)
    other = model.with_params(model._estimates() * [1, 1.02, 1, 1.01])
    assert other.params[0] == model.params[0]  # the life placeholder
    assert other.mean(10.0) != model.mean(10.0)
    assert_allclose(
        model.with_params(model._estimates()).mean(10.0), model.mean(10.0)
    )

    rp = sp.RoystonParmarPH.fit(
        x, (stress == 40.0).astype(float).reshape(-1, 1)
    )
    same = rp.with_params(rp._estimates())
    assert_allclose(same.sf(T, [1.0]), rp.sf(T, [1.0]), rtol=1e-12)


def test_weighted_and_penalised_fits_rebuild_without_their_fit(rossi):
    x, Z, c = _arrays(rossi, ("fin", "age"))
    w = np.where(Z[:, 0] == 1, 2.0, 1.0)
    weighted = sp.WeibullPH.fit(x, Z, c=c, w=w)
    rebuilt = weighted.with_params(weighted._estimates())
    assert rebuilt.weighted is False
    assert_allclose(rebuilt.sf(T, [1, 30]), weighted.sf(T, [1, 30]))

    penalised = sp.WeibullPH.fit_penalised(x, Z, c=c, penalizer=0.05)
    rebuilt = penalised.with_params(penalised._estimates())
    assert rebuilt.penalty is None
    assert_allclose(rebuilt.sf(T, [1, 30]), penalised.sf(T, [1, 30]))


def test_time_varying_fit_rebuilds(rossi):
    x, Z, c = _arrays(rossi, ("fin",))
    model = sp.WeibullPH.fit_tvc(
        i=np.arange(x.size), xl=np.zeros_like(x), xr=x, c=c, Z=Z
    )
    rebuilt = model.with_params(model._estimates())
    assert_allclose(rebuilt.sf(T, [1]), model.sf(T, [1]), rtol=1e-12)


@pytest.mark.parametrize(
    "fitter, dist", [(sp.WeibullPH, "Weibull"), (sp.LogNormalAFT, "LogNormal")]
)
def test_draws_spread_as_the_wald_bound_says(rossi, fitter, dist):
    # The spread of log Hf(t | Z) (Weibull) or of the normal quantile of
    # ff (LogNormal) over models built from covariance draws, against
    # the delta-method standard error behind cb on that scale. 4000
    # draws: the standard deviation is known to about 1.1%.
    from scipy.stats import norm

    # The baseline at the covariate means (center=True): at Z = 0 (age 0)
    # the scale is so poorly determined that normal draws of it fall
    # below 0, which with_params refuses.
    x, Z, c = _arrays(rossi)
    model = fitter.fit(x, Z, c=c, center=True)
    row, t = [1, 25, 2], np.array([20.0, 52.0])
    draws = np.random.default_rng(20261105).multivariate_normal(
        model._estimates(), model.covariance(), size=4000
    )
    z = norm.ppf(0.975)
    if dist == "Weibull":
        values = np.array(
            [np.log(model.with_params(d).Hf(t, row)) for d in draws]
        )
        lo, hi = np.log(model.cb(t, row, on="Hf")).T
    else:
        values = np.array(
            [norm.ppf(model.with_params(d).ff(t, row)) for d in draws]
        )
        lo, hi = norm.ppf(model.cb(t, row, on="ff")).T
    se = (hi - lo) / (2 * z)
    assert_allclose(values.std(axis=0, ddof=1), se, rtol=0.05)


# -- semi-parametric ------------------------------------------------------


def test_additive_hazards_round_trip_and_baseline(rossi):
    x, Z, c = _arrays(rossi)
    model = sp.AdditiveHazards.fit(x, Z, c=c)
    assert model.covariance_names == ["coef_0", "coef_1", "coef_2"]
    same = model.with_params(model._estimates())
    for f in ("sf", "ff", "df", "hf", "Hf"):
        assert_allclose(
            getattr(same, f)(T, Z[:4]), getattr(model, f)(T, Z[:4]), rtol=1e-12
        )
    # At beta = 0 the Lin-Ying baseline is the Nelson-Aalen estimate of
    # the pooled data, whatever the covariates
    zero = model.with_params(np.zeros(3))
    na = sp.NelsonAalen.fit(x, c=c)
    assert_allclose(zero.Hf(T, Z[0]), na.Hf(T), rtol=1e-12)
    assert_allclose(zero.Hf(T, Z[7]), na.Hf(T), rtol=1e-12)
    # The baseline is linear in beta: halfway between two coefficient
    # vectors the cumulative hazard is halfway too (where it is not held)
    b = model._estimates()
    mid = model.with_params(b / 2)

    def H(m):
        return m._baseline_H(T) + T * (np.array(Z[3]) @ m._coef())

    assert_allclose(H(mid), (H(zero) + H(model)) / 2, rtol=1e-12)


def test_additive_hazards_refusals_and_chain(rossi):
    x, Z, c = _arrays(rossi)
    model = sp.AdditiveHazards.fit(x, Z, c=c)
    other = model.with_params(model._estimates() * 1.1)
    with pytest.raises(ValueError, match="with_params"):
        other.covariance()
    with pytest.raises(ValueError, match="with_params"):
        other.standard_errors()
    # It keeps what its baseline needs, so it rebuilds in turn
    again = other.with_params(model._estimates())
    assert_allclose(again.sf(T, Z[0]), model.sf(T, Z[0]), rtol=1e-12)
    restored = sp.AdditiveHazardsModel.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="restored from a dict"):
        restored.with_params(model._estimates())
    with pytest.raises(ValueError, match="covariance_names"):
        model.with_params([0.0, 0.0])
    # Saved without a covariance (nan), and loaded
    loaded = sp.AdditiveHazardsModel.from_dict(other.to_dict())
    assert_allclose(loaded.sf(T, Z[0]), other.sf(T, Z[0]), rtol=1e-15)


@pytest.mark.parametrize("center", [False, True])
def test_proportional_odds_round_trip_and_profile(rossi, center):
    x, Z, c = _arrays(rossi)
    model = sp.ProportionalOdds.fit(x, Z, c=c, center=center)
    assert model.covariance_names == model.parameter_names
    same = model.with_params(model._estimates())
    for f in ("sf", "ff", "df", "hf", "Hf"):
        assert_allclose(
            getattr(same, f)(T, Z[:4]), getattr(model, f)(T, Z[:4]), rtol=1e-12
        )
    # The profile baseline found again at the estimate is the fit's
    assert_allclose(
        model._profile_baseline(model._estimates()), model.g0, rtol=1e-7
    )
    # At other coefficients the baseline maximises the likelihood there:
    # the derivative in the log jumps is zero (checked against the fit's
    # own likelihood, on the centred covariates it fits on).
    from surpyval.univariate.regression._fit_skeleton import covariate_center
    from surpyval.univariate.regression.proportional_odds.proportional_odds import (  # noqa: E501
        _POLikelihood,
    )

    beta = model._estimates() + [0.1, -0.01, 0.02]
    other = model.with_params(beta)
    n = np.ones_like(x)
    mean = covariate_center(Z, n)
    lik = _POLikelihood(x, c, n, np.zeros_like(x), Z - mean)
    at = np.searchsorted(other.x, lik.t)
    shift = 0.0 if center else float(beta @ mean)
    u = np.log(other.g0[at]) - shift
    grad = lik.derivatives(u, -beta)["grad_u"]
    assert np.max(np.abs(grad)) < 1e-8 * lik.d.sum()


def test_proportional_odds_refusals(rossi):
    x, Z, c = _arrays(rossi)
    model = sp.ProportionalOdds.fit(x, Z, c=c)
    other = model.with_params(model._estimates() * 1.1)
    with pytest.raises(ValueError, match="with_params"):
        other.covariance()
    with pytest.raises(ValueError, match="with_params"):
        other.param_cb("coef_0")
    with pytest.raises(ValueError, match="with_params"):
        other.with_params(model._estimates())
    restored = sp.ProportionalOddsModel.from_dict(model.to_dict())
    with pytest.raises(ValueError, match="restored from a"):
        restored.with_params(model._estimates())
    loaded = sp.ProportionalOddsModel.from_dict(other.to_dict())
    assert_allclose(loaded.sf(T, Z[0]), other.sf(T, Z[0]), rtol=1e-15)


@pytest.mark.parametrize("fitter", [sp.ProportionalOdds, sp.AdditiveHazards])
def test_semi_parametric_random_inverts_qf(rossi, fitter):
    x, Z, c = _arrays(rossi)
    model = fitter.fit(x, Z, c=c)
    row = Z[:1]
    draws, rows = model.random(4000, row, random_state=7)
    assert rows.shape == (4000, 3)
    U = np.random.default_rng(7).uniform(0, 1, 4000)
    expected = model.qf(1 - U, row)
    assert_allclose(draws, np.where(np.isnan(expected), np.inf, expected))
    # Beyond the data as often as the curve says: within 3 binomial SEs
    p = float(model.sf(x.max(), row[0]))
    assert abs(np.isinf(draws).mean() - p) < 3 * np.sqrt(p * (1 - p) / 4000)


def test_dataframe_rows_on_a_formula_semi_parametric_model(rossi):
    model = sp.ProportionalOdds.fit_from_df(
        rossi, x_col="week", c_col="censored", formula="fin + prio"
    )
    other = model.with_params(model._estimates() + 0.01)
    rows = rossi.iloc[:3]
    assert other.sf(np.full(3, 30.0), rows).shape == (3,)
    assert isinstance(rows, pd.DataFrame)
