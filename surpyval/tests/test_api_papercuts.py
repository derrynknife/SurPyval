"""The API papercuts of #485, one test (or a few) per item."""

import inspect
import json
import warnings

import matplotlib
import numpy as np
import pytest

import surpyval
import surpyval as sp

matplotlib.use("Agg")

X = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


# -- to_json() without a path returns the JSON, and from_json reads it ----


def test_to_json_without_a_path_returns_a_string(tmp_path):
    model = sp.Weibull.fit(X)
    text = model.to_json()
    assert isinstance(text, str)
    assert json.loads(text) == model.to_dict()
    np.testing.assert_allclose(sp.from_json(text).params, model.params)
    np.testing.assert_allclose(
        type(model).from_json(text).params, model.params
    )
    # the file form is unchanged
    path = tmp_path / "w.json"
    assert model.to_json(path) is None
    np.testing.assert_allclose(sp.from_json(path).params, model.params)
    np.testing.assert_allclose(sp.from_json(str(path)).params, model.params)


def test_to_json_string_for_other_models():
    km = sp.KaplanMeier.fit(X)
    assert sp.from_json(km.to_json()).sf(5) == km.sf(5)
    assert sp.from_json(sp.InstantlyOccurs.to_json()) is sp.InstantlyOccurs


# -- a model built from parameters can be plotted --------------------------


@pytest.mark.parametrize(
    "dist, params",
    [(sp.Weibull, [100, 2]), (sp.Normal, [10, 2]), (sp.Uniform, [2, 5])],
)
def test_from_params_model_plots_its_curve(dist, params):
    # It raised "Can't plot model that was given parameters and no data"
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    model = dist.from_params(params)
    model.plot(ax=ax)
    assert len(ax.collections[0].get_offsets()) == 0  # no data points
    x, y = ax.lines[-1].get_data()
    np.testing.assert_allclose(y, model.ff(x))
    lo, hi = ax.get_xlim()
    assert lo <= model.qf(0.01) and model.qf(0.99) <= hi
    plt.close("all")


def test_restored_model_without_data_plots_its_curve():
    import matplotlib.pyplot as plt

    _, ax = plt.subplots()
    restored = sp.from_dict(sp.Weibull.fit(X).to_dict())
    assert restored.plot(ax=ax) is ax
    plt.close("all")


# -- `how` in any case ------------------------------------------------------


@pytest.mark.parametrize("how", ["mle", "Mle", "mps", "mpp", "mse"])
def test_how_is_case_insensitive(how):
    lower = sp.Weibull.fit(X, how=how)
    upper = sp.Weibull.fit(X, how=how.upper())
    np.testing.assert_allclose(lower.params, upper.params)
    assert lower.method == upper.method


def test_unknown_how_still_raises_listing_the_methods():
    with pytest.raises(ValueError, match="MLE"):
        sp.Weibull.fit(X, how="mlee")


# -- Kaplan-Meier with interval data names Turnbull -------------------------


@pytest.mark.parametrize(
    "fitter", [sp.KaplanMeier, sp.NelsonAalen, sp.FlemingHarrington]
)
def test_step_estimators_point_to_turnbull(fitter):
    with pytest.raises(ValueError, match="interval.*use Turnbull"):
        fitter.fit(xl=[1, 2, 3], xr=[2, 3, 4])
    with pytest.raises(ValueError, match="left.*use Turnbull"):
        fitter.fit([1, 2, 3], c=[0, -1, 0])


# -- an (n, 1) column is one observation per row ---------------------------


def test_column_vectors_are_accepted():
    x = np.array(X).reshape(-1, 1)
    c = np.array([0, 0, 1, 0, 0, 1]).reshape(-1, 1)
    n = np.array([1, 2, 1, 1, 1, 1]).reshape(-1, 1)
    model = sp.Weibull.fit(x, c, n)
    flat = sp.Weibull.fit(x.ravel(), c.ravel(), n.ravel())
    np.testing.assert_allclose(model.params, flat.params)
    assert sp.KaplanMeier.fit(x).sf(5) == sp.KaplanMeier.fit(X).sf(5)
    # a genuine (n, 2) interval array is still intervals
    assert sp.Weibull.fit(xl=[1, 2, 3], xr=[2, 3, 5]).params.size == 2


# -- qf outside [0, 1] -------------------------------------------------------


def test_qf_outside_unit_interval_is_nan():
    # Weibull qf(1.5) was inf, and a Normal's qf(-0.1) was 0; NaN now,
    # with a warning (#576)
    model = sp.Weibull.from_params([1, 2])
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        assert np.isnan(model.qf(1.5)) and np.isnan(model.qf(-0.1))
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        q = sp.Normal.from_params([0, 1]).qf([-0.1, 0, 0.5, 1, 1.1])
    np.testing.assert_array_equal(q, [np.nan, -np.inf, 0, np.inf, np.nan])
    # a bounded support ends where it ends
    np.testing.assert_array_equal(
        sp.Uniform.from_params([2, 5]).qf([0, 1]), [2, 5]
    )


def test_qf_keeps_zero_inflation_and_cure_fraction():
    zi = sp.Weibull.fit([0, 0, 1, 2, 3, 4], zi=True)
    assert zi.qf(zi.f0 / 2) == 0
    lfp = sp.Weibull.fit(
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0] * 5 + [1] * 5, lfp=True
    )
    assert lfp.qf((1 + lfp.lfp_p) / 2) == np.inf


# -- logrank with a continuous "group" ---------------------------------------


def test_logrank_warns_when_the_groups_are_singletons():
    x = np.random.default_rng(0).weibull(2, 50) * 10
    with pytest.warns(UserWarning, match="50 of the 50 groups") as record:
        sp.logrank(x, x * 1.3)
    assert [w.filename for w in record] == [__file__]


def test_logrank_ordinary_groups_do_not_warn():
    x = [9, 13, 13, 18, 23, 28, 31, 34, 45, 48, 161]
    x += [5, 5, 8, 8, 12, 16, 23, 27, 30, 33, 43, 45]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.logrank(x, [1] * 11 + [2] * 12)


# -- fit_best ----------------------------------------------------------------


def test_fit_best_passes_over_out_of_support_candidates_quietly():
    # It warned "Beta distribution failed to fit" and a support warning
    np.random.seed(1)
    x = sp.Weibull.random(50, 10, 2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        model = sp.fit_best(x)
    assert model is not None


def test_fit_best_metric_message_is_spelled_right():
    with pytest.raises(ValueError, match="must be one of"):
        sp.fit_best(X, metric="nope")


# -- models that live in a subpackage ----------------------------------------


@pytest.mark.parametrize(
    "name, where",
    [
        ("laplace", "surpyval.recurrent"),
        ("TrendTestResult", "surpyval.recurrent"),
        ("Gaussian", "surpyval.multivariate"),
    ],
)
def test_top_level_names_a_helper_s_subpackage(name, where):
    with pytest.raises(AttributeError, match=f"it is in {where}"):
        getattr(sp, name)
    assert not hasattr(sp, name)


@pytest.mark.parametrize(
    "name, value, instead",
    [
        ("NUM", np.float64, "numpy.float64"),
        ("TINIEST", np.finfo(float).tiny, "numpy.finfo(float).tiny"),
        ("EPS", np.sqrt(np.finfo(float).eps), "numpy.sqrt("),
    ],
)
def test_613_top_level_constants_are_deprecated(name, value, instead):
    # They still work until v0.24, with a warning naming what to use, and
    # are no longer listed; ``surpyval.np`` stays (custom distributions).
    with pytest.warns(DeprecationWarning, match="v0.24") as caught:
        assert getattr(sp, name) == value
    assert instead in str(caught[0].message)
    assert caught[0].filename == __file__
    assert name not in dir(sp)
    assert "np" in dir(sp)


@pytest.mark.parametrize(
    "name, where",
    [
        (name, "surpyval.recurrent")
        for name in [
            "ARA",
            "ARI",
            "CauseSpecificMCF",
            "CauseSpecificNHPP",
            "CoxLewis",
            "CrowAMSAA",
            "Duane",
            "GeneralizedOneRenewal",
            "GeneralizedRenewal",
            "HPP",
            "NonParametricCounting",
            "ProportionalIntensityHPP",
            "ProportionalIntensityNHPP",
        ]
    ]
    + [
        (name, "surpyval.univariate.competing_risks")
        for name in [
            "CompetingRisks",
            "CompetingRisksProportionalHazards",
            "FineGray",
            "ParametricCompetingRisks",
        ]
    ]
    + [
        (name, "surpyval.degradation")
        for name in [
            "DegradationAnalysis",
            "DestructiveDegradation",
            "GammaProcess",
            "WienerProcess",
        ]
    ],
)
def test_models_are_importable_from_the_top_level(name, where):
    # The model classes are at the top level, as the regression models
    # are; the same objects as in their packages.
    import importlib

    assert getattr(sp, name) is getattr(importlib.import_module(where), name)


# -- readable signatures -----------------------------------------------------


def test_signatures_show_arraylike_not_its_expansion():
    # 3218 characters, the expanded numpy ArrayLike union
    sig = str(inspect.signature(sp.Weibull.fit))
    assert len(sig) < 700
    assert "ArrayLike" in sig and "_SupportsArray" not in sig
    for fn in (sp.KaplanMeier.fit, sp.CoxPH.fit, sp.logrank, sp.fit_best):
        assert "_SupportsArray" not in str(inspect.signature(fn))


# -- MPP heuristic -----------------------------------------------------------


def test_unknown_heuristic_lists_the_heuristics():
    with pytest.raises(ValueError, match="Nelson-Aalen.*Turnbull|Median"):
        sp.Weibull.fit(X, how="MPP", heuristic="median")


# -- the trend tests take the fitters' c -------------------------------------


def _mettas():
    from surpyval.datasets import load_mettas_and_zhao

    d = load_mettas_and_zhao()
    return d["x"].to_numpy(), d["i"].to_numpy(), d["c"].to_numpy()


def test_trend_tests_explain_a_c_passed_as_t():
    from surpyval.recurrent import laplace

    x, i, c = _mettas()
    # It said "array `T` must have one entry per system (6 systems, 33
    # entries)"
    with pytest.raises(ValueError, match="looks like the censoring flags"):
        laplace(x, i, c)


@pytest.mark.parametrize("test", ["laplace", "mil_hdbk_189c"])
def test_trend_tests_take_c_by_keyword(test):
    import surpyval.recurrent as rec

    x, i, c = _mettas()
    direct = getattr(rec, test)(x, i, c=c)
    via_model = rec.CrowAMSAA.fit(x, i, c).trend_test(test=test)
    assert direct.statistic == pytest.approx(via_model.statistic)
    assert direct.n_events == via_model.n_events == 27
    with pytest.raises(ValueError, match="either `T` or `c`"):
        getattr(rec, test)(x, i, T=9000.0, c=c)


# -- CoxPH.fit_tvc_from_df takes a formula ---------------------------------


def _rossi_tv():
    from surpyval.datasets import load_rossi_time_varying

    df = load_rossi_time_varying()
    df["c"] = 1 - df["event"]
    return df


def test_fit_tvc_from_df_takes_a_formula():
    df = _rossi_tv()
    model = sp.CoxPH.fit_tvc_from_df(
        df, "id", "start", "stop", "c", formula="fin + age + prio + employed"
    )
    assert model.feature_names == [
        "fin[T.yes]",
        "age",
        "prio",
        "employed[T.yes]",
    ]
    coded = df.assign(
        fin=(df["fin"] == "yes").astype(float),
        employed=(df["employed"] == "yes").astype(float),
    )
    by_cols = sp.CoxPH.fit_tvc_from_df(
        coded, "id", "start", "stop", "c", ["fin", "age", "prio", "employed"]
    )
    np.testing.assert_allclose(model.beta, by_cols.beta, rtol=1e-8)


def test_fit_tvc_from_df_matches_lifelines_on_rossi():
    # lifelines' CoxTimeVaryingFitter on the same data (its documentation's
    # example): race and mar are coded against the other level there.
    model = sp.CoxPH.fit_tvc_from_df(
        _rossi_tv(),
        "id",
        "start",
        "stop",
        "c",
        formula="fin + age + race + wexp + mar + paro + prio + employed",
    )
    ll = [-0.3567, -0.0463, 0.3387, -0.0256, -0.2937, -0.0642, 0.0851]
    ll += [-1.3283]
    signs = np.array([1, 1, -1, 1, -1, 1, 1, 1])
    np.testing.assert_allclose(model.beta * signs, ll, atol=1e-4)


def test_string_covariate_column_suggests_formula():
    # It was numpy's bare "could not convert string to float: 'no'"
    df = _rossi_tv()
    with pytest.raises(ValueError, match="not numeric.*formula="):
        sp.CoxPH.fit_tvc_from_df(df, "id", "start", "stop", "c", ["fin"])
    with pytest.raises(ValueError, match=r"\['fin'\] are not numeric"):
        sp.CoxPH.fit_from_df(df, "week", ["fin", "age"], "c")
    with pytest.raises(ValueError, match="not numeric"):
        sp.WeibullPH.fit_from_df(df, "week", ["fin"], "c")


# -- CompetingRisks.plot -----------------------------------------------------


def test_competing_risks_plots_its_cumulative_incidences():
    import matplotlib.pyplot as plt

    from surpyval.univariate.competing_risks import CompetingRisks

    x = [1, 2, 3, 4, 5, 6, 7, 8]
    e = ["a", "b", "a", "b", "a", None, "a", "b"]
    model = CompetingRisks.fit(x, e, c=[0, 0, 0, 0, 0, 1, 0, 0])
    _, ax = plt.subplots()
    model.plot(ax=ax)
    # the top of the stack is the sum of the causes' incidences
    top = ax.collections[-1].get_paths()[0].vertices[:, 1].max()
    assert top == pytest.approx(model.cif(8, "a") + model.cif(8, "b"))
    _, ax = plt.subplots()
    model.plot(stacked=False, ax=ax)
    for line, cause in zip(ax.lines, ["a", "b"]):
        xs, ys = line.get_data()
        np.testing.assert_allclose(ys, model.cif(xs, cause))
    plt.close("all")


# -- cs(x, given) ------------------------------------------------------------


def test_cs_takes_given_and_the_old_name_warns():
    # The time already survived is ``given``, as in the regression models'
    # sf_tvc(..., given=); ``X`` works until v0.23 with a warning.
    import warnings

    model = sp.Weibull.from_params([10, 3])
    expected = model.sf(21) / model.sf(10)
    np.testing.assert_allclose(model.cs(11, given=10), expected)
    np.testing.assert_allclose(
        sp.Weibull.cs(11, 10, 10, 3), model.cs(11, given=10)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        old = model.cs(11, X=10)
    np.testing.assert_allclose(old, expected)
    assert len(caught) == 1
    assert issubclass(caught[0].category, DeprecationWarning)
    assert caught[0].filename == __file__
    assert "use 'given'" in str(caught[0].message)


# ---------------------------------------------------------------------------
# The fixes of #276-#282 keep the public names.
# ---------------------------------------------------------------------------


def test_surpyval_namespace_unchanged():
    # Guard: the fixes must not have removed public names.
    for name in ("AdditiveHazards", "CoxPH", "Rayleigh", "Uniform"):
        assert hasattr(surpyval, name)


def test_576_qf_warns_of_a_probability_outside_the_unit_interval():
    # qf(10) for the B10 life gave NaN in silence
    model = sp.Weibull.from_params([10, 3])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        q = model.qf([0.1, 10.0, np.nan])
    assert np.isnan(q[1:]).all() and q[0] == model.qf(0.1)
    (warning,) = caught
    assert warning.filename == __file__
    text = str(warning.message)
    assert "1 of the 3 probabilities given is outside [0, 1] (10.0)" in text
    assert "B10 life pass 0.1" in text
    # NaN is a missing probability, not a mistake: no warning
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(model.qf(np.nan))
        model.qf([0.0, 1.0])
    # The Royston-Parmar model, whose root finder raised a scipy error
    rp = sp.RoystonParmar.fit(sp.Weibull.random(40, 10, 2, random_state=1))
    with pytest.warns(UserWarning, match=r"outside \[0, 1\]"):
        out = rp.qf(np.array([0.5, 1.5]))
    assert np.isfinite(out[0]) and np.isnan(out[1])
