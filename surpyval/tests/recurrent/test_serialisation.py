"""Serialisation of the fitted recurrent-event models.

The recurrence result classes -- the parametric intensity fit
(``ParametricRecurrenceModel``), the nonparametric MCF
(``NonParametricCounting``), the proportional-intensity regression
(``ProportionalIntensityModel``), and the competing-risks containers
(``CauseSpecificMCF``, ``CauseSpecificNHPP``) -- round-trip through
``to_dict``/``from_dict`` (and the JSON file variants). The intensity model is
stateless, so each stores its name plus the fitted parameters (or, for the
MCF, the step arrays), and the reloaded model reproduces every prediction
exactly.
"""

import warnings

import numpy as np
import pytest

from surpyval.recurrent import (
    ARA,
    ARI,
    HPP,
    CoxLewis,
    CrowAMSAA,
    Duane,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
    NonParametricCounting,
    ProportionalIntensityHPP,
    ProportionalIntensityNHPP,
)
from surpyval.recurrent.competing_risks import (
    CauseSpecificMCF,
    CauseSpecificNHPP,
)
from surpyval.recurrent.parametric.parametric_recurrence import (
    ParametricRecurrenceModel,
)
from surpyval.tests._helpers import (
    REPAIR_FLEET_C,
    REPAIR_FLEET_I,
    REPAIR_FLEET_X,
    json_round_trip,
)
from surpyval.univariate.regression.tvc_schedule import StepSchedule


def _pi_data(seed=0, n_items=25):
    rng = np.random.default_rng(seed)
    xs, ii, cc, ZZ = [], [], [], []
    for item in range(n_items):
        z = rng.normal(0, 1)
        for _ in range(int(rng.integers(2, 6))):
            xs.append(float(rng.uniform(0, 500)))
            ii.append(item)
            cc.append(0)
            ZZ.append([z])
        xs.append(500.0)
        ii.append(item)
        cc.append(1)
        ZZ.append([z])
    return np.array(xs), np.array(ii), np.array(cc), np.array(ZZ)


# -- ParametricRecurrenceModel --------------------------------------------


@pytest.mark.parametrize("fitter", [CrowAMSAA, Duane])
def test_parametric_recurrence_round_trip(fitter):
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1000, 40))
    model = fitter.fit(x)
    restored = ParametricRecurrenceModel.from_dict(
        json_round_trip(model.to_dict())
    )
    xt = np.array([100.0, 500.0, 900.0])
    assert np.allclose(model.cif(xt), restored.cif(xt))
    assert np.allclose(model.iif(xt), restored.iif(xt))
    assert np.allclose(model.params, restored.params)
    assert model.dist.name == restored.dist.name


def test_parametric_recurrence_json_file(tmp_path):
    rng = np.random.default_rng(2)
    x = np.sort(rng.uniform(0, 1000, 30))
    model = CrowAMSAA.fit(x)
    fp = tmp_path / "nhpp.json"
    model.to_json(fp)
    restored = ParametricRecurrenceModel.from_json(fp)
    xt = np.array([200.0, 800.0])
    assert np.allclose(model.cif(xt), restored.cif(xt))


def test_parametric_recurrence_rejects_wrong_model():
    with pytest.raises(ValueError, match="ParametricRecurrenceModel"):
        ParametricRecurrenceModel.from_dict({"model": "Other"})


def test_parametric_recurrence_rejects_unknown_dist():
    d = {"model": "ParametricRecurrenceModel", "dist": "Nope", "params": [1.0]}
    with pytest.raises(ValueError, match="Unknown recurrence intensity"):
        ParametricRecurrenceModel.from_dict(d)


# -- NonParametricCounting (MCF) ------------------------------------------


def test_mcf_round_trip():
    x = np.array([5, 10, 15, 4, 9, 12, 20], dtype=float)
    i = np.array([1, 1, 1, 2, 2, 3, 3])
    c = np.array([0, 0, 1, 0, 1, 0, 1])
    model = NonParametricCounting.fit(x, i, c)
    restored = NonParametricCounting.from_dict(
        json_round_trip(model.to_dict())
    )
    grid = np.array([6.0, 11.0, 16.0])
    assert np.allclose(model.mcf(grid), restored.mcf(grid), equal_nan=True)
    assert np.allclose(model.x, restored.x)
    assert np.allclose(model.var, restored.var)


def test_mcf_json_file(tmp_path):
    x = np.array([5, 10, 15, 4, 9], dtype=float)
    i = np.array([1, 1, 1, 2, 2])
    c = np.array([0, 0, 1, 0, 1])
    model = NonParametricCounting.fit(x, i, c)
    fp = tmp_path / "mcf.json"
    model.to_json(fp)
    restored = NonParametricCounting.from_json(fp)
    grid = np.array([6.0, 11.0])
    assert np.allclose(model.mcf(grid), restored.mcf(grid), equal_nan=True)


# -- ProportionalIntensityModel -------------------------------------------


def test_proportional_intensity_nhpp_round_trip():
    x, i, c, Z = _pi_data(seed=3)
    model = ProportionalIntensityNHPP.fit(x, Z, i=i, c=c, baseline=Duane)
    restored = type(model).from_dict(json_round_trip(model.to_dict()))
    xt = np.array([100.0, 300.0, 450.0])
    Zq = np.array([0.5])
    assert np.allclose(model.cif(xt, Zq), restored.cif(xt, Zq))
    assert np.allclose(model.iif(xt, Zq), restored.iif(xt, Zq))
    assert np.allclose(model.params, restored.params)
    assert np.allclose(model.coeffs, restored.coeffs)


def test_proportional_intensity_hpp_round_trip():
    x, i, c, Z = _pi_data(seed=4)
    model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    assert model.kind == "HPP"
    restored = type(model).from_dict(json_round_trip(model.to_dict()))
    xt = np.array([100.0, 300.0])
    Zq = np.array([0.5])
    assert np.allclose(model.cif(xt, Zq), restored.cif(xt, Zq))
    assert np.allclose(model.iif(xt, Zq), restored.iif(xt, Zq))


def test_proportional_intensity_rejects_wrong_model():
    with pytest.raises(ValueError, match="ProportionalIntensityModel"):
        from surpyval.recurrent.regression.proportional_intensity import (
            ProportionalIntensityModel,
        )

        ProportionalIntensityModel.from_dict({"model": "Other"})


# -- Cause-specific containers --------------------------------------------


def test_cause_specific_mcf_round_trip():
    x = np.array([5, 10, 15, 4, 9, 12, 20], dtype=float)
    i = np.array([1, 1, 1, 2, 2, 3, 3])
    c = np.array([0, 0, 1, 0, 1, 0, 1])
    e = np.array(["A", "B", "A", "A", "B", "B", "A"])
    model = CauseSpecificMCF.fit(x, i, c, e=e)
    restored = CauseSpecificMCF.from_dict(json_round_trip(model.to_dict()))
    assert restored.event_types == model.event_types
    grid = np.array([6.0, 11.0, 16.0])
    for cause in model.event_types:
        assert np.allclose(
            model.mcf(grid, cause),
            restored.mcf(grid, cause),
            equal_nan=True,
        )


def test_cause_specific_nhpp_round_trip():
    rng = np.random.default_rng(5)
    xs, ii, cc, es = [], [], [], []
    for item in range(20):
        for _ in range(int(rng.integers(2, 5))):
            xs.append(float(rng.uniform(0, 400)))
            ii.append(item)
            cc.append(0)
            es.append(rng.choice(["A", "B"]))
        xs.append(400.0)
        ii.append(item)
        cc.append(1)
        es.append(None)
    model = CauseSpecificNHPP.fit(
        np.array(xs), np.array(ii), np.array(cc), e=np.array(es, dtype=object)
    )
    restored = CauseSpecificNHPP.from_dict(json_round_trip(model.to_dict()))
    assert restored.dist.name == model.dist.name
    assert restored.event_types == model.event_types
    xt = np.array([50.0, 200.0, 350.0])
    for cause in model.event_types:
        assert np.allclose(model.cif(xt, cause), restored.cif(xt, cause))


def _marked_fit(**kwargs):
    rng = np.random.default_rng(7)
    xs, ii, cc, es = [], [], [], []
    for item in range(12):
        for _ in range(int(rng.integers(3, 7))):
            xs.append(float(rng.uniform(0, 400)))
            ii.append(item)
            cc.append(0)
            es.append(rng.choice(["A", "B"]))
        xs.append(400.0)
        ii.append(item)
        cc.append(1)
        es.append(None)
    return CauseSpecificNHPP.fit(
        np.array(xs),
        np.array(ii),
        np.array(cc),
        e=np.array(es, dtype=object),
        **kwargs,
    )


def test_cause_specific_nhpp_round_trip_keeps_bounds_growth_and_forecasts():
    model = _marked_fit()
    saved = json_round_trip(model.to_dict())
    restored = CauseSpecificNHPP.from_dict(saved)
    xt = np.array([50.0, 200.0, 400.0, 500.0])
    for bound in ("two-sided", "lower", "upper"):
        kw = {"alpha_ci": 0.1, "bound": bound}
        for cause in model.event_types:
            for name in ("cif_cb", "iif_cb", "mtbf_cb"):
                assert np.array_equal(
                    getattr(restored, name)(xt, cause, **kw),
                    getattr(model, name)(xt, cause, **kw),
                ), (name, cause, bound)
        for name in ("total_cif_cb", "total_iif_cb", "total_mtbf_cb"):
            assert np.array_equal(
                getattr(restored, name)(xt, **kw),
                getattr(model, name)(xt, **kw),
            ), (name, bound)
    for name in ("total_cif", "total_iif", "total_mtbf"):
        assert np.array_equal(
            getattr(restored, name)(xt), getattr(model, name)(xt)
        )
    for cause in model.event_types:
        assert np.array_equal(restored.mtbf(xt, cause), model.mtbf(xt, cause))
    # the end of observation is kept, so growth() reads the same time
    assert saved["end"] == 400.0
    assert restored.growth().equals(model.growth())
    assert restored.growth(x=100.0, alpha_ci=0.2).equals(
        model.growth(x=100.0, alpha_ci=0.2)
    )
    assert restored.summary().equals(model.summary())
    # forecasts for given ages; the fitted items' ages are data
    kw = {"age": [400.0] * 12, "horizon": [50.0, 100.0], "alpha_ci": 0.1}
    for event in (None, *model.event_types):
        a = restored.forecast(event=event, **kw)
        b = model.forecast(horizon=kw["horizon"], alpha_ci=0.1, event=event)
        assert np.array_equal(a.expected, b.expected)
        assert np.array_equal(a.lower, b.lower)
        assert np.array_equal(a.upper, b.upper)
        assert np.array_equal(a.period_expected, b.period_expected)
    # a second round trip writes the same document
    assert json_round_trip(restored.to_dict()) == saved
    assert saved["schema"] == 1


def test_cause_specific_nhpp_restored_refuses_what_needs_the_data():
    model = _marked_fit()
    restored = CauseSpecificNHPP.from_dict(json_round_trip(model.to_dict()))
    with pytest.raises(ValueError, match="age is required"):
        restored.forecast(horizon=10.0)
    with pytest.raises(ValueError, match="age is required"):
        restored.forecast(horizon=10.0, event="A")
    with pytest.raises(ValueError, match="requires a model fitted from data"):
        restored.mtbf_cb(400.0, "A", method="crow")
    with pytest.raises(ValueError, match="requires a model fitted from data"):
        restored.iif_cb(400.0, "A", method="crow")
    with pytest.raises(ValueError, match="restored with from_dict"):
        restored.aic()


def test_cause_specific_nhpp_dict_without_covariances_refuses_bounds():
    # A dict saved before the covariances were kept, and a least-squares
    # fit's: the point values restore, the bounds say why they cannot.
    model = _marked_fit()
    old = json_round_trip(model.to_dict())
    del old["covariances"], old["end"]
    restored = CauseSpecificNHPP.from_dict(old)
    assert np.array_equal(restored.total_iif(100.0), model.total_iif(100.0))
    for call in (
        lambda: restored.total_cif_cb(100.0),
        lambda: restored.cif_cb(100.0, "A"),
        lambda: restored.mtbf_cb(100.0, "A"),
        lambda: restored.summary(),
        lambda: restored.growth(x=100.0),
    ):
        with pytest.raises(ValueError, match="restored with from_dict"):
            call()
    with pytest.raises(ValueError, match="x is required"):
        restored.growth()

    mse = _marked_fit(how="MSE")
    saved = json_round_trip(mse.to_dict())
    assert saved["covariances"] == [None, None]
    restored = CauseSpecificNHPP.from_dict(saved)
    with pytest.raises(ValueError, match="refit with how='MLE'"):
        restored.total_cif_cb(100.0)
    assert restored.forecast(age=0.0, horizon=10.0).expected == pytest.approx(
        mse.forecast(age=0.0, horizon=10.0).expected
    )


def test_cause_specific_nhpp_json_file(tmp_path):
    rng = np.random.default_rng(6)
    xs, ii, cc, es = [], [], [], []
    for item in range(15):
        for _ in range(int(rng.integers(2, 5))):
            xs.append(float(rng.uniform(0, 400)))
            ii.append(item)
            cc.append(0)
            es.append(rng.choice(["A", "B"]))
        xs.append(400.0)
        ii.append(item)
        cc.append(1)
        es.append(None)
    model = CauseSpecificNHPP.fit(
        np.array(xs), np.array(ii), np.array(cc), e=np.array(es, dtype=object)
    )
    fp = tmp_path / "csnhpp.json"
    model.to_json(fp)
    restored = CauseSpecificNHPP.from_json(fp)
    xt = np.array([100.0, 300.0])
    for cause in model.event_types:
        assert np.allclose(model.cif(xt, cause), restored.cif(xt, cause))


# ---------------------------------------------------------------------------
# A simulated MCF has no variance, before and after a round
# trip; an old dict's NaN variance reads as none.
# ---------------------------------------------------------------------------


def test_simulated_mcf_round_trip_has_no_variance():
    sim = CrowAMSAA.from_params([10, 2]).time_terminated_simulation(
        20, items=50, random_state=1
    )
    as_dict = sim.to_dict()
    assert as_dict["var"] is None
    restored = NonParametricCounting.from_dict(as_dict)
    with pytest.raises(ValueError, match="no variance"):
        restored.mcf_cb([5])


def test_old_dict_with_nan_variance_reads_as_no_variance():
    old = NonParametricCounting.fit([1, 2, 3]).to_dict()
    old["var"] = float("nan")
    old.pop("origin")
    restored = NonParametricCounting.from_dict(old)
    assert restored.var is None
    assert restored.origin == 0.0


# -- a restored single-process model keeps its Wald bounds -----------------
#
# The cause-specific models save each cause's parameter covariance; the
# single-process models they are made of do the same, so every bound a
# fit has from its covariance, its restored copy has too.


def _restored(model):
    return type(model).from_dict(json_round_trip(model.to_dict()))


@pytest.mark.parametrize(
    "fitter", [HPP, CrowAMSAA, Duane, CoxLewis], ids=lambda f: f.name
)
def test_restored_recurrence_model_gives_the_fits_bounds(fitter):
    rng = np.random.default_rng(11)
    x = np.sort(rng.uniform(0, 1000, 40))
    model = fitter.fit(x)
    restored = _restored(model)
    xt = np.array([100.0, 500.0, 900.0])
    np.testing.assert_array_equal(restored.covariance(), model.covariance())
    np.testing.assert_array_equal(
        restored.standard_errors(), model.standard_errors()
    )
    for bound in ("two-sided", "lower", "upper"):
        for name in ("cif_cb", "iif_cb", "mtbf_cb"):
            np.testing.assert_allclose(
                getattr(restored, name)(xt, bound=bound),
                getattr(model, name)(xt, bound=bound),
                rtol=1e-12,
            )
    for name in model.parameter_names:
        np.testing.assert_allclose(
            restored.param_cb(name), model.param_cb(name), rtol=1e-12
        )
    assert restored.summary().equals(model.summary())
    # Saved again, it is the same dictionary.
    assert restored.to_dict() == json_round_trip(model.to_dict())


def test_restored_recurrence_model_refuses_what_needs_the_data():
    x = np.sort(np.random.default_rng(12).uniform(0, 1000, 40))
    restored = _restored(CrowAMSAA.fit(x))
    for call in (
        lambda: restored.log_likelihood,
        restored.aic,
        restored.bic,
        restored.aic_c,
    ):
        with pytest.raises(ValueError, match="keeps the parameter covariance"):
            call()
    with pytest.raises(ValueError, match="requires a model fitted from data"):
        restored.mtbf_cb([500.0], method="crow")
    with pytest.raises(ValueError, match="requires a model fitted from data"):
        restored.residuals()


def test_a_dict_saved_without_a_covariance_still_loads_and_refuses_bounds():
    x = np.sort(np.random.default_rng(13).uniform(0, 1000, 40))
    model = Duane.fit(x)
    saved = json_round_trip(model.to_dict())
    del saved["covariance"]
    old = ParametricRecurrenceModel.from_dict(saved)
    np.testing.assert_array_equal(old.params, model.params)
    for call in (
        old.covariance,
        old.standard_errors,
        old.summary,
        lambda: old.cif_cb([500.0]),
        lambda: old.param_cb("alpha"),
    ):
        with pytest.raises(ValueError, match="holds no parameter covariance"):
            call()


def test_a_model_with_no_likelihood_saves_no_covariance():
    x = np.sort(np.random.default_rng(14).uniform(0, 1000, 40))
    built = CrowAMSAA.from_params(CrowAMSAA.fit(x).params)
    assert built.to_dict()["covariance"] is None
    restored = _restored(built)
    with pytest.raises(ValueError, match="no likelihood"):
        restored.cif_cb([500.0])


def test_a_saved_covariance_of_the_wrong_size_is_refused():
    x = np.sort(np.random.default_rng(15).uniform(0, 1000, 40))
    saved = json_round_trip(Duane.fit(x).to_dict())
    saved["covariance"] = [[1.0]]
    with pytest.raises(ValueError, match="one row and column per parameter"):
        ParametricRecurrenceModel.from_dict(saved)


@pytest.mark.parametrize("baseline", [None, Duane, CrowAMSAA, CoxLewis])
def test_restored_proportional_intensity_model_gives_the_fits_bounds(baseline):
    x, i, c, Z = _pi_data(seed=5)
    if baseline is None:
        model = ProportionalIntensityHPP.fit(x, Z, i=i, c=c)
    else:
        model = ProportionalIntensityNHPP.fit(
            x, Z, i=i, c=c, baseline=baseline
        )
    restored = _restored(model)
    xt = np.array([100.0, 300.0, 450.0])
    Zq = np.array([0.5])
    np.testing.assert_array_equal(restored.covariance(), model.covariance())
    for bound in ("two-sided", "lower", "upper"):
        for name in ("cif_cb", "iif_cb"):
            np.testing.assert_allclose(
                getattr(restored, name)(xt, Zq, bound=bound),
                getattr(model, name)(xt, Zq, bound=bound),
                rtol=1e-12,
            )
    for name in model.parameter_names:
        np.testing.assert_allclose(
            restored.param_cb(name), model.param_cb(name), rtol=1e-12
        )
    assert restored.summary().equals(model.summary())
    plan = StepSchedule.from_changepoints([0, 200], [[0.0], [1.0]])
    np.testing.assert_allclose(
        restored.cb_tvc(xt, plan), model.cb_tvc(xt, plan), rtol=1e-12
    )
    with pytest.raises(ValueError, match="keeps the parameter covariance"):
        restored.aic()


def test_restored_single_process_bounds_are_the_cause_specific_ones():
    # One cause: the two saved models describe the same fit, and now
    # answer alike after restoring.
    x = np.sort(np.random.default_rng(16).uniform(0, 1000, 40))
    e = np.array(["a"] * 40)
    single = _restored(CrowAMSAA.fit(x))
    marked = CauseSpecificNHPP.from_dict(
        json_round_trip(CauseSpecificNHPP.fit(x, e=e).to_dict())
    )
    xt = np.array([100.0, 500.0, 900.0])
    np.testing.assert_allclose(
        single.cif_cb(xt), marked.cif_cb(xt, "a"), rtol=1e-9
    )
    np.testing.assert_allclose(
        single.iif_cb(xt), marked.iif_cb(xt, "a"), rtol=1e-9
    )


# -- the renewal models keep their Wald bounds too --------------------------


def _interior_renewal_data():
    """Eight items whose Kijima-I fit has q = 2.6, inside its range."""
    rng = np.random.default_rng(8)
    rows = []
    for k in range(8):
        end = rng.uniform(6000, 12000)
        n = rng.poisson(2e-4 * end**1.35)
        ts = np.sort(end * rng.random(n) ** (1 / 1.35))
        rows += [(t, k, 0) for t in ts] + [(end, k, 1)]
    return tuple(map(np.array, zip(*rows)))


_FLEET = (np.array(REPAIR_FLEET_X, float), REPAIR_FLEET_I, REPAIR_FLEET_C)
RENEWALS = {
    # fitter, data, whether the restoration parameter is on its edge
    "GeneralizedOneRenewal": (GeneralizedOneRenewal, _FLEET, False),
    "GeneralizedRenewal-interior": (
        GeneralizedRenewal,
        _interior_renewal_data(),
        False,
    ),
    "GeneralizedRenewal-edge": (GeneralizedRenewal, _FLEET, True),
    "ARA-edge": (ARA, _FLEET, True),
    "ARI-edge": (ARI, _FLEET, True),
}


def _quiet(call, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


@pytest.mark.parametrize("case", list(RENEWALS))
def test_restored_renewal_model_gives_the_fits_wald_bounds(case):
    fitter, (x, i, c), on_edge = RENEWALS[case]
    model = _quiet(fitter.fit, x, i, c=c)
    assert (model._edge_value() is not None) == on_edge
    restored = _restored(model)
    np.testing.assert_array_equal(
        restored.covariance(), _quiet(model.covariance)
    )
    np.testing.assert_array_equal(
        _quiet(restored.standard_errors), _quiet(model.standard_errors)
    )
    restoration = model.parameter_names[0]
    for name in model.parameter_names[1:]:
        for bound in ("two-sided", "lower", "upper"):
            np.testing.assert_array_equal(
                restored.param_cb(name, bound=bound),
                model.param_cb(name, bound=bound),
            )
    if on_edge:
        # Its interval on the edge is the profile likelihood's, which
        # needs the data: refused, saying what the restored model has.
        for call in (
            lambda: restored.param_cb(restoration),
            restored.summary,
        ):
            with pytest.raises(
                ValueError, match="keeps the parameter covariance"
            ):
                call()
    else:
        np.testing.assert_array_equal(
            restored.param_cb(restoration), model.param_cb(restoration)
        )
        assert restored.summary().equals(model.summary())
    with pytest.raises(ValueError, match="keeps the parameter covariance"):
        restored.param_cb(restoration, method="lr")
    with pytest.raises(ValueError, match="keeps the parameter covariance"):
        restored.aic()
    # Saved again, it is the same dictionary.
    assert restored.to_dict() == json_round_trip(model.to_dict())


def test_a_renewal_dict_saved_without_a_covariance_still_refuses():
    model = GeneralizedOneRenewal.fit(*_FLEET[:2], c=_FLEET[2])
    saved = json_round_trip(model.to_dict())
    del saved["covariance"]
    old = type(model).from_dict(saved)
    np.testing.assert_array_equal(old.model.params, model.model.params)
    for call in (old.covariance, old.standard_errors, old.summary):
        with pytest.raises(ValueError, match="holds no parameter covariance"):
            call()
