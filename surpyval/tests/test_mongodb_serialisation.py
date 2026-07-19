"""MongoDB (BSON) compatibility of every serialisable model.

``json.dumps`` is more forgiving than MongoDB's BSON encoder: JSON
silently stringifies integer dictionary keys and accepts numpy's
``np.str_``/``np.float64`` scalars, while pymongo rejects non-string
keys and numpy integer scalars and arrays outright. Storing a fitted
model therefore needs ``to_dict`` to emit only native Python types.

For every serialisable model these tests run the full MongoDB path
without a server:

- ``bson.encode(model.to_dict())`` — what ``insert_one`` does — must
  succeed, which requires string keys and native values throughout;
- the decoded document gets an ``ObjectId`` under ``"_id"`` — what
  ``find_one`` returns — and ``surpyval.from_dict`` must restore the
  right model from it, reproducing its predictions.

A strict walker additionally pins the native-types contract directly,
so a failure names the offending path rather than relying on pymongo's
error message.
"""

import warnings

import bson
import numpy as np
import pytest

import surpyval
from surpyval import (
    AdditiveHazards,
    BuckleyJames,
    CoxPH,
    KaplanMeier,
    MixtureModel,
    Turnbull,
    Weibull,
    WeibullPH,
)
from surpyval.degradation import (
    DegradationAnalysis,
    GammaProcess,
    WienerProcess,
)
from surpyval.recurrent import (
    CrowAMSAA,
    GeneralizedRenewal,
    NonParametricCounting,
    ProportionalIntensityHPP,
)
from surpyval.recurrent.competing_risks import (
    CauseSpecificMCF,
    CauseSpecificNHPP,
)
from surpyval.univariate.competing_risks import (
    CompetingRisks,
    FineGray,
    ParametricCompetingRisks,
)

# -- data helpers -------------------------------------------------------------

X0 = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0]
C0 = [0, 0, 1, 0, 0, 1, 0, 0]


def _semipar_data(seed=0, n=80):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    lin = 0.3 * Z[:, 0] - 0.2 * Z[:, 1]
    x = np.abs(rng.weibull(1.5, n) * 20 * np.exp(-lin)) + 0.5
    return x, Z, np.zeros(n)


def _cr_data(seed=0, n=120):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    x = np.abs(rng.weibull(1.3, n) * 15) + 0.2
    e = rng.choice([1, 2], n)
    c = (rng.random(n) < 0.2).astype(int)
    e = np.where(c == 1, None, e)
    return x, Z, e, c


def _recurrent_marked_data(seed=5, n_items=15):
    rng = np.random.default_rng(seed)
    xs, ii, cc, es = [], [], [], []
    for item in range(n_items):
        for _ in range(int(rng.integers(2, 5))):
            xs.append(float(rng.uniform(0, 400)))
            ii.append(item)
            cc.append(0)
            es.append(rng.choice(["A", "B"]))
        xs.append(400.0)
        ii.append(item)
        cc.append(1)
        es.append(None)
    return (
        np.array(xs),
        np.array(ii),
        np.array(cc),
        np.array(es, dtype=object),
    )


def _pi_data(seed=0, n_items=15):
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


def _deg_data(seed=0):
    rng = np.random.default_rng(seed)
    x = np.tile(np.arange(100, 1100, 100), 4).astype(float)
    slopes = np.repeat([0.31, 0.28, 0.44, 0.37], 10)
    i = np.repeat([1, 2, 3, 4], 10)
    y = 10 + slopes * x + rng.normal(0, 1, x.size)
    return x, y, i


def _process_data(seed=2, monotone=False):
    rng = np.random.default_rng(seed)
    xs, ys, ii = [], [], []
    for u in range(10):
        t = np.arange(0, 20, 2.0)
        inc = rng.normal(1.0, 0.5 if monotone else 1.0, t.size)
        if monotone:
            inc = np.abs(inc)
        xs.append(t)
        ys.append(np.cumsum(inc))
        ii.append(np.full(t.size, u))
    return tuple(np.concatenate(z) for z in (xs, ys, ii))


def _fit_degradation():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return DegradationAnalysis.fit(*_deg_data(), threshold=150)


# -- one builder + prediction check per serialisable model --------------------


def _check_sf(model, restored, t=(2.0, 5.0, 9.0)):
    assert np.allclose(model.sf(np.array(t)), restored.sf(np.array(t)))


def _check_sf_Z(model, restored):
    t = np.array([2.0, 6.0])
    z = np.array([1.0, 0.5])
    assert np.allclose(model.sf(t, Z=z), restored.sf(t, Z=z))


def _check_cif_grid(model, restored):
    t = np.array([50.0, 200.0, 350.0])
    for cause in model.event_types:
        assert np.allclose(
            model.cif(t, cause), restored.cif(t, cause), equal_nan=True
        )


def _fit_parametric():
    return Weibull.fit(X0, C0)


def _fit_non_parametric():
    return KaplanMeier.fit(X0, C0)


def _fit_turnbull():
    return Turnbull.fit([[1, 2], [3, 4], [5, 6], [4, 8], [7, 9], [2, 10]])


def _fit_weibull_ph():
    x, Z, c = _semipar_data()
    return WeibullPH.fit(x=x, Z=Z, c=c)


def _fit_cox():
    x, Z, c = _semipar_data(1)
    return CoxPH.fit(x, Z, c=c)


def _fit_additive_hazards():
    x, Z, c = _semipar_data(2)
    return AdditiveHazards.fit(x, Z, c=c)


def _fit_buckley_james():
    x, Z, c = _semipar_data(3)
    return BuckleyJames.fit(x, Z, c=c)


def _fit_mixture():
    x = np.concatenate([Weibull.random(50, 8, 3), Weibull.random(50, 40, 4)])
    model = MixtureModel(dist=Weibull, m=2)
    model.fit(x=x)
    return model


def _fit_fine_gray():
    x, Z, e, c = _cr_data()
    return FineGray.fit(x, Z, e, c=c, cause=1)


def _fit_parametric_cr():
    x, _, e, c = _cr_data()
    return ParametricCompetingRisks.fit(x, e, c=c)


def _fit_competing_risks():
    rng = np.random.default_rng(3)
    x = np.abs(rng.weibull(1.3, 60) * 15) + 0.2
    e = rng.choice([1, 2], 60)
    return CompetingRisks.fit(x, e)


def _fit_cs_mcf():
    x = np.array([5, 10, 15, 4, 9, 12, 20], dtype=float)
    i = np.array([1, 1, 1, 2, 2, 3, 3])
    c = np.array([0, 0, 1, 0, 1, 0, 1])
    e = np.array(["A", "B", "A", "A", "B", "B", "A"])
    return CauseSpecificMCF.fit(x, i, c, e=e)


def _fit_cs_nhpp():
    x, i, c, e = _recurrent_marked_data()
    return CauseSpecificNHPP.fit(x, i, c, e=e)


def _fit_np_counting():
    x = np.array([5.0, 12.0, 20.0, 8.0, 15.0, 25.0])
    i = np.array([1, 1, 1, 2, 2, 2])
    c = np.array([0, 0, 1, 0, 0, 1])
    return NonParametricCounting.fit(x, i, c)


def _fit_crow_amsaa():
    return CrowAMSAA.fit(
        np.array([10.0, 25.0, 45.0, 70.0, 100.0, 135.0, 175.0])
    )


def _fit_proportional_intensity():
    x, i, c, Z = _pi_data()
    return ProportionalIntensityHPP.fit(x, Z, i=i, c=c)


def _fit_renewal():
    return GeneralizedRenewal.fit_from_parameters(
        [50.0, 2.0], 0.3, kijima="i", dist=Weibull
    )


def _fit_induced():
    return _fit_degradation().induced_life(n_samples=2000, random_state=3)


def _fit_wiener():
    x, y, i = _process_data()
    return WienerProcess.fit(x, y, i, threshold=20.0)


def _fit_gamma():
    x, y, i = _process_data(1, monotone=True)
    return GammaProcess.fit(x, y, i, threshold=15.0)


def _check_R(m, r):
    np.testing.assert_allclose(m.R, r.R)


def _check_beta(m, r):
    np.testing.assert_allclose(m.beta, r.beta)


def _check_per_cause_params(m, r):
    for k in m.causes:
        np.testing.assert_allclose(m.models[k].params, r.models[k].params)


def _check_mcf_per_cause(m, r):
    grid = np.array([6.0, 11.0])
    for k in m.event_types:
        np.testing.assert_allclose(
            m.mcf(grid, k), r.mcf(grid, k), equal_nan=True
        )


def _check_mcf_hat(m, r):
    np.testing.assert_allclose(m.mcf_hat, r.mcf_hat)


def _check_cif(m, r):
    grid = np.array([50.0, 150.0])
    np.testing.assert_allclose(m.cif(grid), r.cif(grid))


def _check_cif_Z(m, r):
    grid = np.array([100.0, 300.0])
    z = np.array([0.5])
    np.testing.assert_allclose(m.cif(grid, z), r.cif(grid, z))


def _check_renewal_params(m, r):
    np.testing.assert_allclose(m.model.params, r.model.params)


def _check_life_model_params(m, r):
    np.testing.assert_allclose(m.life_model.params, r.life_model.params)


def _check_ff_late(m, r):
    grid = np.array([300.0, 450.0])
    np.testing.assert_allclose(m.ff(grid), r.ff(grid))


def _check_process_sf(m, r):
    grid = np.array([5.0, 15.0])
    np.testing.assert_allclose(m.sf(grid), r.sf(grid))


CASES = {
    "parametric": (_fit_parametric, _check_sf),
    "parametric_with_data": (_fit_parametric, _check_sf),
    "non_parametric": (_fit_non_parametric, _check_R),
    "turnbull_interval": (_fit_turnbull, _check_R),
    "parametric_regression": (_fit_weibull_ph, _check_sf_Z),
    "cox": (_fit_cox, _check_sf_Z),
    "additive_hazards": (_fit_additive_hazards, _check_sf_Z),
    "buckley_james": (_fit_buckley_james, _check_beta),
    "mixture": (_fit_mixture, _check_sf),
    "fine_gray": (_fit_fine_gray, _check_beta),
    "parametric_competing_risks": (
        _fit_parametric_cr,
        _check_per_cause_params,
    ),
    "competing_risks": (
        _fit_competing_risks,
        lambda m, r: np.testing.assert_allclose(m.x, r.x),
    ),
    "cause_specific_mcf": (_fit_cs_mcf, _check_mcf_per_cause),
    "cause_specific_nhpp": (_fit_cs_nhpp, _check_cif_grid),
    "np_counting_mcf": (_fit_np_counting, _check_mcf_hat),
    "parametric_recurrence": (_fit_crow_amsaa, _check_cif),
    "proportional_intensity": (_fit_proportional_intensity, _check_cif_Z),
    "renewal": (_fit_renewal, _check_renewal_params),
    "degradation": (_fit_degradation, _check_life_model_params),
    "induced_failure_distribution": (_fit_induced, _check_ff_late),
    "wiener_process": (_fit_wiener, _check_process_sf),
    "gamma_process": (_fit_gamma, _check_process_sf),
}


# -- the BSON-native contract -------------------------------------------------


def _assert_bson_native(obj, path="$"):
    """Every key a str, every value a BSON-encodable native type."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            assert isinstance(k, str), f"non-string key {k!r} at {path}"
            _assert_bson_native(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for j, v in enumerate(obj):
            _assert_bson_native(v, f"{path}[{j}]")
    else:
        assert obj is None or isinstance(
            obj, (str, bool, int, float)
        ), f"non-native {type(obj).__name__} at {path}"
        assert not isinstance(obj, np.ndarray), f"numpy array at {path}"


def _mongo_round_trip(model_dict):
    """encode -> decode -> add _id: what insert_one/find_one do."""
    doc = bson.decode(bson.encode(model_dict))
    doc["_id"] = bson.ObjectId()
    return doc


@pytest.mark.parametrize("name", CASES)
def test_mongo_round_trip(name):
    build, check = CASES[name]
    model = build()
    if name == "parametric_with_data":
        model_dict = model.to_dict(with_data=True)
    else:
        model_dict = model.to_dict()

    _assert_bson_native(model_dict)
    restored = surpyval.from_dict(_mongo_round_trip(model_dict))
    assert type(restored).__name__ == type(model).__name__
    check(model, restored)


def test_numpy_cause_labels_are_stored_native():
    # Cause labels fed in as numpy scalars must not leak numpy types
    # into the document -- BSON rejects np.int64 outright.
    x = np.array([5, 10, 15, 4, 9, 12, 20], dtype=float)
    i = np.array([1, 1, 1, 2, 2, 3, 3])
    c = np.array([0, 0, 1, 0, 1, 0, 1])
    e = np.array([1, 2, 1, 1, 2, 2, 1])
    model = CauseSpecificMCF.fit(x, i, c, e=e)
    d = model.to_dict()
    for label in d["event_types"]:
        assert type(label) in (int, str)
    bson.encode(d)


def test_from_dict_ignores_mongo_id():
    model = Weibull.fit(X0, C0)
    doc = _mongo_round_trip(model.to_dict())
    assert "_id" in doc
    restored = surpyval.from_dict(doc)
    _check_sf(model, restored)
