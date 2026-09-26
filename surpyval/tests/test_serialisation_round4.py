"""Strict-JSON serialisation, ``to_json(with_data=True)`` and checked
class readers, for every serialisable model.

- Strict JSON: ``to_dict`` used to hand ``inf``/``-inf``/``nan`` to
  ``json``, which writes the non-standard ``Infinity``/``NaN`` literals
  that strict parsers (JavaScript, many databases) reject. Now every
  non-finite float is written as ``null`` and recorded under
  ``"non_finite"`` (JSON Pointers by kind), and every reader restores it.
  Dictionaries written before (schema 0/1, holding the literals) still
  load.
- ``to_json(path, with_data=True)`` mirrors ``to_dict(with_data=True)``
  for the models that store their data.
- A class's own ``from_dict`` applies the checks of ``surpyval.from_dict``
  (schema, missing entry named, parameter bounds): it used to skip them.

The round trip is checked for one model of every registered class (and
the variants that carry non-finite values) through
``json.dumps(allow_nan=False)`` / ``json.loads``, comparing every public
method of the restored model with the original's.
"""

import inspect
import json
import warnings

import numpy as np
import pytest

import surpyval
from surpyval import (
    AcceleratedLife,
    AdditiveHazards,
    BuckleyJames,
    CoxPH,
    InstantlyOccurs,
    KaplanMeier,
    MixtureModel,
    NelsonAalen,
    NeverOccurs,
    NonParametric,
    Parametric,
    Power,
    RoystonParmar,
    SurpyvalData,
    Turnbull,
    Weibull,
    WeibullFrailty,
    WeibullPH,
)
from surpyval.serialisation import (
    _PARAMETERIZATIONS,
    _TAGGED_MODELS,
    NON_FINITE_KEY,
    SCHEMA_VERSION,
    decode_non_finite,
    encode_non_finite,
)

# -- data ---------------------------------------------------------------------

X0 = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 11.0]
C0 = [0, 0, 1, 0, 0, 1, 0, 0]


def _semipar_data(seed=0, n=60):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    lin = 0.3 * Z[:, 0] - 0.2 * Z[:, 1]
    x = np.abs(rng.weibull(1.5, n) * 20 * np.exp(-lin)) + 0.5
    c = (rng.random(n) < 0.2).astype(int)
    return x, Z, c


def _cr_data(seed=0, n=80):
    rng = np.random.default_rng(seed)
    Z = rng.normal(0, 1, (n, 2))
    x = np.abs(rng.weibull(1.3, n) * 15) + 0.2
    e = rng.choice([1, 2], n)
    c = (rng.random(n) < 0.2).astype(int)
    e = np.where(c == 1, None, e)
    return x, Z, e, c


def _recurrent_marked_data(seed=5, n_items=12):
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


def _pi_data(seed=0, n_items=12):
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


def _process_data(seed=2, monotone=False):
    rng = np.random.default_rng(seed)
    xs, ys, ii = [], [], []
    for u in range(8):
        t = np.arange(0, 20, 2.0)
        inc = rng.normal(1.0, 0.5 if monotone else 1.0, t.size)
        if monotone:
            inc = np.abs(inc)
        xs.append(t)
        ys.append(np.cumsum(inc))
        ii.append(np.full(t.size, u))
    return tuple(np.concatenate(z) for z in (xs, ys, ii))


def _forest_data(seed=0, n=60):
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=(n, 2))
    x = rng.exponential(np.exp(-Z @ np.array([0.9, -0.6]))) * 15 + 1
    c = (rng.uniform(size=n) < 0.2).astype(int)
    return x, Z, c


# -- one builder per model (and per non-finite-carrying variant) --------------


def _degradation():
    from surpyval.degradation import DegradationAnalysis

    rng = np.random.default_rng(0)
    x = np.tile(np.arange(100, 1100, 100), 4).astype(float)
    y = 10 + np.repeat([0.31, 0.28, 0.44, 0.37], 10) * x
    y = y + rng.normal(0, 1, x.size)
    return DegradationAnalysis.fit(x, y, np.repeat([1, 2, 3, 4], 10), 150)


def _frailty():
    rng = np.random.default_rng(1)
    xs, cs, zs, gs = [], [], [], []
    for g in range(20):
        u = rng.gamma(1 / 0.6, 0.6)
        for _ in range(5):
            z = rng.normal()
            t = 20 * (-np.log(rng.uniform()) / (np.exp(0.8 * z) * u)) ** (
                1 / 1.8
            )
            xs.append(min(t, 40))
            cs.append(0 if t <= 40 else 1)
            zs.append(z)
            gs.append(g)
    return WeibullFrailty.fit(
        x=np.array(xs),
        c=np.array(cs),
        Z=np.array(zs).reshape(-1, 1),
        groups=np.array(gs),
    )


def _destructive():
    from surpyval.degradation import DestructiveDegradation

    rng = np.random.default_rng(1)
    x = np.repeat([10.0, 20.0, 30.0, 40.0], 6)
    y = np.exp(4.0 - 0.02 * x + rng.normal(0, 0.1, 24))
    c = np.zeros(24, dtype=int)
    c[0] = 1
    return DestructiveDegradation.fit(x, y, threshold=20, c=c)


def _copula_with_km_margin():
    from surpyval.multivariate import Clayton

    # An uncensored Kaplan-Meier margin ends with H = inf and an undefined
    # (nan) Greenwood term: the nested margin dict carries its own record.
    return Clayton.from_params(
        2.0,
        margins=[
            KaplanMeier.fit([1.0, 2.0, 3.0, 4.0, 5.0]),
            Weibull.from_params([10, 2]),
        ],
    )


def _accelerated_life():
    rng = np.random.default_rng(7)
    xs, Zs = [], []
    for s in (1.0, 2.0, 3.0, 4.0):
        xs.append(500.0 * s**-1.2 * rng.weibull(2.2, 20))
        Zs.append(np.full(20, s))
    return AcceleratedLife(Weibull, Power).fit(
        x=np.concatenate(xs), Z=np.concatenate(Zs).reshape(-1, 1)
    )


def _cox_delayed_entry():
    x, Z, c = _semipar_data(1)
    tl = np.where(np.arange(x.size) % 3 == 0, 0.2, -np.inf)
    return CoxPH.fit(x, Z, c=c, tl=np.minimum(tl, x - 0.1))


def _mixture():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.weibull(3, 40) * 8, rng.weibull(4, 40) * 40])
    model = MixtureModel(dist=Weibull, m=2)
    model.fit(x=x)
    return model


def _tree():
    from surpyval.beta.ml.forest.tree import SurvivalTree

    x, Z, c = _forest_data()
    np.random.seed(1)
    return SurvivalTree.fit(x=x, Z=Z, c=c, kind="weibull", max_depth=2)


def _forest():
    from surpyval.beta.ml.forest.forest import RandomSurvivalForest

    x, Z, c = _forest_data()
    np.random.seed(1)
    return RandomSurvivalForest.fit(x=x, Z=Z, c=c, n_trees=2, max_depth=2)


def _build(name):
    from surpyval.degradation import GammaProcess, WienerProcess
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
        CompetingRisksProportionalHazards,
        FineGray,
        ParametricCompetingRisks,
    )

    builders = {
        "parametric": lambda: Weibull.fit(X0, C0),
        "parametric_truncated": lambda: Weibull.fit(
            X0, C0, tl=[1, 1, 2, 2, 0, 0, 0, 0]
        ),
        "parametric_lfp_zi": lambda: Weibull.fit(
            [0, 0, 3, 4, 5, 6, 7, 8, 9],
            c=[0, 0, 0, 0, 0, 0, 0, 1, 1],
            lfp=True,
            zi=True,
        ),
        "parametric_offset": lambda: Weibull.fit(
            [13, 14, 15, 16, 17, 18], offset=True
        ),
        "kaplan_meier": lambda: KaplanMeier.fit(X0, C0),
        # every unit dies: H ends at inf, the Greenwood term at nan
        "kaplan_meier_uncensored": lambda: KaplanMeier.fit(X0),
        "nelson_aalen_truncated": lambda: NelsonAalen.fit(
            [3, 4, 5, 6, 7, 8], tl=[1, 1, 2, 2, 0, 0]
        ),
        "turnbull": lambda: Turnbull.fit(
            [[1, 2], [3, np.inf], [5, 6], [4, 8], [7, 9], [2, 10]]
        ),
        "parametric_regression": lambda: WeibullPH.fit(*_semipar_data()[:2]),
        "accelerated_life": _accelerated_life,
        "cox": lambda: CoxPH.fit(*_semipar_data(1)[:2]),
        "cox_delayed_entry": _cox_delayed_entry,
        "frailty": _frailty,
        "additive_hazards": lambda: AdditiveHazards.fit(*_semipar_data(2)[:2]),
        "buckley_james": lambda: BuckleyJames.fit(*_semipar_data(3)[:2]),
        "mixture": _mixture,
        "royston_parmar": lambda: RoystonParmar.fit(
            np.random.default_rng(1).weibull(2, 60) * 10, df=3
        ),
        "fine_gray": lambda: FineGray.fit(
            *_cr_data()[:3], c=_cr_data()[3], cause=1
        ),
        "cr_proportional_hazards": lambda: (
            CompetingRisksProportionalHazards.fit(
                *_cr_data()[:3], c=_cr_data()[3]
            )
        ),
        "parametric_competing_risks": lambda: ParametricCompetingRisks.fit(
            _cr_data()[0], _cr_data()[2], c=_cr_data()[3]
        ),
        "competing_risks": lambda: CompetingRisks.fit(
            np.abs(np.random.default_rng(3).weibull(1.3, 40) * 15) + 0.2,
            np.random.default_rng(4).choice([1, 2], 40),
        ),
        "cause_specific_mcf": lambda: CauseSpecificMCF.fit(
            np.array([5, 10, 15, 4, 9, 12, 20], dtype=float),
            np.array([1, 1, 1, 2, 2, 3, 3]),
            np.array([0, 0, 1, 0, 1, 0, 1]),
            e=np.array(["A", "B", "A", "A", "B", "B", "A"]),
        ),
        "cause_specific_nhpp": lambda: CauseSpecificNHPP.fit(
            *_recurrent_marked_data()[:3], e=_recurrent_marked_data()[3]
        ),
        "mcf": lambda: NonParametricCounting.fit(
            np.array([5.0, 12.0, 20.0, 8.0, 15.0, 25.0]),
            np.array([1, 1, 1, 2, 2, 2]),
            np.array([0, 0, 1, 0, 0, 1]),
        ),
        "parametric_recurrence": lambda: CrowAMSAA.fit(
            np.array([10.0, 25.0, 45.0, 70.0, 100.0, 135.0, 175.0])
        ),
        "proportional_intensity": lambda: ProportionalIntensityHPP.fit(
            _pi_data()[0],
            _pi_data()[3],
            i=_pi_data()[1],
            c=_pi_data()[2],
        ),
        "renewal": lambda: GeneralizedRenewal.fit_from_parameters(
            [50.0, 2.0], 0.3, kijima="i", dist=Weibull
        ),
        "degradation": _degradation,
        "induced_failure_distribution": lambda: _degradation().induced_life(
            n_samples=500, random_state=3
        ),
        "wiener_process": lambda: WienerProcess.fit(
            *_process_data(), threshold=20.0
        ),
        "gamma_process": lambda: GammaProcess.fit(
            *_process_data(1, monotone=True), threshold=15.0
        ),
        "destructive": _destructive,
        "copula": _copula_with_km_margin,
        "survival_tree": _tree,
        "random_survival_forest": _forest,
        "never_occurs": lambda: NeverOccurs,
        "instantly_occurs": lambda: InstantlyOccurs,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return builders[name]()


CASES = [
    "parametric",
    "parametric_truncated",
    "parametric_lfp_zi",
    "parametric_offset",
    "kaplan_meier",
    "kaplan_meier_uncensored",
    "nelson_aalen_truncated",
    "turnbull",
    "parametric_regression",
    "accelerated_life",
    "cox",
    "cox_delayed_entry",
    "frailty",
    "additive_hazards",
    "buckley_james",
    "mixture",
    "royston_parmar",
    "fine_gray",
    "cr_proportional_hazards",
    "parametric_competing_risks",
    "competing_risks",
    "cause_specific_mcf",
    "cause_specific_nhpp",
    "mcf",
    "parametric_recurrence",
    "proportional_intensity",
    "renewal",
    "degradation",
    "induced_failure_distribution",
    "wiener_process",
    "gamma_process",
    "destructive",
    "copula",
    "survival_tree",
    "random_survival_forest",
    "never_occurs",
    "instantly_occurs",
]


class _LazyModels(dict):
    """Each model is fitted the first time a test asks for it (building
    them all up front cost every xdist worker the full ~15 s)."""

    def __missing__(self, name):
        self[name] = _build(name)
        return self[name]

    def values(self):
        return [self[name] for name in CASES]


_MODELS = _LazyModels()


@pytest.fixture
def models():
    return _MODELS


def _model_class(model):
    return model if isinstance(model, type) else type(model)


def _dicts(model):
    """``to_dict()`` and, where supported, ``to_dict(with_data=True)``."""
    out = {"plain": model.to_dict()}
    if "with_data" in inspect.signature(model.to_dict).parameters:
        out["with_data"] = model.to_dict(with_data=True)
    return out


def _strict_loads(text):
    """``json.loads`` that refuses the non-standard literals, as a strict
    parser (JavaScript's ``JSON.parse``) does."""

    def refuse(literal):
        raise ValueError(f"non-standard JSON literal {literal}")

    return json.loads(text, parse_constant=refuse)


# -- every registered class is covered ---------------------------------------


def test_cases_cover_every_registered_model_class(models):
    registered = set(_TAGGED_MODELS) | {
        c for _, c in _PARAMETERIZATIONS.values()
    }
    covered = {_model_class(m).__name__ for m in models.values()}
    # "InstantlyOccurs"/"NeverOccurs" are classes that are the model.
    covered |= {m.__name__ for m in models.values() if isinstance(m, type)}
    assert registered <= covered, registered - covered


# -- (a) strict JSON ----------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_every_dict_is_strict_json(name, models):
    for kind, d in _dicts(models[name]).items():
        # used to raise "Out of range float values are not JSON compliant"
        text = json.dumps(d, allow_nan=False)
        _strict_loads(text)
        assert d["schema"] == SCHEMA_VERSION


def test_non_finite_values_are_null_with_a_record(models):
    d = models["kaplan_meier_uncensored"].to_dict()
    assert d["H"][-1] is None and d["greenwood"][-1] is None
    last = len(d["H"]) - 1
    assert d[NON_FINITE_KEY] == {
        "inf": [f"/H/{last}"],
        "nan": [f"/greenwood/{last}"],
    }
    d = models["parametric"].to_dict(with_data=True)
    assert d["data"]["t"][0] == [None, None]
    assert "/data/t/0/0" in d[NON_FINITE_KEY]["-inf"]
    assert "/data/t/0/1" in d[NON_FINITE_KEY]["inf"]
    # no non-finite value, no record
    assert NON_FINITE_KEY not in models["parametric"].to_dict()


def test_restored_values_are_the_original_non_finite_ones(models):
    km = models["kaplan_meier_uncensored"]
    r = surpyval.from_dict(
        json.loads(json.dumps(km.to_dict(), allow_nan=False))
    )
    assert np.isposinf(r.H[-1]) and np.isnan(r.greenwood[-1])
    np.testing.assert_array_equal(r.H, km.H)

    wb = models["parametric_truncated"]
    d = json.loads(json.dumps(wb.to_dict(with_data=True), allow_nan=False))
    r = Parametric.from_dict(d)
    np.testing.assert_array_equal(r.data["t"], wb.data["t"])
    assert np.isposinf(r.data["t"][:, 1]).all()

    cox = models["cox_delayed_entry"]
    d = json.loads(json.dumps(cox.to_dict(), allow_nan=False))
    r = surpyval.from_dict(d)
    np.testing.assert_array_equal(r.tl, cox.tl)
    assert np.isneginf(r.tl).any()


# -- the full round trip: every public method ---------------------------------

# Methods not compared: constructors and writers, plots, anything random,
# and the mixture's EM steps (which refit the model in place).
_SKIP_PREFIXES = (
    "fit",
    "from_",
    "to_",
    "plot",
    "random",
    "simulate",
    "sample",
    "bootstrap",
)
_SKIP = {
    "EM",
    "expectation",
    "maximisation",
    "initialise_params",
    "get_uniform_random_number",
    "initialize_simulation",
}
_T = np.array([2.0, 6.0, 9.0])
_Z = np.array([[0.5, -0.2], [0.1, 0.3], [1.0, 0.0]])


def _candidate_args(model):
    out = [
        (),
        (_T,),
        (_T, _Z),
        (_T, np.full((3, 1), 2.5)),
        (_T, _Z[0]),
        (np.array([0.1, 0.5, 0.9]),),
        (6.0,),
    ]
    for attr in ("event_types", "causes"):
        for label in list(getattr(model, attr, None) or [])[:2]:
            out += [(_T, label), (_T, _Z[0], label)]
    return out


def _same(a, b):
    if isinstance(a, (list, tuple)):
        return (
            isinstance(b, (list, tuple))
            and len(a) == len(b)
            and all(_same(x, y) for x, y in zip(a, b))
        )
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if hasattr(a, "columns") and hasattr(a, "to_numpy"):
        return list(a.columns) == list(b.columns) and _same(
            a.to_numpy(), b.to_numpy()
        )
    if isinstance(a, (str, bool, type(None))):
        return a == b
    if isinstance(a, (np.ndarray, np.generic, int, float)):
        a, b = np.asarray(a), np.asarray(b)
        if a.dtype.kind in "biufc" and b.dtype.kind in "biufc":
            return a.shape == b.shape and np.allclose(
                a, b, rtol=1e-9, atol=0, equal_nan=True
            )
        return a.shape == b.shape and all(
            _same(x, y) for x, y in zip(a.ravel(), b.ravel())
        )
    # other objects (fitted sub-models, result records): same type
    return type(a) is type(b)


def _public_methods(model):
    """The public methods the model's class defines (not callables held
    as instance attributes, such as a life model's ``fun``)."""
    cls = _model_class(model)
    for name in sorted(dir(cls)):
        if name.startswith("_") or name.startswith(_SKIP_PREFIXES):
            continue
        if name in _SKIP or not callable(getattr(cls, name, None)):
            continue
        if isinstance(getattr(cls, name), type):
            continue
        yield name, getattr(model, name)


def _call(method, args):
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return method(*args)


def _compare_every_method(model, restored):
    compared = 0
    for name, method in _public_methods(model):
        for args in _candidate_args(model):
            try:
                expected = _call(method, args)
            except Exception:
                continue
            try:
                got = _call(getattr(restored, name), args)
            except ValueError as err:
                # The documented exception: what needs the fitted data
                # (plot data, residuals, likelihood inference) says so.
                assert "data" in str(err), (name, err)
                break
            assert _same(expected, got), name
            compared += 1
            break
    return compared


@pytest.mark.parametrize("name", CASES)
def test_json_round_trip_reproduces_every_public_method(name, models):
    model = models[name]
    for kind, d in _dicts(model).items():
        text = json.dumps(d, allow_nan=False)
        restored = surpyval.from_dict(_strict_loads(text))
        assert _model_class(restored) is _model_class(model)
        # the class's own reader gives the same model
        again = _model_class(model).from_dict(_strict_loads(text))
        assert _model_class(again) is _model_class(model)
        if isinstance(model, type):
            assert restored is model and again is model
            continue
        assert json.dumps(again.to_dict()) == json.dumps(restored.to_dict())
        assert _compare_every_method(model, restored) > 0


def test_with_data_restores_every_method_of_the_parametric_models(models):
    # With the data stored, nothing may raise on the restored model.
    for name in ("parametric", "parametric_truncated", "kaplan_meier"):
        model = models[name]
        text = json.dumps(model.to_dict(with_data=True), allow_nan=False)
        restored = surpyval.from_dict(json.loads(text))
        for method_name, method in _public_methods(model):
            for args in _candidate_args(model):
                try:
                    expected = _call(method, args)
                except Exception:
                    continue
                got = _call(getattr(restored, method_name), args)
                assert _same(expected, got), (name, method_name)
                break


# -- old dictionaries (NaN/Infinity literals) still load ----------------------


def _legacy_text(d, schema=1):
    """What SurPyval wrote before: the non-finite floats themselves, as
    json's NaN/Infinity literals, under schema 1."""
    old = decode_non_finite(d)
    old["schema"] = schema
    return json.dumps(old)


@pytest.mark.parametrize(
    "name",
    [
        "kaplan_meier_uncensored",
        "parametric_truncated",
        "cox_delayed_entry",
        "copula",
    ],
)
def test_legacy_dicts_with_nan_and_infinity_still_load(name, models):
    model = models[name]
    d = _dicts(model).get("with_data", model.to_dict())
    text = _legacy_text(d)
    assert "Infinity" in text or "NaN" in text
    restored = surpyval.from_dict(json.loads(text))
    _compare_every_method(model, restored)
    restored = _model_class(model).from_dict(json.loads(text))
    _compare_every_method(model, restored)


def test_legacy_file_loads(tmp_path):
    km = KaplanMeier.fit(X0)
    fp = tmp_path / "old_km.json"
    fp.write_text(_legacy_text(km.to_dict(), schema=0))
    assert "Infinity" in fp.read_text()
    restored = surpyval.from_json(fp)
    assert np.isposinf(restored.H[-1])
    np.testing.assert_allclose(restored.sf(_T), km.sf(_T))
    assert np.isposinf(NonParametric.from_json(fp).H[-1])


def test_legacy_cox_null_entry_times_still_read_as_minus_inf(models):
    # Schema-1 Cox dicts wrote a missing entry time as a bare null.
    cox = models["cox_delayed_entry"]
    d = decode_non_finite(cox.to_dict())
    d["tl"] = [None if np.isneginf(v) else v for v in d["tl"]]
    d["schema"] = 1
    restored = surpyval.from_dict(json.loads(json.dumps(d, allow_nan=False)))
    np.testing.assert_array_equal(restored.tl, cox.tl)


# -- (b) to_json(with_data=True) ---------------------------------------------


@pytest.mark.parametrize("name", ["parametric_truncated", "kaplan_meier"])
def test_to_json_with_data(name, models, tmp_path):
    model = models[name]
    fp = tmp_path / "model.json"
    model.to_json(fp, with_data=True)
    on_disk = _strict_loads(fp.read_text())
    assert on_disk == json.loads(
        json.dumps(model.to_dict(with_data=True), allow_nan=False)
    )
    assert "data" in on_disk
    restored = surpyval.from_json(fp)
    if name == "kaplan_meier":
        expected = model.bootstrap_cb(_T, B=20, random_state=1)
        got = restored.bootstrap_cb(_T, B=20, random_state=1)
        np.testing.assert_allclose(expected, got)
    else:
        assert restored.bic() == pytest.approx(model.bic())
        np.testing.assert_array_equal(restored.data["t"], model.data["t"])

    model.to_json(fp)
    assert "data" not in _strict_loads(fp.read_text())


def test_to_json_with_data_refused_where_to_dict_has_no_data(models, tmp_path):
    with pytest.raises(TypeError, match="does not store the fitted data"):
        models["cox"].to_json(tmp_path / "cox.json", with_data=True)


@pytest.mark.parametrize("name", CASES)
def test_to_json_writes_strict_json(name, models, tmp_path):
    model = models[name]
    fp = tmp_path / "model.json"
    model.to_json(fp)
    _strict_loads(fp.read_text())
    restored = surpyval.from_json(fp)
    assert _model_class(restored) is _model_class(model)


# -- (c) class readers apply the shared checks --------------------------------


@pytest.mark.parametrize(
    "name", [n for n in CASES if n not in ("never_occurs", "instantly_occurs")]
)
def test_class_from_dict_names_a_missing_entry(name, models):
    model = models[name]
    reader = _model_class(model)
    full = json.loads(json.dumps(model.to_dict(), allow_nan=False))
    named = 0
    for key in full:
        d = {k: v for k, v in full.items() if k != key}
        try:
            reader.from_dict(d)
        except ValueError as err:
            # a missing entry used to escape as a bare KeyError
            if f"no {key!r} entry, which {reader.__name__}" in str(err):
                named += 1
                # (without its dispatch key the package reader cannot
                # tell which class to use, and says that instead)
                if key not in ("parameterization", "model"):
                    with pytest.raises(ValueError, match=f"no '{key}' entry"):
                        surpyval.from_dict(d)
    # Any other exception than ValueError fails the test above; and at
    # least one entry is reported by name.
    assert named > 0


@pytest.mark.parametrize("name", CASES)
def test_class_from_dict_refuses_a_newer_or_bad_schema(name, models):
    model = models[name]
    reader = _model_class(model)
    d = model.to_dict()
    for bad, message in ((99, "schema version 99"), ("2", "integer")):
        d["schema"] = bad
        with pytest.raises(ValueError, match=message):
            reader.from_dict(d)


@pytest.mark.parametrize("name", ["parametric", "kaplan_meier", "cox"])
def test_class_from_dict_refuses_a_non_dict(name, models):
    with pytest.raises(ValueError, match="dict"):
        _model_class(models[name]).from_dict(["not", "a", "dict"])


def test_parametric_from_dict_checks_parameter_bounds(models):
    d = models["parametric"].to_dict()
    d["params"] = [-1.0, 2.0]
    # Parametric.from_dict used to restore a negative Weibull scale
    with pytest.raises(ValueError, match="outside its bounds"):
        Parametric.from_dict(d)
    d = models["parametric_lfp_zi"].to_dict()
    d["p"] = 1.5
    with pytest.raises(ValueError, match="proportion"):
        Parametric.from_dict(d)


@pytest.mark.parametrize(
    "record, message",
    [
        ({"inf": ["/H/0"]}, "not null"),
        ({"inf": ["/nothing"]}, "does not exist"),
        ({"huge": ["/H/0"]}, "unknown kind"),
        ({"inf": "/H/0"}, "not a list"),
        ({"inf": ["H/0"]}, "not a JSON Pointer"),
    ],
)
def test_corrupt_non_finite_record_is_refused(record, message, models):
    d = models["kaplan_meier"].to_dict()
    d[NON_FINITE_KEY] = record
    with pytest.raises(ValueError, match=message):
        NonParametric.from_dict(d)


def test_degenerate_class_readers_check_the_schema():
    for cls in (NeverOccurs, InstantlyOccurs):
        d = cls.to_dict()
        d["schema"] = 99
        with pytest.raises(ValueError, match="schema version 99"):
            cls.from_dict(d)


# -- the encoding itself ------------------------------------------------------


def test_encode_decode_round_trip_and_pointer_escaping():
    original = {
        "a/b": [1.0, np.inf],
        "c~d": {"e": (np.float64(-np.inf), 2)},
        "arr": np.array([[np.nan, 1.0]]),
        "none": None,
        "n": np.int64(3),
    }
    d = encode_non_finite(dict(original))
    json.dumps(d, allow_nan=False)
    assert d[NON_FINITE_KEY] == {
        "inf": ["/a~1b/1"],
        "-inf": ["/c~0d/e/0"],
        "nan": ["/arr/0/0"],
    }
    assert type(d["n"]) is int and d["none"] is None
    back = decode_non_finite(json.loads(json.dumps(d)))
    assert back["a/b"] == [1.0, np.inf]
    assert back["c~d"]["e"] == [-np.inf, 2]
    assert np.isnan(back["arr"][0][0])
    # an unrecorded null stays None
    assert back["none"] is None
    assert NON_FINITE_KEY not in back


def test_decode_leaves_input_untouched_and_is_idempotent():
    d = encode_non_finite({"x": [np.inf]})
    frozen = json.dumps(d)
    once = decode_non_finite(d)
    assert json.dumps(d) == frozen
    assert decode_non_finite(once) == once == {"x": [np.inf]}
    plain = {"x": [1.0]}
    assert decode_non_finite(plain) is plain


def test_surpyval_data_to_json_is_strict_and_round_trips(tmp_path):
    data = SurpyvalData(x=[1.0, 2.0, 3.0], c=[0, 1, 0], tl=[0.5, -np.inf, 0])
    text = data.to_json()
    parsed = _strict_loads(text)
    assert parsed[NON_FINITE_KEY] == {
        "inf": ["/t/0/1", "/t/1/1", "/t/2/1"],
        "-inf": ["/t/1/0"],
    }
    restored = SurpyvalData.from_json(text)
    np.testing.assert_array_equal(restored.t, data.t)
    # files written before still load
    legacy = json.dumps(decode_non_finite(parsed))
    assert "Infinity" in legacy
    np.testing.assert_array_equal(SurpyvalData.from_json(legacy).t, data.t)
    fp = tmp_path / "data.json"
    data.to_json(fp)
    _strict_loads(fp.read_text())
    np.testing.assert_array_equal(SurpyvalData.from_json(fp).t, data.t)
