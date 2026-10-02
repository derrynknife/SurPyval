"""The package-level readers ``surpyval.from_dict`` / ``surpyval.from_json``.

Every serialisable model writes either a ``"model"`` class tag or a
``"parameterization"`` marker into its ``to_dict``; the package-level
readers dispatch on those, so a caller can restore any model without
knowing which class wrote it. These tests pin:

- registry integrity: every registered tag resolves to a class of that
  name with a ``from_dict``;
- dispatch: a round-trip through ``surpyval.from_dict`` restores the
  right class and reproduces predictions, for a representative model
  of every dictionary shape (parametric, non-parametric,
  parametric-regression, and the tagged families);
- the file reader ``surpyval.from_json``;
- clear errors for unrecognisable input.
"""

import json

import numpy as np
import pytest

import surpyval
from surpyval import CoxPH, KaplanMeier, MixtureModel, Weibull, WeibullPH
from surpyval.serialisation import (
    _PARAMETERIZATIONS,
    _TAGGED_MODELS,
    _resolve,
)
from surpyval.tests._helpers import json_round_trip


def _regression_data(seed=0, n=60):
    rng = np.random.default_rng(seed)
    Z = np.column_stack(
        [rng.integers(0, 2, n).astype(float), rng.normal(0, 1, n)]
    )
    x = 10 * np.exp(-0.5 * Z[:, 0]) * rng.weibull(2.0, n)
    c = (rng.random(n) < 0.2).astype(int)
    return x, Z, c


# -- registry integrity ------------------------------------------------------


def test_every_tag_resolves_to_its_class():
    for tag, module in _TAGGED_MODELS.items():
        resolved = _resolve(module, tag)
        # Some fitters follow surpyval's singleton pattern, binding the
        # module-level name to an instance of the class; from_dict is a
        # classmethod so both dispatch identically.
        name = getattr(resolved, "__name__", type(resolved).__name__)
        assert name == tag
        assert callable(resolved.from_dict)


def test_every_parameterization_resolves():
    for module, name in _PARAMETERIZATIONS.values():
        cls = _resolve(module, name)
        assert cls.__name__ == name
        assert callable(cls.from_dict)


# -- the three untagged core shapes ------------------------------------------


def test_parametric_dispatch():
    model = Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "Parametric"
    assert restored.dist.name == "Weibull"
    t = np.array([2.0, 5.0, 9.0])
    assert np.allclose(model.sf(t), restored.sf(t))


def test_non_parametric_dispatch():
    model = KaplanMeier.fit([3.0, 4.0, 5.0, 6.0, 7.0], [0, 1, 0, 0, 1])
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "NonParametric"
    assert restored.model == "Kaplan-Meier"
    assert np.allclose(model.R, restored.R)


def test_parametric_regression_dispatch():
    x, Z, c = _regression_data()
    model = WeibullPH.fit(x=x, Z=Z, c=c)
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "ParametricRegressionModel"
    t = np.array([2.0, 6.0])
    z = np.array([1.0, 0.5])
    assert np.allclose(model.sf(t, Z=z), restored.sf(t, Z=z))


def _formula_df(seed=0, n=200):
    import pandas as pd

    rng = np.random.default_rng(seed)
    sex = rng.choice(["M", "F"], n)
    age = rng.uniform(30, 70, n)
    x = 10 * np.exp(-0.4 * (sex == "M")) * rng.weibull(2.0, n)
    return pd.DataFrame(
        {
            "time": x,
            "c": rng.integers(0, 2, n).astype(int),
            "age": age,
            "sex": sex,
        }
    )


@pytest.mark.parametrize(
    "formula", ["age + sex", "age + sex + age:sex", "np.log(age) + sex"]
)
def test_formula_regression_round_trips_with_raw_covariates(formula):
    # #244: a formula fit with a categorical must round-trip so the restored
    # model expands *raw* covariates (sex -> sex[F], sex[M]) identically.
    import pandas as pd

    df = _formula_df()
    model = WeibullPH.fit_from_df(df, x_col="time", c_col="c", formula=formula)
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))

    raw_Z = pd.DataFrame({"age": [40.0, 55.0], "sex": ["M", "F"]})
    t = np.array([5.0, 12.0])
    assert np.allclose(model.sf(t, raw_Z), restored.sf(t, raw_Z))


def test_formula_cox_round_trips_with_raw_covariates():
    import pandas as pd

    df = _formula_df(seed=2)
    model = CoxPH.fit_from_df(df, x_col="time", c_col="c", formula="age + sex")
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))

    raw_Z = pd.DataFrame({"age": [40.0, 55.0], "sex": ["M", "F"]})
    t = np.array([5.0, 12.0])
    assert np.allclose(model.sf(t, raw_Z), restored.sf(t, raw_Z))


def test_stateful_transform_formula_round_trips():
    import pandas as pd

    df = _formula_df(seed=3)
    model = WeibullPH.fit_from_df(
        df, x_col="time", c_col="c", formula="scale(age) + sex"
    )
    # scale() keeps fitted statistics (the training mean and sd); they are
    # stored with the formula, so the restored model scales new data by
    # the *training* statistics, as the original does.
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    raw_Z = pd.DataFrame({"age": [40.0, 55.0], "sex": ["M", "F"]})
    t = np.array([5.0, 12.0])
    assert np.allclose(model.sf(t, raw_Z), restored.sf(t, raw_Z))


# -- tagged families (one representative per family) --------------------------


def test_semi_parametric_dispatch():
    x, Z, c = _regression_data(seed=1)
    model = CoxPH.fit(x=x, Z=Z, c=c)
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "SemiParametricRegressionModel"
    t = np.array([2.0, 6.0])
    z = np.array([1.0, 0.5])
    assert np.allclose(model.sf(t, Z=z), restored.sf(t, Z=z))


def test_mixture_model_dispatch():
    x = np.concatenate([Weibull.random(50, 8, 3), Weibull.random(50, 40, 4)])
    model = MixtureModel(dist=Weibull, m=2)
    model.fit(x=x)
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "MixtureModel"
    t = np.array([5.0, 20.0])
    assert np.allclose(model.sf(t), restored.sf(t))


def test_parametric_recurrence_dispatch():
    from surpyval.recurrent import CrowAMSAA

    x = np.array([10.0, 25.0, 45.0, 70.0, 100.0, 135.0, 175.0])
    model = CrowAMSAA.fit(x)
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "ParametricRecurrenceModel"
    t = np.array([50.0, 150.0])
    assert np.allclose(model.cif(t), restored.cif(t))


def test_mcf_dispatch():
    from surpyval.recurrent import NonParametricCounting

    x = np.array([5.0, 12.0, 20.0, 8.0, 15.0, 25.0])
    i = np.array([1, 1, 1, 2, 2, 2])
    c = np.array([0, 0, 1, 0, 0, 1])
    model = NonParametricCounting.fit(x, i, c)
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "NonParametricCounting"
    assert np.allclose(model.mcf_hat, restored.mcf_hat)


def test_renewal_dispatch():
    from surpyval.recurrent import GeneralizedRenewal

    model = GeneralizedRenewal.fit_from_parameters(
        [50.0, 2.0], 0.3, kijima="i", dist=Weibull
    )
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "RenewalModel"
    assert np.isclose(model.restoration, restored.restoration)
    assert np.allclose(model.model.params, restored.model.params)


def test_competing_risks_dispatch():
    from surpyval.univariate.competing_risks import CompetingRisks

    rng = np.random.default_rng(3)
    n = 60
    x = np.abs(rng.weibull(1.3, n) * 15) + 0.2
    e = rng.choice([1, 2], n)
    model = CompetingRisks.fit(x, e)
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "CompetingRisks"


def test_process_model_dispatch():
    from surpyval.degradation import WienerProcess

    rng = np.random.default_rng(2)
    xs, ys, ii = [], [], []
    for u in range(10):
        t = np.arange(0, 20, 2.0)
        xs.append(t)
        ys.append(np.cumsum(rng.normal(1.0, 1.0, t.size)))
        ii.append(np.full(t.size, u))
    model = WienerProcess.fit(
        np.concatenate(xs),
        np.concatenate(ys),
        np.concatenate(ii),
        threshold=20.0,
    )
    restored = surpyval.from_dict(json_round_trip(model.to_dict()))
    assert type(restored).__name__ == "WienerProcessModel"
    t = np.array([5.0, 15.0])
    assert np.allclose(model.sf(t), restored.sf(t))


# -- the file reader ----------------------------------------------------------


def test_from_json_file(tmp_path):
    model = Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    fp = tmp_path / "weibull.json"
    model.to_json(fp)
    restored = surpyval.from_json(fp)
    assert restored.dist.name == "Weibull"
    t = np.array([2.0, 5.0, 9.0])
    assert np.allclose(model.sf(t), restored.sf(t))


def test_from_json_file_tagged(tmp_path):
    x, Z, c = _regression_data(seed=4)
    model = CoxPH.fit(x=x, Z=Z, c=c)
    fp = tmp_path / "cox.json"
    model.to_json(fp)
    restored = surpyval.from_json(fp)
    assert type(restored).__name__ == "SemiParametricRegressionModel"


# -- errors -------------------------------------------------------------------


def test_schema_version_is_the_oldest_that_reads_the_document():
    from surpyval.serialisation import SCHEMA_VERSION

    # no non-finite value: the schema-1 layout, which v0.20 reads
    d = Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0]).to_dict()
    assert "non_finite" not in d
    assert d["schema"] == 1
    # an infinite cumulative hazard is written as null with a record, which
    # a schema-1 reader would misread, so the document is schema 2
    d = KaplanMeier.fit([3.0, 4.0, 5.0, 6.0, 7.0]).to_dict()
    assert "non_finite" in d
    assert d["schema"] == SCHEMA_VERSION == 2
    # a nested record makes the enclosing document schema 2 as well
    from surpyval.multivariate import Clayton

    rng = np.random.default_rng(0)
    x = rng.weibull(2.0, size=(50, 2)) * 10
    km_margin = Clayton.fit(x, margins=[KaplanMeier, Weibull]).to_dict()
    assert km_margin["schema"] == 2
    plain = Clayton.fit(x, margins=[Weibull, Weibull]).to_dict()
    assert plain["schema"] == 1


def test_unversioned_documents_still_load():
    # documents written before schema versioning have no "schema" key
    model = Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0])
    d = {k: v for k, v in model.to_dict().items() if k != "schema"}
    restored = surpyval.from_dict(d)
    t = np.array([2.0, 5.0])
    assert np.allclose(model.sf(t), restored.sf(t))


def test_newer_schema_is_refused():
    d = Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0]).to_dict()
    d["schema"] = 99
    with pytest.raises(ValueError, match="schema version 99"):
        surpyval.from_dict(d)


def test_from_dict_rejects_non_dict():
    with pytest.raises(ValueError, match="dict"):
        surpyval.from_dict("not a dict")


def test_from_dict_rejects_empty_dict():
    with pytest.raises(ValueError, match="recognisable"):
        surpyval.from_dict({})


def test_from_dict_rejects_unknown_tag():
    with pytest.raises(ValueError, match="NotAModel"):
        surpyval.from_dict({"model": "NotAModel"})


def test_from_dict_rejects_unknown_parameterization():
    with pytest.raises(ValueError, match="bayesian"):
        surpyval.from_dict({"parameterization": "bayesian"})


# The schema-1 promise, checked against a real schema-1 reader: point
# SURPYVAL_SCHEMA1_READER at a checkout of a release that reads schema 1
# (e.g. ``git worktree add /tmp/surpyval-v0.20 v0.20.0``) to run it.
_SCHEMA1_READER = __import__("os").environ.get("SURPYVAL_SCHEMA1_READER")


@pytest.mark.skipif(
    not _SCHEMA1_READER, reason="SURPYVAL_SCHEMA1_READER not set"
)
def test_schema_1_documents_read_identically_in_a_schema_1_release(
    tmp_path,
):
    import subprocess
    import sys

    from surpyval.recurrent import CrowAMSAA

    rng = np.random.default_rng(0)
    x = rng.weibull(2.0, 60) * 10
    Z = rng.normal(size=(60, 1))
    models = {
        "weibull": Weibull.fit(x),
        "weibull_fixed": Weibull.fit(x, fixed={"beta": 2.0}),
        "weibull_offset": Weibull.fit(x + 5, offset=True),
        "weibullph": WeibullPH.fit(x=x, Z=Z),
        "cox": CoxPH.fit(x=x, Z=Z),
        "crow": CrowAMSAA.fit(
            np.cumsum(rng.exponential(1, 20)), c=[0] * 19 + [1]
        ),
    }
    docs = {k: m.to_dict() for k, m in models.items()}
    assert all(d["schema"] == 1 for d in docs.values())
    path = tmp_path / "docs.json"
    path.write_text(json.dumps(docs, allow_nan=False))
    reader = (
        "import json, sys, numpy as np\n"
        "sys.path.insert(0, sys.argv[1])\n"
        "import surpyval\n"
        "docs = json.load(open(sys.argv[2]))\n"
        "q = np.array([3.0, 8.0, 15.0]); Z = np.zeros((3, 1))\n"
        "out = {}\n"
        "for k, d in docs.items():\n"
        "    m = surpyval.from_dict(d)\n"
        "    if k in ('weibullph', 'cox'): v = m.sf(q, Z)\n"
        "    elif k == 'crow': v = m.cif(q)\n"
        "    else: v = m.sf(q)\n"
        "    out[k] = np.asarray(v, float).ravel().tolist()\n"
        "print(json.dumps(out))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", reader, _SCHEMA1_READER, str(path)],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
    )
    old = json.loads(result.stdout.strip().splitlines()[-1])
    q = np.array([3.0, 8.0, 15.0])
    Z0 = np.zeros((3, 1))
    for k, m in models.items():
        if k in ("weibullph", "cox"):
            now = m.sf(q, Z0)
        elif k == "crow":
            now = m.cif(q)
        else:
            now = m.sf(q)
        np.testing.assert_allclose(old[k], np.ravel(now), rtol=1e-9)


# ---------------------------------------------------------------------------
# Missing keys are named, the schema is checked, and invalid
# parameters are refused, by the package readers and by the
# class readers alike.
# ---------------------------------------------------------------------------


@pytest.fixture
def weibull_dict():
    return Weibull.fit([3.0, 4.0, 5.0, 6.0, 7.0]).to_dict()


@pytest.mark.parametrize("key", ["distribution", "params", "how"])
def test_missing_key_names_the_key(weibull_dict, key):
    # Used to be a bare KeyError from inside Parametric.from_dict.
    del weibull_dict[key]
    with pytest.raises(ValueError, match=f"no '{key}' entry"):
        surpyval.from_dict(weibull_dict)


def test_missing_key_in_a_tagged_model():
    Z = np.array([[0.0], [1.0], [0.0], [1.0], [1.0]])
    d = surpyval.CoxPH.fit([1, 2, 3, 4, 5.0], Z).to_dict()
    del d["beta"]
    with pytest.raises(ValueError, match="no 'beta' entry"):
        surpyval.from_dict(d)
    d = KaplanMeier.fit([1, 2, 3, 4]).to_dict()
    del d["model"]
    with pytest.raises(ValueError, match="no 'model' entry"):
        surpyval.from_dict(d)


@pytest.mark.parametrize("schema", ["2", 2.0, 1.0, "1", True, None, -1])
def test_schema_must_be_a_non_negative_integer(weibull_dict, schema):
    # "2" and 2.0 used to slip past the version check.
    weibull_dict["schema"] = schema
    with pytest.raises(ValueError, match="schema"):
        surpyval.from_dict(weibull_dict)


def test_integer_schemas_still_read(weibull_dict):
    for schema in (0, 1, np.int64(1)):
        weibull_dict["schema"] = schema
        surpyval.from_dict(weibull_dict)
    del weibull_dict["schema"]
    surpyval.from_dict(weibull_dict)


def test_newer_schema_with_unknown_model_asks_for_upgrade():
    with pytest.raises(ValueError, match="Upgrade surpyval"):
        surpyval.from_dict({"model": "FromTheFuture", "schema": 99})


@pytest.mark.parametrize(
    "change, match",
    [
        (dict(params=[-5.0, 2.0]), "alpha=-5.0 .* outside its bounds"),
        (dict(params=[5.0, -2.0]), "beta=-2.0 .* outside its bounds"),
        (dict(params=[np.nan, 2.0]), "NaN"),
        (dict(lfp=True, p=1.5), "'p'=1.5 is a proportion"),
        (dict(zi=True, f0=-0.1), "'f0'=-0.1 is a proportion"),
    ],
)
def test_invalid_parameters_are_refused(weibull_dict, change, match):
    weibull_dict.update(change)
    with pytest.raises(ValueError, match=match):
        surpyval.from_dict(weibull_dict)


def test_class_from_json_gets_the_same_checks(tmp_path, weibull_dict):
    path = tmp_path / "w.json"
    weibull_dict["params"] = [-5.0, 2.0]
    path.write_text(json.dumps(weibull_dict))
    with pytest.raises(ValueError, match="outside its bounds"):
        Weibull.fit([1, 2, 3.0]).from_json(path)
    weibull_dict["params"] = [5.0, 2.0]
    weibull_dict["schema"] = "1"
    path.write_text(json.dumps(weibull_dict))
    with pytest.raises(ValueError, match="schema"):
        Weibull.fit([1, 2, 3.0]).from_json(path)
