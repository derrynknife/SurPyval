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


def _rt(d):
    """JSON round-trip a dict (proves it is JSON-serialisable)."""
    return json.loads(json.dumps(d))


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
    restored = surpyval.from_dict(_rt(model.to_dict()))
    assert type(restored).__name__ == "Parametric"
    assert restored.dist.name == "Weibull"
    t = np.array([2.0, 5.0, 9.0])
    assert np.allclose(model.sf(t), restored.sf(t))


def test_non_parametric_dispatch():
    model = KaplanMeier.fit([3.0, 4.0, 5.0, 6.0, 7.0], [0, 1, 0, 0, 1])
    restored = surpyval.from_dict(_rt(model.to_dict()))
    assert type(restored).__name__ == "NonParametric"
    assert restored.model == "Kaplan-Meier"
    assert np.allclose(model.R, restored.R)


def test_parametric_regression_dispatch():
    x, Z, c = _regression_data()
    model = WeibullPH.fit(x=x, Z=Z, c=c)
    restored = surpyval.from_dict(_rt(model.to_dict()))
    assert type(restored).__name__ == "ParametricRegressionModel"
    t = np.array([2.0, 6.0])
    z = np.array([1.0, 0.5])
    assert np.allclose(model.sf(t, Z=z), restored.sf(t, Z=z))


# -- tagged families (one representative per family) --------------------------


def test_semi_parametric_dispatch():
    x, Z, c = _regression_data(seed=1)
    model = CoxPH.fit(x=x, Z=Z, c=c)
    restored = surpyval.from_dict(_rt(model.to_dict()))
    assert type(restored).__name__ == "SemiParametricRegressionModel"
    t = np.array([2.0, 6.0])
    z = np.array([1.0, 0.5])
    assert np.allclose(model.sf(t, Z=z), restored.sf(t, Z=z))


def test_mixture_model_dispatch():
    x = np.concatenate([Weibull.random(50, 8, 3), Weibull.random(50, 40, 4)])
    model = MixtureModel(dist=Weibull, m=2)
    model.fit(x=x)
    restored = surpyval.from_dict(_rt(model.to_dict()))
    assert type(restored).__name__ == "MixtureModel"
    t = np.array([5.0, 20.0])
    assert np.allclose(model.sf(t), restored.sf(t))


def test_parametric_recurrence_dispatch():
    from surpyval.recurrent import CrowAMSAA

    x = np.array([10.0, 25.0, 45.0, 70.0, 100.0, 135.0, 175.0])
    model = CrowAMSAA.fit(x)
    restored = surpyval.from_dict(_rt(model.to_dict()))
    assert type(restored).__name__ == "ParametricRecurrenceModel"
    t = np.array([50.0, 150.0])
    assert np.allclose(model.cif(t), restored.cif(t))


def test_mcf_dispatch():
    from surpyval.recurrent import NonParametricCounting

    x = np.array([5.0, 12.0, 20.0, 8.0, 15.0, 25.0])
    i = np.array([1, 1, 1, 2, 2, 2])
    c = np.array([0, 0, 1, 0, 0, 1])
    model = NonParametricCounting.fit(x, i, c)
    restored = surpyval.from_dict(_rt(model.to_dict()))
    assert type(restored).__name__ == "NonParametricCounting"
    assert np.allclose(model.mcf_hat, restored.mcf_hat)


def test_renewal_dispatch():
    from surpyval.recurrent import GeneralizedRenewal

    model = GeneralizedRenewal.fit_from_parameters(
        [50.0, 2.0], 0.3, kijima="i", dist=Weibull
    )
    restored = surpyval.from_dict(_rt(model.to_dict()))
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
    restored = surpyval.from_dict(_rt(model.to_dict()))
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
    restored = surpyval.from_dict(_rt(model.to_dict()))
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
