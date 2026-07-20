"""Serialisation of the fitted renewal / imperfect-repair models.

``RenewalModel`` covers the generalized-renewal (Kijima-I/II), G1 renewal, ARA
and ARI families. These processes have no closed-form intensity -- their mean
cumulative function comes from a sampler closure -- so serialisation stores the
family, the underlying distribution (by name) and its parameters, the
restoration parameter and the family option (``kijima_type`` / memory ``m``),
and the family's fitter rebuilds the sampler on load. The reloaded model
reproduces the (seeded) simulated MCF exactly.
"""

import json

import numpy as np
import pytest

from surpyval import Weibull
from surpyval.recurrent import (
    ARA,
    ARI,
    CrowAMSAA,
    GeneralizedOneRenewal,
    GeneralizedRenewal,
)
from surpyval.recurrent.renewal.renewal_model import RenewalModel


def _rt(d):
    return json.loads(json.dumps(d))


def _make(name):
    if name == "GR-i":
        return GeneralizedRenewal.fit_from_parameters(
            [50.0, 2.0], 0.3, kijima="i", dist=Weibull
        )
    if name == "GR-ii":
        return GeneralizedRenewal.fit_from_parameters(
            [50.0, 2.0], 0.3, kijima="ii", dist=Weibull
        )
    if name == "G1R":
        return GeneralizedOneRenewal.fit_from_parameters(
            [50.0, 2.0], 0.2, dist=Weibull
        )
    if name == "ARA":
        return ARA.fit_from_parameters([50.0, 2.0], 0.4, m=2, dist=Weibull)
    if name == "ARI":
        return ARI.fit_from_parameters([60.0, 2.0], 0.3, m=1, dist=CrowAMSAA)
    raise ValueError(name)


@pytest.mark.parametrize("name", ["GR-i", "GR-ii", "G1R", "ARA", "ARI"])
def test_renewal_round_trip(name):
    model = _make(name)
    restored = RenewalModel.from_dict(_rt(model.to_dict()))
    # the restoration parameter and family option survive
    assert np.isclose(model.restoration, restored.restoration)
    assert np.allclose(model.model.params, restored.model.params)
    assert model.model.dist.name == restored.model.dist.name
    # the seeded simulated MCF reproduces exactly (the sampler was rebuilt)
    t = np.array([10.0, 50.0, 100.0])
    assert np.allclose(
        model.mcf(t, items=300, seed=7),
        restored.mcf(t, items=300, seed=7),
    )


def test_renewal_kijima_type_preserved():
    for kijima in ("i", "ii"):
        model = GeneralizedRenewal.fit_from_parameters(
            [50.0, 2.0], 0.3, kijima=kijima, dist=Weibull
        )
        restored = RenewalModel.from_dict(model.to_dict())
        assert restored.kijima_type == kijima


def test_renewal_memory_preserved():
    model = ARA.fit_from_parameters([50.0, 2.0], 0.4, m=3, dist=Weibull)
    restored = RenewalModel.from_dict(model.to_dict())
    assert restored.m == 3


def test_renewal_json_file(tmp_path):
    model = _make("ARA")
    fp = tmp_path / "renewal.json"
    model.to_json(fp)
    restored = RenewalModel.from_json(fp)
    t = np.array([20.0, 80.0])
    assert np.allclose(
        model.mcf(t, items=200, seed=3),
        restored.mcf(t, items=200, seed=3),
    )


def test_renewal_rejects_wrong_dict():
    with pytest.raises(ValueError, match="RenewalModel"):
        RenewalModel.from_dict({"model": "Other"})


def test_renewal_rejects_unknown_family():
    d = _make("ARA").to_dict()
    d["family"] = "Nope"
    with pytest.raises(ValueError, match="Unknown renewal family"):
        RenewalModel.from_dict(d)


def test_renewal_rejects_non_distribution_attribute():
    # The stored distribution name must resolve to a genuine
    # distribution fitter: an untrusted dict must not be able to pull
    # arbitrary surpyval attributes through the lookup -- issue #206.
    model = GeneralizedRenewal.fit_from_parameters(
        [50.0, 2.0], 0.3, kijima="i", dist=Weibull
    )
    d = model.to_dict()
    d["dist"] = "CoxPH"  # real surpyval attribute, not a ParametricFitter
    with pytest.raises(ValueError, match="Unknown distribution"):
        RenewalModel.from_dict(d)
