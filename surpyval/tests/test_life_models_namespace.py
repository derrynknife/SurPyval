"""The life models live in ``surpyval.life_models``; the names they had at
the top level until v0.22 still work, with a DeprecationWarning, until
v0.23."""

import warnings

import pytest

import surpyval as sp
from surpyval import life_models
from surpyval.univariate.regression import accelerated_life

MOVED = {
    "DualExponential": "DualExponential",
    "DualPower": "DualPower",
    "ExponentialLifeModel": "Exponential",
    "Eyring": "Eyring",
    "InverseExponential": "InverseExponential",
    "InverseEyring": "InverseEyring",
    "InversePower": "InversePower",
    "LifeModel": "LifeModel",
    "Linear": "Linear",
    "Power": "Power",
    "PowerExponential": "PowerExponential",
}


def test_every_life_model_is_in_the_namespace():
    for name in accelerated_life.LIFE_MODELS.values():
        assert getattr(life_models, name.name) is name
    # The exponential life model has its plain name there; at the top level
    # Exponential is the distribution.
    assert life_models.Exponential is accelerated_life.ExponentialLifeModel
    assert sp.Exponential is not life_models.Exponential
    assert set(life_models.__all__) == set(MOVED.values()) | {
        "GeneralLogLinear"
    }


@pytest.mark.parametrize("old, new", sorted(MOVED.items()))
def test_old_top_level_name_warns_and_still_works(old, new):
    with pytest.warns(DeprecationWarning, match=f"life_models.{new}") as w:
        value = getattr(sp, old)
    assert value is getattr(life_models, new)
    assert w[0].filename == __file__
    assert old not in dir(sp)


def test_the_namespace_does_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitter = sp.AcceleratedLife(sp.Weibull, sp.life_models.Power)
    assert "life_models" in dir(sp)
    assert fitter is not None


def test_general_log_linear_is_not_at_the_top_level():
    # New in 0.22, so never at the top level in a release: no shim
    with pytest.raises(AttributeError):
        sp.GeneralLogLinear
