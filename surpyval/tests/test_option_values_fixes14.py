"""An unknown option value raises one ValueError, naming the argument and
the values it takes, everywhere; 'R' and 'F' are accepted for 'sf' and
'ff' by every ``cb`` (#416, principles 2 and 21)."""

import numpy as np
import pytest

import surpyval as sp
from surpyval.recurrent import CauseSpecificMCF, NonParametricCounting
from surpyval.tests.conformance.registry import (
    CASE_BY_NAME,
    fitted,
    recurrent_data,
)

X = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
C = np.array([0, 0, 1, 0, 0, 1, 0, 0])


# -- recurrent MCF bounds ---------------------------------------------------
def _mcf_models():
    npc = NonParametricCounting.fit(**recurrent_data())
    cs = CauseSpecificMCF.fit(**recurrent_data(with_e=True))
    return [
        (npc.mcf_cb, {}),
        (cs.mcf_cb, {"event": "a"}),
    ]


@pytest.mark.parametrize("bound_type", ["exp", "normal"])
@pytest.mark.parametrize("interp", ["step", "linear"])
def test_mcf_cb_refuses_an_unknown_bound(bound_type, interp):
    # bound='both' raised UnboundLocalError ('stat').
    for mcf_cb, kw in _mcf_models():
        with pytest.raises(ValueError, match="'bound' must be one of"):
            mcf_cb(
                [10.0, 30.0],
                bound="both",
                interp=interp,
                bound_type=bound_type,
                **kw,
            )


def test_mcf_cb_refuses_an_unknown_interp():
    # An unknown interp returned the bounds at every event time, whatever
    # the query.
    for mcf_cb, kw in _mcf_models():
        with pytest.raises(ValueError, match=r"'interp' must be one of"):
            mcf_cb([10.0, 30.0], interp="cubic", **kw)
        with pytest.raises(ValueError, match="'bound_type' must be one of"):
            mcf_cb([10.0, 30.0], bound_type="log", **kw)


def test_mcf_refuses_an_unknown_interp_outside_the_support():
    model = NonParametricCounting.fit(**recurrent_data())
    model.set_support(-1.0, 100.0)
    with pytest.raises(ValueError, match="'interp' must be one of"):
        model.mcf([200.0], interp="bogus")
    with pytest.raises(ValueError, match="'interp' must be one of"):
        model.mcf_cb([200.0], interp="bogus")


# -- univariate non-parametric interp ---------------------------------------
_FITTERS = ["KaplanMeier", "NelsonAalen", "FlemingHarrington", "Turnbull"]


@pytest.mark.parametrize("fitter", _FITTERS)
def test_non_parametric_refuses_an_unknown_interp(fitter):
    # It reached scipy's interp1d, which raised NotImplementedError.
    model = getattr(sp, fitter).fit(X, c=C)
    calls = {
        name: getattr(model, name)
        for name in ("sf", "ff", "Hf", "hf", "df", "cb", "R_cb")
    }
    for name, f in calls.items():
        with pytest.raises(ValueError, match="'interp' must be one of"):
            f([2.5, 4.5], interp="bogus")
    with pytest.raises(ValueError, match="'interp' must be one of"):
        model.plot(interp="bogus")
    # With a support set and every query outside the data, too.
    model.set_support(0.0, 20.0)
    for name, f in calls.items():
        with pytest.raises(ValueError, match="'interp' must be one of"):
            f([15.0], interp="bogus")


@pytest.mark.parametrize("fitter", _FITTERS)
def test_non_parametric_accepts_its_documented_interp_kinds(fitter):
    model = getattr(sp, fitter).fit(X, c=C)
    kinds = ("linear", "cubic", "nearest", "nearest-up", "zero")
    kinds += ("slinear", "quadratic", "previous", "next")
    for kind in kinds:
        assert np.isfinite(model.sf([2.5, 4.5], interp=kind)).all()


# -- cause-specific Cox ------------------------------------------------------
@pytest.mark.parametrize("interp", ["linear", "bogus"])
def test_cause_specific_cox_takes_only_step(interp):
    # Any interp, even 'bogus', was accepted and ignored: 'linear' gave the
    # step curve.
    case = CASE_BY_NAME["CompetingRisksProportionalHazards[Cox]"]
    model = fitted(case)
    Z = case.Z[0]
    for name in ("sf", "ff", "Hf", "hf", "df"):
        with pytest.raises(ValueError, match=r"'interp' must be 'step'"):
            getattr(model, name)([5.0], Z, interp=interp)
    step = model.sf([5.0], Z, interp="step")
    np.testing.assert_array_equal(step, model.sf([5.0], Z))


# -- on= aliases -----------------------------------------------------------
def test_destructive_degradation_cb_takes_the_on_aliases():
    # on='R' raised ValueError; every other cb takes 'R' and 'F'.
    model = fitted(CASE_BY_NAME["DestructiveDegradation"])
    x = [10.0, 30.0]
    for alias, name in (("R", "sf"), ("F", "ff")):
        got = model.cb(x, on=alias, n_boot=20, random_state=1)
        want = model.cb(x, on=name, n_boot=20, random_state=1)
        np.testing.assert_array_equal(got, want)
    with pytest.raises(ValueError, match="'on' must be one of"):
        model.cb(x, on="hf", n_boot=20, random_state=1)
    with pytest.raises(ValueError, match="'bound' must be one of"):
        model.cb(x, bound="both", n_boot=20, random_state=1)
