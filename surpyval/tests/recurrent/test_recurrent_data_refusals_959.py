"""#959: ``fit_from_recurrent_data`` refuses what ``fit(data)`` refuses --
covariates a fitter has no ``Z`` for, event types it has no ``e`` for --
rather than dropping the covariates or pooling the failure modes."""

import inspect
import warnings

import numpy as np
import pytest

import surpyval.recurrent as R
from surpyval.utils.recurrent_utils import handle_xicn

x = np.array([1.0, 2.0, 3.0, 5.0, 6.0, 8.0, 10.0, 10.0, 4.0, 9.0, 12.0])
i = np.array([1, 1, 1, 1, 2, 2, 2, 1, 3, 3, 3])
c = np.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1])
e = np.array(
    ["a", "b", "a", "b", "a", "b", None, None, "a", "b", None], dtype=object
)
Z = np.array([[0.0]] * 4 + [[1.0]] * 3 + [[0.0]] + [[1.0]] * 3)

DATA = {
    "covariates": handle_xicn(x, i, c, Z=Z),
    "event types": handle_xicn(x, i, c, e=e),
    "both": handle_xicn(x, i, c, Z=Z, e=e),
}

FITTERS = sorted(
    name
    for name in dir(R)
    if not name.startswith("_")
    and hasattr(getattr(R, name), "fit_from_recurrent_data")
    and hasattr(getattr(R, name), "fit")
)


def _outcome(call):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = call()
        except (ValueError, TypeError) as err:
            return type(err), str(err)
    # The cause-specific models hold one model per type, and no params.
    params = getattr(model, "params", None)
    return "fitted", None if params is None else np.asarray(params, float)


def test_every_recurrent_fitter_is_checked():
    assert len(FITTERS) == 13


@pytest.mark.parametrize("kind", list(DATA))
@pytest.mark.parametrize("name", FITTERS)
def test_both_ways_of_fitting_take_the_same_data(name, kind):
    fitter = getattr(R, name)
    data = DATA[kind]
    via_fit = _outcome(lambda: fitter.fit(data))
    via_data = _outcome(lambda: fitter.fit_from_recurrent_data(data))
    assert via_fit[0] == via_data[0], (via_fit, via_data)
    if via_fit[0] == "fitted":
        if via_fit[1] is not None:
            np.testing.assert_allclose(via_fit[1], via_data[1])
    else:
        assert via_fit[1] == via_data[1]


@pytest.mark.parametrize("name", FITTERS)
def test_nothing_is_dropped(name):
    # A fit of data holding covariates or event types the fitter has no
    # argument for is refused; one it has arguments for fits.
    fitter = getattr(R, name)
    takes = set(inspect.signature(fitter.fit).parameters)
    for kind, data in DATA.items():
        holds = {"covariates": {"Z"}, "event types": {"e"}}.get(
            kind, {"Z", "e"}
        )
        outcome = _outcome(lambda: fitter.fit_from_recurrent_data(data))
        if not holds <= takes:
            assert outcome[0] in (ValueError, TypeError), (kind, outcome)


def test_the_issue_example():
    with pytest.raises(ValueError, match="takes no covariates") as err:
        R.CrowAMSAA.fit_from_recurrent_data(DATA["covariates"])
    assert "data.without('covariates')" in str(err.value)
    with pytest.raises(ValueError, match="takes no event types") as err:
        R.CrowAMSAA.fit_from_recurrent_data(DATA["event types"])
    assert "data.for_event_type(t)" in str(err.value)
    # Left out on purpose, it fits.
    plain = R.CrowAMSAA.fit_from_recurrent_data(
        DATA["both"].without("covariates", "event_types")
    )
    assert np.isfinite(plain.params).all()


def test_cause_specific_mcf_does_not_drop_covariates():
    with pytest.raises(ValueError, match="takes no covariates"):
        R.CauseSpecificMCF.fit_from_recurrent_data(DATA["both"])


def test_proportional_intensity_given_event_types_and_no_covariates():
    # Was a raw AttributeError ('NoneType' object has no attribute 'shape').
    for fitter in (R.ProportionalIntensityHPP, R.ProportionalIntensityNHPP):
        with pytest.raises(TypeError, match="needs covariates"):
            fitter.fit_from_recurrent_data(DATA["event types"])


def test_proportional_intensity_starts_from_its_baseline_fit():
    # The start is the baseline's fit without the covariates, which the
    # baseline's own fit now refuses: the fit is the same as before.
    rng = np.random.default_rng(0)
    xs, ids, Zs = [], [], []
    for u in range(200):
        t = np.cumsum(rng.exponential(10 / (1 + u % 2), 8))
        xs += list(t)
        ids += [u] * 8
        Zs += [[u % 2]] * 8
    xs, ids = np.array(xs), np.array(ids)
    Zs = np.array(Zs, dtype=float)
    data = handle_xicn(xs, ids, np.zeros_like(xs), Z=Zs)
    start = R.ProportionalIntensityNHPP._baseline_start(data, R.Duane)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = R.Duane.fit(xs, ids, np.zeros_like(xs))
    np.testing.assert_allclose(start, base.params)
