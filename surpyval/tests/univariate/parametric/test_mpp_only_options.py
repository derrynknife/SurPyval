"""The four options of ``fit`` that only probability plotting reads.

``heuristic``, ``rr``, ``on_d_is_0`` and ``turnbull_estimator`` change a
fit only with ``how="MPP"``. Given (other than at their defaults) with
another method they were ignored in silence, so the call did not do what
it said. Such a call now warns with a ``FutureWarning`` at the user's
line, and its result is unchanged to the last digit; from v0.31 it is
refused.
"""

import warnings

import numpy as np
import pytest

import surpyval as sp
from surpyval.univariate.parametric.optimised_fit import MPP_ONLY_OPTIONS
from surpyval.utils.deprecation import REMOVED_IN

X = np.array([3.1, 4.7, 5.2, 6.8, 7.5, 8.9, 10.4, 11.0, 12.6, 15.3])
C = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 1])

GIVEN = {
    "heuristic": "Blom",
    "rr": "x",
    "on_d_is_0": True,
    "turnbull_estimator": "Kaplan-Meier",
}


def _fit_quietly(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        return sp.Weibull.fit(X, c=C, **kwargs)


@pytest.mark.parametrize("how", ["MLE", "MPS", "MSE"])
@pytest.mark.parametrize("option", sorted(GIVEN))
def test_an_option_another_method_ignores_warns(how, option):
    expected = _fit_quietly(how=how)
    with pytest.warns(FutureWarning) as caught:
        model = sp.Weibull.fit(X, c=C, how=how, **{option: GIVEN[option]})
    (warning,) = (w for w in caught if "read only by" in str(w.message))
    message = str(warning.message)
    assert message.startswith(f"Weibull.fit: {option}=")
    assert f"how='{how}' ignored it" in message
    assert f"From v{REMOVED_IN} this is refused" in message
    assert f"drop {option}, or fit with how='MPP'" in message
    # It points at this line, not inside the package
    assert warning.filename == __file__
    # and the fit is the one without the option, to the last digit
    np.testing.assert_array_equal(model.params, expected.params)


def test_mom_too():
    x = X[C == 0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        expected = sp.Weibull.fit(x, how="MOM")
    with pytest.warns(FutureWarning, match="how='MOM' ignored it"):
        model = sp.Weibull.fit(x, how="MOM", rr="x")
    np.testing.assert_array_equal(model.params, expected.params)


def test_every_ignored_option_is_named_once():
    with pytest.warns(FutureWarning) as caught:
        sp.Weibull.fit(X, c=C, **GIVEN)
    messages = [
        str(w.message) for w in caught if "read only" in str(w.message)
    ]
    assert len(messages) == 1
    assert (
        "drop heuristic, rr, on_d_is_0 and turnbull_estimator" in messages[0]
    )
    assert "are read only by how='MPP'" in messages[0]


def test_the_defaults_given_by_name_do_not_warn():
    _fit_quietly(**MPP_ONLY_OPTIONS)
    # in any case, as the heuristic is read
    _fit_quietly(heuristic="nelson-aalen")


def test_mpp_reads_them_and_does_not_warn():
    plain = _fit_quietly(how="MPP")
    model = _fit_quietly(how="MPP", **GIVEN)
    # the options are used: the fit moves
    assert not np.allclose(model.params, plain.params)


def test_a_fit_from_data_warns_too():
    data = sp.SurpyvalData(x=X, c=C)
    with pytest.warns(FutureWarning, match="rr='x' is read only by"):
        sp.Weibull.fit(data, rr="x")


def test_hypoexponential_warns_too():
    with pytest.warns(FutureWarning, match="Hypoexponential.fit: rr='x'"):
        sp.Hypoexponential.fit(X, stages=2, rr="x")


def test_a_misspelt_heuristic_with_another_method_is_no_longer_silent():
    # MLE never read the heuristic, so a name MPP refuses passed unseen
    with pytest.warns(FutureWarning, match="heuristic='Blomm' is read only"):
        sp.Weibull.fit(X, heuristic="Blomm")
