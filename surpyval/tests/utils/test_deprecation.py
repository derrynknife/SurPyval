"""
``renamed_arguments``: an old argument name keeps working for one release
with a ``DeprecationWarning`` at the caller's line (principle 21, #422).
"""

import warnings

import pytest

from surpyval.utils.deprecation import REMOVED_IN, renamed_arguments
from surpyval.utils.shapes import keeps_query_shape


@renamed_arguments(B="n_boot", confidence=("alpha_ci", lambda c: 1 - c))
def _boot(x, n_boot=1000, alpha_ci=0.05):
    return x, n_boot, alpha_ci


class _Model:
    @renamed_arguments(t="x")
    @keeps_query_shape
    def sf(self, x):
        return x * 0 + 1.0

    @classmethod
    @renamed_arguments(seed="random_state")
    def draw(cls, size, random_state=None):
        return size, random_state


def test_new_names_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert _boot(1, n_boot=5, alpha_ci=0.1) == (1, 5, 0.1)


def test_old_name_is_mapped_and_warns_at_the_caller():
    with pytest.warns(DeprecationWarning, match="'B' is deprecated") as rec:
        assert _boot(1, B=5) == (1, 5, 0.05)
    assert rec[0].filename == __file__
    assert "v" + REMOVED_IN in str(rec[0].message)
    assert "use 'n_boot'" in str(rec[0].message)


def test_converted_value():
    with pytest.warns(DeprecationWarning, match="alpha_ci="):
        _, _, alpha = _boot(1, confidence=0.9)
    assert alpha == pytest.approx(0.1)


def test_both_names_raise():
    with pytest.raises(ValueError, match="pass 'n_boot' only"):
        _boot(1, B=5, n_boot=6)


def test_method_with_shape_wrapper_and_classmethod():
    with pytest.warns(DeprecationWarning) as rec:
        assert _Model().sf(t=[[1.0, 2.0]]).shape == (1, 2)
    assert rec[0].filename == __file__
    with pytest.warns(DeprecationWarning) as rec:
        assert _Model.draw(3, seed=1) == (3, 1)
    assert rec[0].filename == __file__
    assert "_Model.draw" in str(rec[0].message)


def test_every_deprecation_names_its_removal_release():
    # The two older deprecations, undated until 0.21, go with the renames.
    import importlib
    import sys

    import surpyval as surv

    km = surv.KaplanMeier.fit([1, 2, 3, 4, 5])
    with pytest.warns(DeprecationWarning, match="v" + REMOVED_IN):
        km.band([2, 3], n_sims=100)
    sys.modules.pop("surpyval.experimental", None)
    with pytest.warns(DeprecationWarning, match="v" + REMOVED_IN):
        importlib.import_module("surpyval.experimental")
