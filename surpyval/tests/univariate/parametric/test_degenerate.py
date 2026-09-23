"""The degenerate distributions: InstantlyOccurs (point mass at 0) and
NeverOccurs (point mass at +inf).

Previously these lived in ``parametric/__init__.py`` with a partial API
(no df/hf/qf/mean) and no serialisation; they are now first-class
members of the distributions package.
"""

import numpy as np

import surpyval
from surpyval import InstantlyOccurs, NeverOccurs

X = np.array([0.0, 1.0, 5.0, 100.0])


def test_never_occurs_api():
    assert (NeverOccurs.sf(X) == 1.0).all()
    assert (NeverOccurs.ff(X) == 0.0).all()
    assert (NeverOccurs.df(X) == 0.0).all()
    assert (NeverOccurs.hf(X) == 0.0).all()
    assert (NeverOccurs.Hf(X) == 0.0).all()
    assert (NeverOccurs.qf(np.array([0.1, 0.5, 0.99])) == np.inf).all()
    assert NeverOccurs.mean() == np.inf
    assert (NeverOccurs.random(5) == np.inf).all()


def test_instantly_occurs_api():
    assert (InstantlyOccurs.sf(X) == 0.0).all()
    assert (InstantlyOccurs.ff(X) == 1.0).all()
    assert (InstantlyOccurs.hf(X) == np.inf).all()
    assert (InstantlyOccurs.Hf(X) == np.inf).all()
    df = InstantlyOccurs.df(X)
    assert df[0] == np.inf and (df[1:] == 0.0).all()
    assert (InstantlyOccurs.qf(np.array([0.1, 0.9])) == 0.0).all()
    assert InstantlyOccurs.mean() == 0.0
    assert (InstantlyOccurs.random(5) == 0.0).all()


def test_degenerate_serialisation_round_trip():
    # Stateless: the class is the model, so from_dict returns the class
    # itself and identity is preserved (the survival-tree leaves rely on
    # ``model is NeverOccurs``).
    for cls in (NeverOccurs, InstantlyOccurs):
        d = cls.to_dict()
        assert d["model"] == cls.name
        assert d["schema"] == 1
        assert surpyval.from_dict(d) is cls


def test_import_paths_preserved():
    # The historical import locations must keep working, and must resolve
    # to the same class objects.
    from surpyval.univariate.parametric import (
        InstantlyOccurs as from_parametric_i,
    )
    from surpyval.univariate.parametric import NeverOccurs as from_parametric_n
    from surpyval.univariate.parametric.distributions.degenerate import (
        NeverOccurs as from_module_n,
    )

    assert from_parametric_n is NeverOccurs is from_module_n
    assert from_parametric_i is InstantlyOccurs


def test_exact_event_time_refuses_density_and_hazard():
    # A point mass has no density: all of its probability sits at T, so
    # the density is a Dirac delta rather than a function of x. These
    # used to return inf at T (and, for hf, at every x after it), which
    # integrates to inf rather than 1 and propagates silently into
    # whatever consumes it.
    import pytest

    from surpyval import ExactEventTime

    x = np.array([4.0, 5.0, 6.0])
    with pytest.raises(NotImplementedError, match="no density"):
        ExactEventTime.df(x, 5.0)
    with pytest.raises(NotImplementedError, match="no hazard rate"):
        ExactEventTime.hf(x, 5.0)
    # log_df is inherited and reaches hf, so it refuses too rather than
    # returning the nan that log(inf) - inf used to give.
    with pytest.raises(NotImplementedError):
        ExactEventTime.log_df(x, 5.0)


def test_exact_event_time_keeps_the_functions_that_are_defined():
    # sf, ff and Hf are genuine step functions and are unaffected.
    from surpyval import ExactEventTime

    x = np.array([4.0, 4.999, 5.0, 6.0])
    np.testing.assert_array_equal(ExactEventTime.sf(x, 5.0), [1, 1, 0, 0])
    np.testing.assert_array_equal(ExactEventTime.ff(x, 5.0), [0, 0, 1, 1])
    Hf = np.asarray(ExactEventTime.Hf(x, 5.0), dtype=float)
    np.testing.assert_array_equal(Hf, [0.0, 0.0, np.inf, np.inf])
    # Hf is -log R(x), and used to be an alias for hf that happened to
    # take the same two values.
    with np.errstate(divide="ignore"):
        expected = -np.log(np.asarray(ExactEventTime.sf(x, 5.0), dtype=float))
    np.testing.assert_array_equal(Hf, expected)


def test_exact_event_time_still_fits_and_serialises():
    # The estimator brackets T between the censoring bounds and never
    # touches a density, so refusing df and hf cannot affect it.
    from surpyval import ExactEventTime

    model = ExactEventTime.fit(x=[1.0, 2.0, 8.0, 9.0], c=[1, 1, -1, -1])
    np.testing.assert_allclose(model.params, [5.0])
    restored = surpyval.from_dict(model.to_dict())
    np.testing.assert_allclose(restored.params, model.params)
