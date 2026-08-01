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
