"""The shared parameter entries are filled into every distribution method
docstring (``surpyval.utils.docstrings``)."""

import inspect

import pytest

import surpyval
from surpyval.univariate.parametric import ParametricFitter
from surpyval.utils.docstrings import PARAMS_MARKER, fill_param_docs

DISTRIBUTIONS = sorted(
    name
    for name in surpyval.__all__
    if isinstance(getattr(surpyval, name, None), ParametricFitter)
)


@pytest.mark.parametrize("name", DISTRIBUTIONS)
def test_no_unfilled_marker(name):
    dist = getattr(surpyval, name)
    for attr in dir(type(dist)):
        doc = getattr(getattr(type(dist), attr, None), "__doc__", None)
        if isinstance(doc, str):
            assert PARAMS_MARKER not in doc, f"{name}.{attr}"


def test_filled_text_names_the_parameters():
    doc = inspect.getdoc(surpyval.Weibull.sf)
    assert "alpha : numpy array or scalar" in doc
    assert "    shape parameter for the Weibull distribution" in doc
    # the entries sit where the marker was: after x, before Returns
    assert doc.index("x : numpy") < doc.index("alpha :") < doc.index("Returns")


def test_fill_uses_only_the_class_own_entries():
    class Base:
        _param_docs = """
            a : float
                The scale.
            """

        def f(self):
            """Parameters
            ----------
            {params}
            """

    class Child(Base):
        def g(self):
            """{params}"""

    fill_param_docs(Base)
    fill_param_docs(Child)
    assert (
        "            a : float\n                The scale." in Base.f.__doc__
    )
    assert Child.g.__doc__ == "{params}"


def test_no_docstrings_is_a_no_op():
    class C:
        _param_docs = "a : float"

        def f(self):
            pass

    fill_param_docs(C)
    assert C.f.__doc__ is None
