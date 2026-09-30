"""A coefficient the data cannot determine is reported, not invented
(principle 12; #476).

A covariate column that adds nothing to the others leaves the likelihood
(or the estimating equations) flat along some combination of the
coefficients, so a fit that returns a value for it returns wherever its
search stopped. As R's ``coxph`` and ``lm`` do, such a coefficient is
*aliased*: the fit runs without the column, reports its coefficient as
``nan``, lists the column in ``model.aliased``, predicts as though the
coefficient were 0 and warns once, naming it. For every registered model
with covariates and coefficients (``Case.coefficients``):

- **aliasing**: the fixture refitted with a copy of its last covariate
  column appended gives exactly one warning naming the new column, a
  ``nan`` coefficient for it (in every cause's fit) and ``aliased`` equal
  to it, and the other coefficients and every function of the model --
  queried with the new column set to arbitrary values -- equal to the fit
  without it;
- **aliasing_constant**: the same with a constant column, for the models
  with an intercept that absorbs it (``Case.intercept``).

The fit without the column is the fixture's fit, so the tolerance is
tight: the aliased fit solves the same problem from the same start.
"""

import numpy as np
import pytest

from surpyval.tests.conformance.leaks import deliberate, quiet
from surpyval.tests.conformance.registry import cases_for, fitted, predictions

RTOL = 1e-6


def _values(size):
    # Arbitrary values for the aliased column in a query: its coefficient
    # is taken as 0, so they must not matter.
    return np.linspace(-3.0, 5.0, size)


def _check(case, extra):
    assert case.coefficients is not None, (
        "a model with covariates declares its coefficients "
        "(Case.coefficients), or excludes 'aliasing' with the reason"
    )
    ref = fitted(case)
    data = case.data()
    Z = np.asarray(data[case.covariates], dtype=float)
    p = Z.shape[1]
    # (A raw numerical warning is passed on to the leak check.)
    with deliberate() as caught:
        model = case.fit({**data, case.covariates: np.c_[Z, extra(Z)]})
    found = [text for _, text in caught if "cannot be estimated" in text]
    assert len(found) == 1, found
    assert found[0].startswith(
        "Covariate column(s) {} of Z cannot be estimated".format(p)
    ), found[0]

    coef = np.atleast_2d(np.asarray(case.coefficients(model), dtype=float))
    want = np.atleast_2d(np.asarray(case.coefficients(ref), dtype=float))
    assert coef.shape[-1] == p + 1
    assert np.isnan(coef[:, p]).all(), coef
    np.testing.assert_allclose(coef[:, :p], want, rtol=RTOL)
    np.testing.assert_array_equal(model.aliased, [p])

    Zq = np.asarray(case.Z, dtype=float)
    with quiet():
        got = predictions(case, model, Z=np.c_[Zq, _values(Zq.shape[0])])
        expected = predictions(case, ref)
    assert got.keys() == expected.keys()
    for key in expected:
        np.testing.assert_allclose(
            got[key], expected[key], rtol=RTOL, atol=1e-12, err_msg=key
        )


@pytest.mark.parametrize("case", cases_for("aliasing"))
def test_repeated_column_is_aliased(case):
    _check(case, lambda Z: Z[:, -1])


@pytest.mark.parametrize("case", cases_for("aliasing_constant"))
def test_constant_column_is_aliased(case):
    _check(case, lambda Z: np.ones(Z.shape[0]))
