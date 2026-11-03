"""A weight is not a count under Efron's ties: the worked example in
docs/Weighted Data.rst.

Two subjects, A and B, at equal risk (the partial likelihood at beta = 0);
A dies at time 1, B is censored at 2. A as one row with weight 3 is one
death, against a risk set of weight 4: ``-3 log 4``. A as three identical
rows (or one row with count ``n = 3``) is three tied deaths, and Efron's
correction shrinks the risk set as each is processed: ``-log 4 - log 3 -
log 2``. Under Breslow's ties both are ``-3 log 4``.

R's ``coxph`` gives the same (survival 3.8.12, R 4.6.2):

.. code-block:: r

    one <- data.frame(t = c(1, 2), s = c(1, 0), z = c(1, 0), w = c(3, 1))
    three <- data.frame(t = c(1, 1, 1, 2), s = c(1, 1, 1, 0),
                        z = c(1, 1, 1, 0))
    coxph(Surv(t, s) ~ z, data = one, weights = w, ties = "efron",
          init = 0, control = coxph.control(iter.max = 0))$loglik[1]
    # -4.158883083359671; three copies -3.178053830347945; with
    # ties = "breslow" both -4.158883083359671
"""

import warnings

import numpy as np
import pytest

import surpyval as sp

R_EFRON_WEIGHT_3 = -4.158883083359671
R_EFRON_COPIES = -3.178053830347945
R_BRESLOW = -4.158883083359671


def _loglik_at_zero(**kw):
    with warnings.catch_warnings():
        # Two subjects: few effective units, and A's death separates
        warnings.simplefilter("ignore")
        model = sp.CoxPH.fit(**kw)
    return -float(model.neg_ll_of(np.zeros(1)))


ONE = dict(x=[1.0, 2.0], Z=[[1.0], [0.0]], c=[0, 1])
COPIES = dict(x=[1.0, 1.0, 1.0, 2.0], Z=[[1.0]] * 3 + [[0.0]], c=[0, 0, 0, 1])


def test_the_numbers_in_closed_form():
    assert R_EFRON_WEIGHT_3 == pytest.approx(-3 * np.log(4), rel=1e-15)
    assert R_EFRON_COPIES == pytest.approx(
        -np.log(4) - np.log(3) - np.log(2), rel=1e-15
    )
    assert round(R_EFRON_WEIGHT_3, 2) == -4.16
    assert round(R_EFRON_COPIES, 2) == -3.18


def test_efron_a_weight_of_3_is_not_three_copies():
    weighted = _loglik_at_zero(**ONE, w=[3.0, 1.0])
    np.testing.assert_allclose(weighted, R_EFRON_WEIGHT_3, rtol=1e-12)
    copies = _loglik_at_zero(**COPIES)
    np.testing.assert_allclose(copies, R_EFRON_COPIES, rtol=1e-12)
    # A count is repetition: n = 3 is the three copies
    counted = _loglik_at_zero(**ONE, n=[3, 1])
    np.testing.assert_allclose(counted, R_EFRON_COPIES, rtol=1e-12)
    # So are three copies of weight 1
    np.testing.assert_allclose(
        _loglik_at_zero(**COPIES, w=np.ones(4)), R_EFRON_COPIES, rtol=1e-12
    )


def test_breslow_a_weight_of_3_is_three_copies():
    kw = dict(tie_method="breslow")
    for value in (
        _loglik_at_zero(**ONE, w=[3.0, 1.0], **kw),
        _loglik_at_zero(**COPIES, **kw),
        _loglik_at_zero(**ONE, n=[3, 1], **kw),
    ):
        np.testing.assert_allclose(value, R_BRESLOW, rtol=1e-12)
