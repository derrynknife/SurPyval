"""The starting point of an offset maximum-likelihood fit (#622)."""

import warnings

import numpy as np

import surpyval as sp


def test_622_offset_interval_fit_in_a_lower_tail_window_is_searchable():
    # A start far below the data, with a tiny shape, puts every interval
    # in the distribution's lower tail. Its stand-in value was taken with
    # ``float`` of the bounds less the offset, which the search traces:
    # TypeError: float() argument must be ... not 'ArrayBox'.
    xl = np.array([7.3, 8.0, 10.2, 11.7, 12.4, 13.1, 13.9])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sp.LogNormal.fit(
            xl=xl, xr=xl + 0.7, offset=True, init=[-6.4e9, 22.6, 4e-10]
        )
    assert np.isfinite(model.neg_ll())
    assert model.gamma < xl.min()
