"""
Contains to and from transformations between surpyval and scikit-survival's
input forms ('xZc' and 'Xy' respectively).
"""

import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame


def surv_xZc_to_sksurv_Xy(
    x: NDArray, Z: NDArray, c: NDArray
) -> tuple[NDArray, NDArray]:
    """Transforms surpyval's xZc format to scikit-survival's Xy format."""
    X = Z
    y = np.array(
        list(zip(np.logical_not(c), x)),
        dtype=[("Status", bool), ("Survival", "<f8")],
    )
    return X, y


def sksurv_Xy_to_surv_xZc(X: DataFrame, y: NDArray):
    return y["time"], X.to_numpy(), np.logical_not(y["cens"]).astype(int)
