from collections import defaultdict
from itertools import combinations
from math import isclose

import numpy as np
from numpy.typing import ArrayLike, NDArray


def score(
    x: ArrayLike,
    Z: ArrayLike | NDArray,
    c: ArrayLike,
    predict_handle,
    tie_tol: float = 1e-8,
) -> float:
    # Steps:
    # 1. Form all pairs of samples
    # 2. Omit pairs where earlier time sample is censored
    # 3. Omit pairs whose times are both equal and censored
    #    (the number of permissible pairs, n_permissible_pairs, is the
    #    number of pairs after the above omissions)
    # 4. If x_1 < x_2 and x_hat_1 < x_hat_2 => concordance += 1
    # 5. If x_1 < x_2 and x_hat_1 == x_hat_2 => concordance += 0.5
    # 6. If x_1 == x_2 and both are deaths,
    #    if x_hat_1 == x_hat_2 => concordance += 1
    #    else concordance += 0.5
    # 7. If x_1 == x_2 and only one of them are deaths (let's say x_1),
    #    if x_hat_1 < x_hat_2 => concordance += 1
    #    otherwise => concordance += 0.5
    # c-index = concordance / n_permissible_pairs

    # Correct input
    x = np.array(x, ndmin=1)
    c = np.array(c, ndmin=1)
    Z = np.array(Z, ndmin=2)

    # Package xcZ together
    ixcZ = []
    for i in range(len(x)):
        ixcZ.append((i, x[i], c[i], Z[i]))

    pairs = combinations(ixcZ, 2)

    def predict(i, x, Z):
        """Inner function to get memoised prediction if available,
        otherwise compute, memoise, and return it."""
        # Already memoised
        if memoised_predictions[i] is not None:
            return memoised_predictions[i]

        # Need to calculate it
        memoised_predictions[i] = predict_handle(x, Z)
        return memoised_predictions[i]

    memoised_predictions: dict[int, None | float] = defaultdict(lambda: None)
    concordance = 0.0
    n_permissible_pairs = 0

    for tup_1, tup_2 in pairs:
        # Get right ordering
        if tup_1[1] > tup_2[1]:
            tup_1, tup_2 = tup_2, tup_1

        # Unpack tuple
        i_1, x_1, c_1, Z_1 = tup_1
        i_2, x_2, c_2, Z_2 = tup_2

        # Omit pair if x_1 is censored and x_1 != x_2
        if c_1 == 1 and x_1 != x_2:
            continue

        # Omit pair if x_1 == x_2 and are censored
        if x_1 == x_2 and c_1 == c_2 == 1:
            continue

        n_permissible_pairs += 1

        x_hat_1 = predict(i_1, x_1, Z_1)
        x_hat_2 = predict(i_2, x_2, Z_2)

        if x_1 != x_2:
            if x_hat_1 > x_hat_2:
                concordance += 1
            elif isclose(x_hat_1, x_hat_2, abs_tol=tie_tol):
                concordance += 0.5
        elif x_1 == x_2:
            if c_1 == 0 and c_2 == 0:
                if isclose(x_hat_1, x_hat_2, abs_tol=tie_tol):
                    concordance += 1
                else:
                    concordance += 0.5
            else:
                if c_1 == 0 and x_hat_1 > x_hat_2:
                    concordance += 1
                elif c_2 == 0 and x_hat_2 > x_hat_1:
                    concordance += 1
                else:
                    concordance += 0.5

    return concordance / n_permissible_pairs

    # If you want to debug:
    # return {
    #     "c_index": concordance / n_permissible_pairs,
    #     "n_concordant_pairs": n_concordant_pairs,
    #     "n_discordant_pairs": n_discordant_pairs,
    #     "n_tied_predictions": n_tied_predictions,
    #     "n_tied_time_samples": n_tied_time_samples,
    #     "concordance": concordance,
    #     "n_permissible_pairs": n_permissible_pairs,
    # }
